import pytest
import torch

from packboost.core import PackBoost


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA capable runtime"
)

def test_encode_cuts_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    torch.manual_seed(0)
    N, F = 10_000, 2376  # deliberately not multiples of 32 to exercise padding
    X_cpu = torch.randint(0, 5, (N, F), dtype=torch.int8)

    ref = pack_cpu.encode_cuts(X_cpu)
    result = pack_gpu.encode_cuts(X_cpu.to("cuda"))
    torch.cuda.synchronize()

    torch.testing.assert_close(result.cpu(), ref)

def test_et_sample_1b_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")
    pack_cpu.nfeatsets = pack_gpu.nfeatsets = 32

    torch.manual_seed(1)
    bF, M, rounds = 2376*4, 10_000//32, 3

    X_cpu = torch.randint(0, 2 ** 32, (bF, M), dtype=torch.int64).to(torch.uint32)
    Fsch = torch.randint(0, bF, (rounds, 32 * pack_cpu.nfeatsets), dtype=torch.int16).to(
        torch.uint16
    )
    round_idx = 2

    ref = pack_cpu.et_sample_1b(X_cpu, Fsch, round_idx)
    result = pack_gpu.et_sample_1b(X_cpu.to("cuda"), Fsch.to("cuda"), round_idx)
    torch.cuda.synchronize()

    torch.testing.assert_close(result.cpu(), ref)

def test_prep_vars_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    torch.manual_seed(0)

    # (K=nfolds, Dm=depth-bits, N=samples)
    cases = [
        (1, 3, 1),                         # tiny, single-bitpack sanity
        (2, 5, 33),                        # N just over a warp
        (8, 7, 32 * 512 - 1),              # just under a full grid tile
        (8, 7, 32 * 512),                  # exact full grid tile
        (8, 7, 32 * 512 + 1),              # just over a full grid tile
        (16, 10, 100_003),                 # bigger K + near-max Dm
    ]

    lo, hi = -(1 << 30), (1 << 30) - 1   # Q30-ish range

    for K, Dm, N in cases:
        L = torch.randint(0, 4, (K, Dm, N), dtype=torch.uint8)
        Y = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
        P = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)

        ref_LE, ref_G = pack_cpu.prep_vars(L, Y, P)
        out_LE, out_G = pack_gpu.prep_vars(L.cuda(), Y.cuda(), P.cuda())
        torch.cuda.synchronize()

        assert out_LE.dtype == ref_LE.dtype, f"dtype mismatch: {out_LE.dtype} vs {ref_LE.dtype}"
        torch.testing.assert_close(out_LE.cpu(), ref_LE, msg=f"LE mismatch K={K} Dm={Dm} N={N}")
        torch.testing.assert_close(out_G.cpu(),  ref_G,  msg=f"G mismatch  K={K} Dm={Dm} N={N}")


def test_h0_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    torch.manual_seed(123)

    # (K=nfolds, max_depth, N). Keep max_depth <= 8 (SMEM bound).
    cases = [
        (1, 3, 1),                 # tiny sanity
        (2, 5, 33),                # N just over a warp
        (8, 7, 32 * 512 + 1),      # just over a full grid tile
        (4, 8, 100_003),           # D=8 boundary, large N
    ]

    # Q30-ish range for Y/P, to reuse prep_vars path
    lo, hi = -(1 << 30), (1 << 30) - 1

    for K, D, N in cases:
        # Build inputs for prep_vars so LE/G match Murky packing exactly
        L = torch.randint(0, 4, (K, D, N), dtype=torch.uint8)     # bits per depth
        Y = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
        P = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)

        # CPU prep_vars → reference LE, G (signed storage OK)
        ref_LE, ref_G = pack_cpu.prep_vars(L, Y, P)

        # Also create an unsigned-storage variant to exercise launcher branches
        bits = D * (D - 1) // 2
        if D <= 6:
            LE_unsigned = ref_LE.to(torch.uint16)
        elif D <= 8:
            LE_unsigned = ref_LE.to(torch.uint32)
        else:
            LE_unsigned = ref_LE.to(torch.uint64)

        for use_unsigned in (True,):
            LE_host = LE_unsigned if use_unsigned else ref_LE
            G_host  = ref_G

            W_host = torch.ones(N, dtype=torch.int16)

            # CPU reference
            H0_ref = pack_cpu.h0(G_host, W_host, LE_host, D)

            # GPU result
            H0_out = pack_gpu.h0(G_host.cuda(), W_host.cuda(), LE_host.cuda(), D)
            torch.cuda.synchronize()

            # Shape check: [K, 2^D, 2]
            nodes = 1 << D
            assert H0_out.shape == (K, nodes, 2)
            assert H0_ref.shape == (K, nodes, 2)

            # Last row (leaf plane) must be zero by construction
            assert torch.count_nonzero(H0_out[:, nodes - 1, :]).item() == 0
            assert torch.count_nonzero(H0_ref[:, nodes - 1, :]).item() == 0

            # Exact equality
            torch.testing.assert_close(H0_out.cpu(), H0_ref, rtol=0, atol=0,
                                       msg=f"H0 mismatch K={K} D={D} N={N} (unsigned={use_unsigned})")


def test_repack_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    torch.manual_seed(7)

    # (K=nfolds, D=max_depth, N=samples, nfeatsets, nsets)
    cases = [
        (1, 3, 1,                7,   2),          # tiny, nfeatsets not multiple of 8
        (2, 5, 33,               17,  3),          # N just over a warp; tail nfeatsets
        (8, 7, 32 * 512 - 1,     32,  3),          # just under full grid tile
        (8, 7, 32 * 512 + 5,     63,  4),          # over tile + tail nfeatsets (63)
        (16, 8, 100_003,         40,  2),          # D=8 boundary; larger K
    ]

    for K, D, N, nfeatsets, nsets in cases:
        # Build LE via prep_vars to ensure Murky-identical bit packing

        L = torch.empty((K, D, N), dtype=torch.uint8)
        for d in range(1, D+1):
            L[:, d-1].random_(0, 1 << d)  # valid range: 0 .. (2^d - 1)

        Y = torch.zeros(N, dtype=torch.int32)
        P = torch.zeros(N, dtype=torch.int32)
        LE, _ = pack_cpu.prep_vars(L, Y, P)  # dtype auto: u16/u32/u64 based on D

        # Build FST: [nsets, nfeatsets, D], each depth is a shuffled tiling of [0..K-1]
        base = torch.arange(K, dtype=torch.uint8)
        rep = (nfeatsets + K - 1) // K
        row = base.repeat(rep)[:nfeatsets]
        FST = torch.empty((nsets, nfeatsets, D), dtype=torch.uint8)
        for s in range(nsets):
            for d in range(D):
                perm = torch.randperm(nfeatsets)
                FST[s, :, d] = row[perm]

        tree_set = (3 * K + 5) % nsets  # deterministic, within range

        # CPU reference
        LF_ref = pack_cpu.repack(FST, LE, tree_set)

        # GPU output
        LF_out = pack_gpu.repack(FST.cuda(), LE.cuda(), tree_set)
        torch.cuda.synchronize()

        # Checks
        assert LF_out.dtype == LF_ref.dtype, f"dtype mismatch: {LF_out.dtype} vs {LF_ref.dtype}"
        assert LF_out.shape == LF_ref.shape == (nfeatsets, N)
        torch.testing.assert_close(
            LF_out.cpu(), LF_ref, rtol=0, atol=0,
            msg=f"LF mismatch K={K} D={D} N={N} nfeatsets={nfeatsets} nsets={nsets}"
        )


def test_h_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    torch.manual_seed(42)

    # (K=nfolds, D=max_depth (<=7), N=samples, nfeatsets)
    cases = [
        (1, 3, 1,                 7),    # tiny sanity; tail nfeatsets
        (2, 5, 33,               17),    # N just over a warp; tail nfeatsets
        (8, 7, 32 * 512 - 1,     32),    # just under full grid tile
        (8, 7, 32 * 512 + 5,     63),    # over tile + tail nfeatsets
        (16, 7, 100_003,         40),    # large N, varied nfeatsets
    ]

    # Q30-ish range to match prior conventions; we'll use G (from prep_vars) as Y for H.
    lo, hi = -(1 << 30), (1 << 30) - 1

    for K, D, N, nfeatsets in cases:
        assert D <= 7, "GPU H shared-mem variant is defined up to D=7"

        # --- Build LE and G via prep_vars so packing matches Murky exactly ---
        # L shape: [K, D-1, N] (depth bits for d=1..D-1)
        L_bits = torch.randint(0, 4, (K, max(D - 1, 0), N), dtype=torch.uint8)
        Y_raw  = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
        P_raw  = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)

        LE_ref, G_ref = pack_cpu.prep_vars(L_bits, Y_raw, P_raw)  # LE dtype auto (u16/u32/u64)

        # --- Build LF via repack (Murky FST wiring) ---
        # FST: [nsets, nfeatsets, D], each depth is a shuffled tiling of [0..K-1]
        nsets = 3
        base = torch.arange(K, dtype=torch.uint8)
        rep  = (nfeatsets + K - 1) // K
        row  = base.repeat(rep)[:nfeatsets]
        FST  = torch.empty((nsets, nfeatsets, D), dtype=torch.uint8)
        for s in range(nsets):
            for d in range(D):
                perm = torch.randperm(nfeatsets)
                FST[s, :, d] = row[perm]
        tree_set = (3 * K + 5) % nsets

        LF_ref = pack_cpu.repack(FST, LE_ref, tree_set)  # [nfeatsets, N], dtype(u16/u32/u64)

        # --- Make XS with enough columns to cover N samples (N <= 32*M) ---
        M = (N + 31) // 32
        XS_cpu = torch.randint(
            0, 2 ** 32, (nfeatsets, 32 * M), dtype=torch.int64
        ).to(torch.uint32)

        # --- CPU reference ---
        Y32_cpu = G_ref#.to(torch.int32)                 # H expects int32 Y
        W_cpu   = torch.ones(N, dtype=torch.int16)
        H_ref   = pack_cpu.H(XS_cpu, Y32_cpu, W_cpu, LF_ref, D)

        # --- GPU output ---
        XS_gpu  = XS_cpu.cuda()
        Y32_gpu = Y32_cpu.cuda()
        W_gpu   = W_cpu.cuda()
        LF_gpu  = LF_ref.cuda()

        H_out = pack_gpu.H(XS_gpu, Y32_gpu, W_gpu, LF_gpu, D)
        torch.cuda.synchronize()

        # --- Checks ---
        nodes_tot = (1 << D) - 1
        assert H_out.shape == (nfeatsets, nodes_tot, 2, 32)
        assert H_ref.shape == (nfeatsets, nodes_tot, 2, 32)

        # Exact equality
        torch.testing.assert_close(
            H_out.cpu(), H_ref, rtol=0, atol=0,
            msg=f"H mismatch K={K} D={D} N={N} nfeatsets={nfeatsets}"
        )

def test_cut_matches_cpu_reference():
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    torch.manual_seed(1234)

    cases = [
        (1, 3, 1,                 7,   2),
        (4, 5, 33,                17,  3),
        (8, 7, 32 * 256 + 3,      32,  3),
    ]

    base_L2 = 10_000.0
    base_lr = 0.01
    qgrad_bits = 12

    for K, D, N, nfeatsets, rounds in cases:
        pack_cpu.nfeatsets = pack_gpu.nfeatsets = nfeatsets

        # Valid-by-depth random L bits
        L_bits = torch.empty((K, max(D - 1, 0), N), dtype=torch.uint8)
        for d in range(1, D):
            L_bits[:, d-1].random_(0, 1 << d)

        lo, hi = -(1 << 30), (1 << 30) - 1
        Y_raw  = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
        P_raw  = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)

        LE_ref, G_ref = pack_cpu.prep_vars(L_bits, Y_raw, P_raw)

        # FST depth-wise shuffled tiling
        FST = torch.empty((rounds, nfeatsets, D), dtype=torch.uint8)
        base = torch.arange(K, dtype=torch.uint8)
        rep  = (nfeatsets + K - 1) // K
        row  = base.repeat(rep)[:nfeatsets]
        for s in range(rounds):
            for d in range(D):
                perm = torch.randperm(nfeatsets)
                FST[s, :, d] = row[perm]

        tree_set = min(1, rounds - 1)

        LF_ref = pack_cpu.repack(FST, LE_ref, tree_set)

        M = (N + 31) // 32
        XS_cpu = torch.randint(0, 2 ** 32, (nfeatsets, 32 * M), dtype=torch.int64).to(torch.uint32)

        W_ref = torch.ones(N, dtype=torch.int16)
        H_ref   = pack_cpu.H(XS_cpu, G_ref, W_ref, LF_ref, D)
        H0_full = pack_cpu.h0(G_ref, W_ref, LE_ref, D)
        H0_ref  = H0_full[:, : (1 << D) - 1, :]

        F_row_elems = 32 * nfeatsets
        # Generate as uint16 directly
        F = torch.randint(0, 1 << 16, (rounds, F_row_elems), dtype=torch.int64).to(torch.uint16)

        V_cpu = torch.zeros((rounds, K, 2 * ((1 << D) - 1)), dtype=torch.int32)
        I_cpu = torch.zeros((rounds, K,     ((1 << D) - 1)), dtype=torch.uint16)

        lr_eff = base_lr / float(K)

        pack_cpu.cut(F, FST, H_ref, H0_ref, V_cpu, I_cpu,
                     tree_set=tree_set, L2=base_L2, lr=lr_eff,
                     qgrad_bits=qgrad_bits, max_depth=D)

        # GPU
        F_gpu   = F.cuda()
        FST_gpu = FST.cuda()
        H_gpu   = H_ref.cuda()
        H0_gpu  = H0_ref.cuda()
        V_gpu   = torch.zeros_like(V_cpu, device="cuda")
        I_gpu   = torch.zeros_like(I_cpu, device="cuda")

        pack_gpu.cut(F_gpu, FST_gpu, H_gpu, H0_gpu, V_gpu, I_gpu,
                     tree_set=tree_set, L2=base_L2, lr=lr_eff,
                     qgrad_bits=qgrad_bits, max_depth=D)
        torch.cuda.synchronize()

        torch.testing.assert_close(V_gpu.cpu(), V_cpu, rtol=0, atol=0)
        torch.testing.assert_close(I_gpu.cpu(), I_cpu, rtol=0, atol=0)


def test_advance_and_predict_matches_cpu_reference():
    import pytest, torch
    from packboost.core import PackBoost
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

    torch.manual_seed(2025)

    cases = [
        (1, 3,  1,         4096, 2),
        (4, 5,  33,        8192, 3),
        (8, 8,  32*256+7,  65536, 3),
    ]

    for K0, D, N, R, rounds in cases:
        nodes = (1 << D) - 1
        Dm = D - 1
        M = (N + 31) // 32

        L_dtype = (torch.uint8 if D <= 8 else torch.int16)

        P_cpu = torch.zeros(N, dtype=torch.int32)
        P_gpu = P_cpu.cuda()

        X_cpu = torch.randint(0, 2**32, (R, M), dtype=torch.int64).to(torch.uint32)
        X_gpu = X_cpu.cuda()

        L_old_cpu = torch.zeros((K0, Dm, N), dtype=L_dtype)
        L_new_cpu = torch.zeros_like(L_old_cpu)
        L_old_gpu = L_old_cpu.cuda()
        L_new_gpu = L_new_cpu.cuda()

        V_cpu = torch.randint(-(1<<15), 1<<15, (rounds, K0, 2*nodes), dtype=torch.int32)
        V_gpu = V_cpu.cuda()

        I_vals = torch.randint(0, R, (rounds, K0, nodes), dtype=torch.int32) & 0xFFFF
        I_cpu  = I_vals.to(torch.uint16)
        I_gpu  = I_cpu.cuda()

        tree_set = min(1, rounds - 1)

        pack_cpu = PackBoost(device="cpu")
        pack_gpu = PackBoost(device="cuda")

        P_ref, Ln_ref = pack_cpu.advance_and_predict(
            P_cpu.clone(), X_cpu, L_old_cpu.clone(), L_new_cpu.clone(),
            V_cpu, I_cpu, tree_set
        )

        P_out, Ln_out = pack_gpu.advance_and_predict(
            P_gpu.clone(), X_gpu, L_old_gpu.clone(), L_new_gpu.clone(),
            V_gpu, I_gpu, tree_set
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(P_out.cpu(), P_ref, rtol=0, atol=0)
        torch.testing.assert_close(Ln_out.cpu(), Ln_ref, rtol=0, atol=0)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA capable runtime"
)

# ---------- small helpers ----------

def _mk_era_ends(N, E):
    # X-space eras that sum to N (no 32-alignment)
    base, r = divmod(N, E)
    ends = []
    s = 0
    for i in range(E):
        s += base + (1 if i < r else 0)
        ends.append(s)
    return torch.tensor(ends, dtype=torch.int32)


def _norm_H0_layout(t, K0, E, nodes_all):
    s = tuple(t.shape)
    if s == (K0, nodes_all, E, 2):
        return t.contiguous()
    if s == (K0, E, nodes_all, 2):
        return t.permute(0, 2, 1, 3).contiguous()
    if s == (K0, nodes_all, 2, E):
        return t.permute(0, 1, 3, 2).contiguous()
    raise AssertionError(f"Unexpected H0 shape {s}")

def _norm_He_layout(He, K1, E, nodes):
    # Accept either [K1, E, nodes, 2, 32] or [K1, nodes, E, 2, 32], return [K1, E, nodes, 2, 32]
    if He.shape == (K1, E, nodes, 2, 32):
        return He
    if He.shape == (K1, nodes, E, 2, 32):
        return He.permute(0, 2, 1, 3, 4).contiguous()
    raise AssertionError(f"Unexpected He shape: {tuple(He.shape)}")

def test_h0_des_matches_per_era_baseline_tail():
    torch.manual_seed(2025)
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    K0, D, E = 4, 6, 4
    nodes_all = 1 << D
    N = 32 * 64 + 13                      # NOT a multiple of 32
    era_ends = _mk_era_ends(N, E)         # X-space

    # build LE,G in X-space
    L_bits = torch.empty((K0, D, N), dtype=torch.uint8)
    for d in range(1, D + 1):
        L_bits[:, d - 1].random_(0, 1 << d)
    lo, hi = -(1 << 30), (1 << 30) - 1
    Y = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    P = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    LE, G = pack_cpu.prep_vars(L_bits, Y, P)

    W = torch.ones(N, dtype=torch.int16)
    H0_out = pack_gpu.h0_des(G.cuda(), W.cuda(), LE.cuda(), D, era_ends.cuda())
    #H0_out = _norm_H0_layout(H0_out, K0, E, nodes_all)

    print(H0_out.shape)

    # CPU per-era baseline
    s = 0; H0_refs = []
    for e in era_ends.tolist():
        H0_e = pack_cpu.h0(G[s:e], W[s:e], LE[:, s:e], D)
        H0_refs.append(H0_e.unsqueeze(2))
        s = e
    H0_ref = torch.cat(H0_refs, dim=2).contiguous()
    print(H0_ref.shape)
    torch.testing.assert_close(H0_out.cpu(), H0_ref, rtol=0, atol=0)



# ---------- 2) h_des vs per-era baseline ----------
def test_h_des_matches_per_era_baseline():
    torch.manual_seed(7)
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    # Config
    K0, K1, D, E = 4, 8, 6, 4
    nodes = (1 << D) - 1

    # N must be multiple of 32 and eras sized to multiples of 32
    N = 32 * 64  # 2048
    era_ends = _mk_era_ends(N, E)

    # Make LE, G
    L_bits = torch.empty((K0, D, N), dtype=torch.uint8)
    for d in range(1, D + 1):
        L_bits[:, d - 1].random_(0, 1 << d)
    lo, hi = -(1 << 30), (1 << 30) - 1
    Y = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    P = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    LE, G = pack_cpu.prep_vars(L_bits, Y, P)  # G int16

    # Repack for a single tree_set (like training)
    rounds = 2
    # FST: [rounds, nfeatsets=K1, D] with depth-wise shuffled tiling of [0..K0-1]
    base = torch.arange(K0, dtype=torch.uint8)
    rep = (K1 + K0 - 1) // K0
    row = base.repeat(rep)[:K1]
    FST = torch.empty((rounds, K1, D), dtype=torch.uint8)
    for s in range(rounds):
        for d in range(D):
            perm = torch.randperm(K1)
            FST[s, :, d] = row[perm]
    tree_set = 0
    LF = pack_cpu.repack(FST, LE, tree_set)  # [K1, N]

    # XS: [K1, N] uint32 bitmasks per column
    XS = torch.randint(0, 2**32, (K1, N), dtype=torch.int64).to(torch.uint32)

    # --- GPU DES h ---
    W = torch.ones(N, dtype=torch.int16)
    He_out = pack_gpu.h_des(XS.cuda(), G.cuda(), W.cuda(), LF.cuda(), D, era_ends.cuda())
    torch.cuda.synchronize()
    He_out = _norm_He_layout(He_out, K1, E, nodes)

    # --- CPU per-era baseline via non-DES H on each era and stack ---
    He_refs = []
    s = 0
    for e in era_ends.tolist():
        H_e = pack_cpu.H(XS[:, s:e], G[s:e], W[s:e], LF[:, s:e], D)  # [K1, nodes, 2, 32]
        He_refs.append(H_e.unsqueeze(1))                     # [K1, 1, nodes, 2, 32]
        s = e
    He_ref = torch.cat(He_refs, dim=1)                       # [K1, E, nodes, 2, 32]

    torch.testing.assert_close(He_out.cpu(), He_ref, rtol=0, atol=0)
# ---------- 3) cut_des (CPU vs CUDA) using DES-built H/H0 ----------
# ---------- 3) cut_des (CPU vs CUDA) using DES-built H/H0 ----------
def test_cut_des_matches_cpu_reference():
    torch.manual_seed(11)
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    K0, K1, D, E = 4, 8, 6, 4
    nodes_all = 1 << D
    nodes = nodes_all - 1
    rounds = 2
    N = 32 * 64
    era_ends = _mk_era_ends(N, E)

    # Build LE, G
    L_bits = torch.empty((K0, D, N), dtype=torch.uint8)
    for d in range(1, D + 1):
        L_bits[:, d - 1].random_(0, 1 << d)
    lo, hi = -(1 << 30), (1 << 30) - 1
    Y = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    P = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    LE, G = pack_cpu.prep_vars(L_bits, Y, P)

    # FST & LF
    base = torch.arange(K0, dtype=torch.uint8)
    rep = (K1 + K0 - 1) // K0
    row = base.repeat(rep)[:K1]
    FST = torch.empty((rounds, K1, D), dtype=torch.uint8)
    for s in range(rounds):
        for d in range(D):
            perm = torch.randperm(K1)
            FST[s, :, d] = row[perm]
    tree_set = 1
    LF = pack_cpu.repack(FST, LE, tree_set)

    # XS and DES stats (GPU)
    XS = torch.randint(0, 2**32, (K1, N), dtype=torch.int64).to(torch.uint32)
    W = torch.ones(N, dtype=torch.int16)
    H0_all_gpu = pack_gpu.h0_des(G.cuda(), W.cuda(), LE.cuda(), D, era_ends.cuda())             # [K0, 2**D, E, 2]
    He_gpu     = pack_gpu.h_des  (XS.cuda(), G.cuda(), W.cuda(), LF.cuda(), D, era_ends.cuda()) # [K1, E, nodes, 2, 32]
    torch.cuda.synchronize()

    # Canonical DES layouts for selector
    H0_cpu = H0_all_gpu[:, :nodes, :, :].cpu().contiguous()  # [K0, nodes, E, 2]
    He_cpu = He_gpu.cpu().contiguous()                       # [K1, E, nodes, 2, 32]

    # F, outputs
    F = torch.randint(0, 1 << 16, (rounds, 32 * K1), dtype=torch.int64).to(torch.uint16)
    V_cpu = torch.zeros((rounds, K0, 2 * nodes), dtype=torch.int32)
    I_cpu = torch.zeros((rounds, K0,     nodes), dtype=torch.uint16)
    V_gpu = torch.zeros_like(V_cpu, device="cuda")
    I_gpu = torch.zeros_like(I_cpu, device="cuda")

    L2 = 1e4
    lr = 0.01 / float(K0)
    qbits = 12

    # CPU selector on CPU-cast DES stats
    pack_cpu.cut_des(F, FST, He_cpu, H0_cpu, V_cpu, I_cpu,
                     tree_set=tree_set, L2=L2, lr=lr, qgrad_bits=qbits, max_depth=D)

    # CUDA selector on GPU stats
    H0e_gpu = H0_all_gpu[:, :nodes, :, :].contiguous()        # [K0, nodes, E, 2]
    pack_gpu.cut_des(F.cuda(), FST.cuda(), He_gpu.contiguous(), H0e_gpu, V_gpu, I_gpu,
                     tree_set=tree_set, L2=L2, lr=lr, qgrad_bits=qbits, max_depth=D)
    torch.cuda.synchronize()

    torch.testing.assert_close(V_gpu.cpu(), V_cpu, rtol=0, atol=0)
    torch.testing.assert_close(I_gpu.cpu(), I_cpu, rtol=0, atol=0)


# ---------- 4) end-to-end DES: (h0_des, h_des) → cut_des ----------
def test_des_end_to_end_cut_agrees_with_stacked_baseline():
    torch.manual_seed(29)
    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    K0, K1, D, E = 4, 8, 6, 4
    nodes_all = 1 << D
    nodes = nodes_all - 1
    rounds = 2
    N = 32 * 64
    era_ends = _mk_era_ends(N, E)

    # Build LE, G
    L_bits = torch.empty((K0, D, N), dtype=torch.uint8)
    for d in range(1, D + 1):
        L_bits[:, d - 1].random_(0, 1 << d)
    lo, hi = -(1 << 30), (1 << 30) - 1
    Y = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    P = torch.randint(lo, hi + 1, (N,), dtype=torch.int32)
    LE, G = pack_cpu.prep_vars(L_bits, Y, P)

    # FST & LF
    base = torch.arange(K0, dtype=torch.uint8)
    rep = (K1 + K0 - 1) // K0
    row = base.repeat(rep)[:K1]
    FST = torch.empty((rounds, K1, D), dtype=torch.uint8)
    for s in range(rounds):
        for d in range(D):
            perm = torch.randperm(K1)
            FST[s, :, d] = row[perm]
    tree_set = 0
    LF = pack_cpu.repack(FST, LE, tree_set)

    # XS + DES stats via GPU
    XS = torch.randint(0, 2**32, (K1, N), dtype=torch.int64).to(torch.uint32)
    W = torch.ones(N, dtype=torch.int16)
    H0_all_gpu = pack_gpu.h0_des(G.cuda(), W.cuda(), LE.cuda(), D, era_ends.cuda())             # [K0, 2**D, E, 2]
    He_gpu     = pack_gpu.h_des  (XS.cuda(), G.cuda(), W.cuda(), LF.cuda(), D, era_ends.cuda()) # [K1, E, nodes, 2, 32]
    torch.cuda.synchronize()

    H0e_gpu = H0_all_gpu[:, :nodes, :, :].contiguous()   # [K0, nodes, E, 2]
    H0_cpu  = H0e_gpu.cpu().contiguous()
    He_cpu  = He_gpu.cpu().contiguous()

    F = torch.randint(0, 1 << 16, (rounds, 32 * K1), dtype=torch.int64).to(torch.uint16)

    V_cpu = torch.zeros((rounds, K0, 2 * nodes), dtype=torch.int32)
    I_cpu = torch.zeros((rounds, K0,     nodes), dtype=torch.uint16)
    V_gpu = torch.zeros_like(V_cpu, device="cuda")
    I_gpu = torch.zeros_like(I_cpu, device="cuda")

    L2 = 1e4
    lr = 0.01 / float(K0)
    qbits = 12

    pack_cpu.cut_des(F, FST, He_cpu, H0_cpu, V_cpu, I_cpu,
                     tree_set=tree_set, L2=L2, lr=lr, qgrad_bits=qbits, max_depth=D)

    pack_gpu.cut_des(F.cuda(), FST.cuda(), He_gpu.contiguous(), H0e_gpu, V_gpu, I_gpu,
                     tree_set=tree_set, L2=L2, lr=lr, qgrad_bits=qbits, max_depth=D)
    torch.cuda.synchronize()

    torch.testing.assert_close(V_gpu.cpu(), V_cpu, rtol=0, atol=0)
    torch.testing.assert_close(I_gpu.cpu(), I_cpu, rtol=0, atol=0)
