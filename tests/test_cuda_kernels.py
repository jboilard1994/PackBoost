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

            # CPU reference
            H0_ref = pack_cpu.h0(G_host, LE_host, D)

            # GPU result
            H0_out = pack_gpu.h0(G_host.cuda(), LE_host.cuda(), D)
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
        L = torch.randint(0, 4, (K, D - 1, N), dtype=torch.uint8)
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
