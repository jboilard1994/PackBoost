import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor
import torch.nn.functional as Fn
print('Installing kernels...')
from packboost.cuda import kernels
print('kernels successfully Installed!')


class PackBoost(BaseEstimator, RegressorMixin):
    def __init__(self, device='cuda'):
        self.device = device
        self.nfeatsets = 32

    def fit(self,
            X: np.ndarray,         # int8 [N, F]
            y: np.ndarray,         # float32 [N]
            Xv: np.ndarray = None, # int8  [Nv, F]
            Yv: np.ndarray = None, # float32 [Nv]
            nfolds: int = 8,
            rounds: int = 10_000,
            max_depth: int = 7,
            callbacks: list = None,
            *,
            lr: float = 0.07,
            L2: float = 100_000.0,
            nfeatsets: int = 32,     # e.g., 16*2
            qgrad_bits: int = 12,
            seed: int = 42,
            era_ids: np.ndarray | None = None  # NEW (sorted, contiguous eras)
            ):
        """
        If `era_ids` is provided (sorted, contiguous eras), use DES:
        - H0 -> per-era parents via h0_des()
        - H  -> per-era left stats via h_des()
        - cut -> directional-only DES via cut_des()
        Otherwise, fall back to the original non-DES route.
        """
        assert X.dtype == np.int8 and y.dtype == np.float32
        device = torch.device(self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu")
        callbacks = [] if callbacks is None else callbacks

        # ---------- meta ----------
        self.nfeatsets = int(nfeatsets)
        self.nfolds    = int(nfolds)
        self.max_depth = int(max_depth)
        D   = self.max_depth
        Dm  = max(D - 1, 0)
        nodes  = (1 << D) - 1
        lanes  = 32
        leaf_dtype = (torch.uint8 if D <= 8 else torch.uint16)

        # ---------- 0) eras -> era_bounds (contiguous offsets) ----------
        N, F = X.shape
        self.train_N = int(N)
        if era_ids is None:
            era_bounds = torch.tensor([0, N], device=device, dtype=torch.int64)
            self.era_bounds_ = era_bounds
            use_des = False
        else:
            # accept numpy/torch; assume already sorted by era and contiguous
            e_np = era_ids if isinstance(era_ids, np.ndarray) else np.asarray(era_ids)
            assert e_np.shape[0] == N, "era_ids must be length N"
            # boundaries where era changes
            change = np.flatnonzero(e_np[1:] != e_np[:-1]) + 1
            bounds_np = np.concatenate(([0], change, [N])).astype(np.int64)
            era_bounds = torch.from_numpy(bounds_np).to(device=device)
            self.era_bounds_ = era_bounds
            use_des = True

        # ---------- 1) encode_cuts(X) ----------
        X_t = torch.from_numpy(X).to(device=device, dtype=torch.int8)
        XB  = self.encode_cuts(X_t).contiguous()                    # [4F, M] uint32
        bF, M = XB.shape
        Np = 32 * M                                                 # padded length for XS/XB only
        del X_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Q30 labels (length **N**, not Np) and preds buffer
        yq30  = (y * (1 << 30)).astype(np.int64)
        Y_i32 = torch.from_numpy(yq30[:N].astype(np.int32)).to(device=device)
        P     = torch.zeros(N, dtype=torch.int32, device=device)

        # stash for callbacks / parity dumps
        self.Y_i32 = Y_i32
        self.P_    = P
        self.dY    = Y_i32  # legacy
        self.dP    = P

        # leaf buffers (length **N**)
        L_old = torch.zeros((nfolds, Dm, N), dtype=leaf_dtype, device=device)
        L_new = torch.zeros_like(L_old)

        # ---------- 2) schedules ----------
        rng = np.random.RandomState(seed)

        # Fsch: [rounds, 32*nfeatsets] uint16 indices into XB rows (0..bF-1)
        if bF >= (1 << 16):
            raise ValueError(f"encode_cuts produced {bF} bitplanes (4*F). "
                            f"Schedule dtype is uint16; reduce F or extend to uint32.")
        Fsch_cpu = torch.from_numpy(  # keep a CPU copy in case you run CPU path
            rng.randint(0, bF, size=(rounds, lanes * nfeatsets), dtype=np.uint16)
        ).contiguous()
        self.Fsch = Fsch_cpu.to(device=device, dtype=torch.uint16).contiguous()  # uint16 on device

        # FST: [rounds, nfeatsets, D] uint8; depth-wise shuffled tiling of [0..nfolds-1]
        base = torch.arange(nfolds, dtype=torch.uint8, device=device)
        rep  = (nfeatsets + nfolds - 1) // nfolds
        row  = base.repeat(rep)[:nfeatsets]                                     # [nfeatsets]
        FST  = torch.empty((rounds, nfeatsets, D), dtype=torch.uint8, device=device)
        for s in range(rounds):
            for d in range(D):
                perm = torch.randperm(nfeatsets, device=device)
                FST[s, :, d] = row[perm]
        FST = FST.contiguous()

        # ---------- 3) outputs ----------
        V = torch.zeros((rounds, nfolds, 2 * nodes), dtype=torch.int32,  device=device)
        I = torch.zeros((rounds, nfolds,     nodes), dtype=torch.uint16, device=device)  # <- uint16

        # ---------- 4) optional validation ----------
        use_val = (Xv is not None) and (Yv is not None)
        if use_val:
            assert Xv.dtype == np.int8 and Yv.dtype == np.float32
            Nv = int(Xv.shape[0])
            self.val_N = Nv

            Xv_t = torch.from_numpy(Xv).to(device=device, dtype=torch.int8)
            XBv  = self.encode_cuts(Xv_t).contiguous()                           # [4F, Mv]
            Mv   = XBv.shape[1]                                                  # no need to pad Pv/Yv to 32*Mv
            Pv   = torch.zeros(Nv, dtype=torch.int32, device=device)
            yvq30 = (Yv * (1 << 30)).astype(np.int64)
            Yv_i32 = torch.from_numpy(yvq30[:Nv].astype(np.int32)).to(device=device)

            self.Pv_    = Pv
            self.Yv_i32 = Yv_i32
            self.Yv     = torch.from_numpy(Yv).to(device=device)

            Lv  = torch.zeros((nfolds, Dm, Nv), dtype=leaf_dtype, device=device)
            Lvn = torch.zeros_like(Lv)

            del Xv_t
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            XBv = Pv = Yv_i32 = Lv = Lvn = None

        # ---------- 5) boosting loop ----------
        lr_per_fold = lr / float(nfolds)
        self.tree_set = 0

        for t in range(rounds):
            # (a) sample features -> XS [nfeatsets, 32*M] uint32 (device)
            if XB.is_cuda and torch.cuda.is_available():
                XS = self.et_sample_1b(XB, self.Fsch, t).contiguous()
            else:
                XS = self.et_sample_1b(XB, Fsch_cpu, t).to(device)  # CPU fallback then to device

            # (b) packed leaves + quantized gradients (length **N**)
            LE, G = self.prep_vars(L_old, Y_i32, P)                 # LE: u16/u32/u64 ; G: int16

            # (c) repack by feature schedule (device), output length **N**
            LF = self.repack(FST, LE, t).contiguous()               # [nfeatsets, N]

            # (d) histograms
            if use_des:
                # DES path: per-era H (left) and per-era H0 (parents)
                He  = self.h_des(XS, G, LF, D, era_bounds=self.era_bounds_).contiguous()   # [K1, nE, nodes, 2, 32]
                H0e = self.h0_des(G, LE, D, era_bounds=self.era_bounds_).contiguous()      # [K0, nE, 2**D, 2]
                # trim parents to `nodes` for cut kernel (same as non-DES slice)
                H0e = H0e[:, :, :He.size(2), :].contiguous()                                # [K0, nE, nodes, 2]
                # (e) choose best cuts (DES directional-only)
                self.cut_des(self.Fsch, FST, He, H0e, V, I,
                            tree_set=t, L2=L2, lr=lr_per_fold,
                            qgrad_bits=qgrad_bits, max_depth=D)
                # free DES temps early
                del He, H0e
            else:
                # Non-DES path (original)
                H  = self.H (XS, G, LF, D).contiguous()                                     # [K1, nodes, 2, 32]
                H0 = self.h0(G, LE, D).contiguous()                                         # [K0, 2**D, 2]
                self.cut(self.Fsch, FST, H, H0[:, : H.size(1), :].contiguous(), V, I,
                        tree_set=t, L2=L2, lr=lr_per_fold,
                        qgrad_bits=qgrad_bits, max_depth=D)
                del H, H0

            # (f) advance + predict (length **N**)
            self.advance_and_predict(P, XB, L_old, L_new, V, I, tree_set=t)
            L_old, L_new = L_new, L_old   # ping-pong

            # (g) validation forward (optional, length **Nv**)
            if use_val:
                self.advance_and_predict(Pv, XBv, Lv, Lvn, V, I, tree_set=t)
                Lv, Lvn = Lvn, Lv

            # (h) callbacks
            for cb in callbacks:
                try:
                    cb(self)
                except Exception:
                    pass

            self.tree_set = t + 1

            # free big temporaries ASAP
            del XS, LE, G, LF
            if device.type == "cuda" and (t % 256 == 255):
                torch.cuda.empty_cache()

        # ---------- 6) stash artifacts for inference ----------
        self.FST  = FST                         # uint8, device
        self.V    = V                           # int32, device
        self.I    = I                           # uint16, device
        self.X_packed_ = XB                     # keep packed train X if you want quick train scoring

        return self




    def predict(self, X):
        """
        Predict with the currently trained model.

        X : np.ndarray[int8] or torch.Tensor[int8] of shape [N, F]
            Raw discrete features (0..4 expected per your pipeline).
        Returns:
            np.ndarray[int32] if X is numpy, else torch.Tensor[int32] (length N).
        """
        # --- device & inputs ---
        device = torch.device(self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu")
        if isinstance(X, np.ndarray):
            assert X.dtype == np.int8, "X must be int8"
            X_t = torch.from_numpy(X).to(device=device, dtype=torch.int8)
            return_numpy = True
        else:
            # torch path
            assert torch.is_tensor(X) and X.dtype == torch.int8, "X must be torch.int8"
            X_t = X.to(device=device, dtype=torch.int8, copy=False)
            return_numpy = False

        N = X_t.shape[0]
        # guardrails
        assert hasattr(self, "V") and hasattr(self, "I"), "Model not fitted: missing V/I"
        assert hasattr(self, "tree_set") and self.tree_set > 0, "Model has no trees"

        # --- 1) encode cuts -> packed uint32 words [4F, M] ---
        XB = self.encode_cuts(X_t).contiguous()           # [4F, M] uint32
        M  = XB.shape[1]
        Np = 32 * M                                       # padded length

        # --- 2) buffers ---
        P  = torch.zeros(Np, dtype=torch.int32, device=device)
        D  = int(self.max_depth)
        Dm = max(D - 1, 0)
        leaf_dtype = (torch.uint8 if D <= 8 else torch.int16)
        L  = torch.zeros((self.nfolds, Dm, Np), dtype=leaf_dtype, device=device)
        Ln = torch.zeros_like(L)

        del X_t
        torch.cuda.empty_cache()

        # --- 3) walk trees round-by-round ---
        for k in range(int(self.tree_set)):
            self.advance_and_predict(P, XB, L, Ln, self.V, self.I, tree_set=k)
            L, Ln = Ln, L  # ping-pong

        # --- 4) trim padding & return ---
        out = P[:N]
        if return_numpy:
            return out.detach().cpu().numpy()
        return out


    
    def encode_cuts(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, F], int8 expected
        if X.device.type != 'cpu' and torch.cuda.is_available():
            return kernels.encode_cuts(X.contiguous())

        N, F = X.shape
        M = (N + 31) // 32
        Np = M * 32

        if Np != N:
            pad = torch.zeros((Np - N, F), dtype=torch.int8, device=X.device)
            X = torch.cat([X, pad], dim=0)

        # Pack 32-sample bitplanes per (feature, 4 thresholds)
        X = X.t().contiguous()  # [F, Np]
        thresholds = torch.arange(4, dtype=torch.int8, device=X.device).view(1, 1, 1, 4)  # [1,1,1,4]
        bitplanes = (X.view(F, M, 32, 1) > thresholds).to(torch.uint32)                  # [F, M, 32, 4]

        # weights: 2^0..2^31 as uint32
        weights = (1 << torch.arange(32, dtype=torch.int64, device=X.device)).to(torch.uint32)  # [32]
        weights = weights.view(1, 1, 32, 1)

        words = (bitplanes * weights).sum(dim=2, dtype=torch.int64).to(torch.uint32)    # [F, M, 4]
        XB = words.permute(0, 2, 1).contiguous().reshape(4 * F, M)                       # [4F, M]
        return XB


    def et_sample_1b(self, X : Tensor, Fsch : Tensor, round : int) -> Tensor:
        if X.device.type != 'cpu' and torch.cuda.is_available():
            XS = torch.empty((self.nfeatsets, X.shape[1]*32), dtype=torch.uint32, device=X.device)
            return kernels.et_sample_1b(X.contiguous(), XS.contiguous(), Fsch.contiguous(), round)
        M = X.shape[1]
        nfeatsets = self.nfeatsets
        Fs = Fsch[round].view(nfeatsets, 32).to(dtype=torch.long, device=X.device)
        return X.to(torch.int32)[Fs, :].transpose(1, 2).contiguous().view(nfeatsets, M*32).to(torch.uint32)

    def prep_vars(self, L: torch.Tensor, Y: torch.Tensor, P: torch.Tensor):
        if L.is_cuda and torch.cuda.is_available():
            return kernels.prep_vars(L.contiguous(), Y.contiguous(), P.contiguous())

        K, Dm, N = L.shape
        max_depth = Dm + 1
        LE = torch.zeros((K, N), dtype=torch.int64, device=L.device)

        for d in range(1, max_depth):
            off = (d*(d-1)) // 2
            field = (L[:, d-1].to(torch.int64) & ((1 << d) - 1))
            LE |= (field << off)

        out_dtype = torch.uint64 if Dm>8 else (torch.uint32 if Dm>6 else torch.uint16)
        LE = LE.to(out_dtype)

        g = (Y.to(torch.int32) - P.to(torch.int32)) >> 20
        G = g.clamp_(-32767, 32767).to(torch.int16)

        return LE.contiguous(), G.contiguous()


    def h0(self, G: torch.Tensor, LE: torch.Tensor, max_depth: int) -> torch.Tensor:
        nfolds, N = LE.shape
        nodes = 1 << max_depth
        if (G.is_cuda or LE.is_cuda) and torch.cuda.is_available():
            return kernels.h0_sm_butterfly(G.contiguous(), LE.contiguous(), int(max_depth))

        # --- CPU vectorized fallback ---
        H0 = torch.zeros((nfolds, nodes, 2), dtype=torch.int64, device=LE.device)
        H0_sum = H0[..., 0]  # [nfolds, nodes]
        H0_cnt = H0[..., 1]  # [nfolds, nodes]

        le64 = LE.to(torch.int64, copy=False).contiguous()   # [nfolds, N]
        g64  = G.to(torch.int64,  copy=False).contiguous()   # [N]
        SRC  = g64.unsqueeze(0).expand(nfolds, N).contiguous()
        ONES = torch.ones_like(SRC, dtype=torch.int64)

        for d in range(max_depth):
            if d == 0:
                idx = torch.zeros((nfolds, N), dtype=torch.long, device=LE.device)
            else:
                s = (d * (d - 1)) // 2
                base = (1 << d) - 1
                mask = (1 << d) - 1
                idx = (((le64 >> s) & mask) + base).to(torch.long)
            H0_sum.scatter_add_(1, idx, SRC)
            H0_cnt.scatter_add_(1, idx, ONES)

        return H0

    def repack(self, 
            FST: torch.Tensor,     # [nsets, nfeatsets, max_depth], uint8
            LE:  torch.Tensor,     # [nfolds, N], packed (same dtype used in prep_vars)
            tree_set: int) -> torch.Tensor:
        """
        Returns LF: [nfeatsets, N] with Murky-parity bit packing:
        LF[fs, j] = OR_{d=1..max_depth-1} ( LE[ FST[tree_set, fs, d], j ] & mask(d) )
        where mask(d) = ((1<<d)-1) << ((d*(d-1))//2)
        """
        # --- Fast path: GPU kernel ---
        if FST.is_cuda and LE.is_cuda and torch.cuda.is_available():
            nfeatsets, N = FST.shape[1], LE.shape[1]
            LF = torch.empty((nfeatsets, N), dtype=LE.dtype, device=LE.device)
            # Uses your compiled CUDA kernel; expects contiguous inputs.
            kernels.repack_trees_for_features(
                FST.contiguous(), LE.contiguous(), LF, int(tree_set)
            )
            return LF.contiguous()

        # --- CPU vectorized path ---
        # Shapes
        assert FST.dim() == 3, "FST must be [nsets, nfeatsets, max_depth]"
        assert LE.dim()  == 2, "LE must be [nfolds, N]"
        nsets, nfeatsets, max_depth = FST.shape
        nfolds, N = LE.shape
        device = LE.device
        out_dtype = LE.dtype  # keep parity with packed dtype used upstream

        # Depths d = 1..max_depth-1
        Dm = max_depth - 1
        # Indices: which[fs, d1] in [0..nfolds-1], with d1 = d-1
        which_2d = FST[tree_set, :, 1:].to(torch.long)            # [nfeatsets, Dm]
        idx_flat = which_2d.reshape(-1)                           # [nfeatsets*Dm]
        
        d64 = torch.arange(1, max_depth, device=device, dtype=torch.int64)  # [Dm]
        offsets64 = (d64 * (d64 - 1)) // 2                                  # [Dm]
        # blocks64 = ((1<<d) - 1) << offsets
        blocks64 = ((torch.ones_like(d64) << d64) - 1) << offsets64          # [Dm]
        masks = blocks64.to(dtype=out_dtype)                                 # cast down (wraps as needed)

        
        # Gather LE rows in one shot and reshape -> [nfeatsets, Dm, N]
        LE_sel = LE.index_select(0, idx_flat).view(nfeatsets, Dm, N)


        # Apply masks and reduce across depth.
        # Masks touch disjoint bit ranges, so OR == sum of masked parts.
        LF = (LE_sel & masks.view(1, Dm, 1)).sum(dim=1).to(dtype=out_dtype)  # [nfeatsets, N]
        return LF.contiguous()
    

    def H(self, XS: torch.Tensor, Y: torch.Tensor, LF: torch.Tensor, max_depth: int) -> torch.Tensor:
        """
        XS : [nfeatsets, 32*M] (uint32/int32), packed features
        Y  : [N]               (int32/int64), target/gradient
        LF : [nfeatsets, N]    (uint16/uint32/uint64), path codes (Murky parity)
        max_depth: int (<= 7 for the SMEM variant on GPU)
        Returns:
        H: [nfeatsets, (1<<max_depth)-1, 2, 32] int64
            last dim = lane (0..31); channel 0=sum(y*v), 1=count(v)
        """
        nfeatsets, cols_32M = XS.shape
        N = int(Y.shape[0])
        nodes_tot = (1 << max_depth) - 1

        # --- GPU fast path ---
        if (XS.is_cuda or Y.is_cuda or LF.is_cuda) and torch.cuda.is_available():
            # Expect your compiled extension to expose kernels.h_sm .to(torch.int32)
            return kernels.h_sm(XS.contiguous(), Y.contiguous(), LF.contiguous(), int(max_depth))

        # --- CPU vectorized fallback ---
        dev = XS.device
        TORCH_INT64 = torch.int64

        # Validate shapes
        assert cols_32M % 32 == 0, "XS must have columns divisible by 32"
        M = cols_32M // 32

        # Prepare output
        H_sum = torch.zeros((nfeatsets, nodes_tot, 32), dtype=TORCH_INT64, device=dev)
        H_cnt = torch.zeros_like(H_sum)

        # Casts (no copies if already correct)
        Y64  = Y.to(TORCH_INT64,  copy=False).contiguous()                     # [N]
        LF64 = LF.to(TORCH_INT64, copy=False).contiguous()                     # [F, N]
        XS64 = XS.to(TORCH_INT64, copy=False).contiguous().view(nfeatsets, M, 32)  # [F, M, 32]

        # Build V = bit-unpacked features per lane:
        # V[f, l, s] = ((XS[f, b, l] >> k) & 1) where s = 32*b + k
        # Vectorized unpack: [F,M,32(lane),32(k)]
        bit_ids = torch.arange(32, dtype=TORCH_INT64, device=dev)
        masks   = (1 << bit_ids)                                              # [32]
        V_bits  = ((XS64[..., None] & masks) != 0).to(TORCH_INT64)            # [F, M, 32, 32]
        V       = V_bits.permute(0, 2, 1, 3).reshape(nfeatsets, 32, 32 * M)   # [F, 32, 32*M]
        if V.size(2) > N:                                                     # trim to N
            V = V[:, :, :N].contiguous()                                      # [F, 32, N]

        # Common broadcasts
        Y_b  = Y64.view(1, 1, N)                      # [1,1,N]
        ones = torch.ones((1, 1, N), dtype=TORCH_INT64, device=dev)

        # -------- depth = 0 (node 0) --------
        H_sum[:, 0, :] = (V * Y_b).sum(dim=2)         # [F, 32]
        H_cnt[:, 0, :] = V.sum(dim=2)                 # [F, 32]

        # -------- depth = 1 (nodes 1,2) --------
        tk1 = (LF64 & 1)                               # [F, N]
        m0  = (tk1 == 0).to(TORCH_INT64).unsqueeze(1)  # [F,1,N]
        m1  = 1 - m0                                   # [F,1,N]

        H_sum[:, 1, :] = (V * (Y_b * m0)).sum(dim=2)   # -> node 1
        H_cnt[:, 1, :] = (V * m0).sum(dim=2)
        H_sum[:, 2, :] = (V * (Y_b * m1)).sum(dim=2)   # -> node 2
        H_cnt[:, 2, :] = (V * m1).sum(dim=2)

        # -------- depth = 2 (nodes 3..6) via one-hot --------
        tk2 = ((LF64 >> 1) & 3).to(torch.long)         # [F, N]
        oh2 = Fn.one_hot(tk2, num_classes=4).to(TORCH_INT64)  # [F, N, 4]
        oh2 = oh2.unsqueeze(1)                         # [F,1,N,4]
        V4  = V.unsqueeze(-1)                          # [F,32,N,1]
        sum2 = (V4 * (Y_b.unsqueeze(-1) * oh2)).sum(dim=2)   # [F,32,4]
        cnt2 = (V4 * oh2).sum(dim=2)                   # [F,32,4]
        H_sum[:, 3:7, :] = sum2.permute(0, 2, 1).contiguous()  # [F,4,32]
        H_cnt[:, 3:7, :] = cnt2.permute(0, 2, 1).contiguous()

        # -------- depths >= 3 (nodes 7..(2^D-2)) via scatter_add --------
        if max_depth > 3:
            F32 = nfeatsets * 32
            V_flat  = V.view(F32, N)                               # [F*32, N]
            Y_flat  = Y64.view(1, N).expand(F32, N)               # [F*32, N]

            for d in range(3, max_depth):
                s     = (d * (d - 1)) // 2
                base  = (1 << d) - 1
                maskd = (1 << d) - 1

                idx_fn = (((LF64 >> s) & maskd) + base).to(torch.long).contiguous()  # [F,N]
                idx_b  = idx_fn.repeat_interleave(32, dim=0)                          # [F*32, N]

                out_sum = torch.zeros((F32, nodes_tot), dtype=TORCH_INT64, device=dev)
                out_cnt = torch.zeros_like(out_sum)

                out_sum.scatter_add_(1, idx_b, (V_flat * Y_flat))
                out_cnt.scatter_add_(1, idx_b, V_flat)

                # reshape back to [F, 32, nodes] -> [F, nodes, 32]
                H_sum[:, :, :] += out_sum.view(nfeatsets, 32, nodes_tot).permute(0, 2, 1)
                H_cnt[:, :, :] += out_cnt.view(nfeatsets, 32, nodes_tot).permute(0, 2, 1)

        # stack channels: [F, nodes, 2, 32]
        H = torch.stack([H_sum, H_cnt], dim=2).contiguous()
        return H

    def cut(self,
            F:  torch.Tensor,   # [rounds, 32*K1]        uint16 (preferred) or int16 storage
            FST:torch.Tensor,   # [rounds, K1, D]        uint8
            H:  torch.Tensor,   # [K1, nodes, 2, 32]     int64
            H0: torch.Tensor,   # [K0, nodes, 2]         int64
            V:  torch.Tensor,   # [rounds, K0, 2*nodes]  int32 (OUT)
            I:  torch.Tensor,   # [rounds, K0, nodes]    uint16 (OUT; legacy int16 also accepted)
            tree_set: int,
            L2: float,
            lr: float,
            qgrad_bits: int,
            max_depth: int):
        use_cuda = (F.is_cuda or FST.is_cuda or H.is_cuda or H0.is_cuda or V.is_cuda or I.is_cuda)
        if use_cuda and torch.cuda.is_available():
            from packboost.cuda import kernels
            kernels.cut_cuda(
                F.contiguous(), FST.contiguous(),
                H.contiguous(), H0.contiguous(),
                V, I,
                int(tree_set), float(L2), float(lr),
                int(qgrad_bits), int(max_depth),
            )
            return V, I

        # -------- CPU reference --------
        device = H.device
        K1, nodes = int(H.shape[0]), int(H.shape[1])
        lanes = 32
        K0 = int(H0.shape[0])
        D  = int(FST.shape[2])
        assert H.shape[2] == 2 and H.shape[3] == 32
        assert nodes == (1 << max_depth) - 1
        assert I.dtype in (torch.uint16, torch.int16), "I must be uint16 or int16"

        # depth(leaf) = floor(log2(leaf+1))
        leaves = torch.arange(nodes, device=device, dtype=torch.float32) + 1.0
        depth  = torch.floor(torch.log2(leaves)).to(torch.long)  # [nodes]

        # L2 scaling & qscale (match CUDA)
        L2_eff = torch.tensor(L2, dtype=torch.float32, device=device) * torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=device), 5.0 - depth.float()
        )  # [nodes]
        shift  = 31 - int(qgrad_bits)  # matches CUDA kernel
        qscale = torch.tensor(lr, dtype=torch.float32, device=device) * float(1 << shift) * torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=device), -(float(max_depth) - depth.float())
        )  # [nodes]

        # fold routing per (k, leaf)
        FST_ts = FST[tree_set]  # [K1, D]
        tf_all = FST_ts.gather(1, depth.unsqueeze(0).expand(K1, nodes)).to(torch.long)  # [K1, nodes]

        # Stats
        H_sum = H[:, :, 0, :].to(torch.float32)  # [K1, nodes, 32]
        H_cnt = H[:, :, 1, :].to(torch.float32)  # [K1, nodes, 32]
        H0_sum = H0[:, :, 0].to(torch.float32)   # [K0, nodes]
        H0_cnt = H0[:, :, 1].to(torch.float32)   # [K0, nodes]

        # Fold totals per (k, leaf)
        leaf_idx = torch.arange(nodes, device=device, dtype=torch.long).unsqueeze(0).expand(K1, nodes)
        G01 = H0_sum[tf_all, leaf_idx]  # [K1, nodes]
        N01 = H0_cnt[tf_all, leaf_idx]  # [K1, nodes]

        # Broadcasts
        L2e = L2_eff.view(1, nodes, 1)
        G0  = H_sum                                  # [K1, n, 32]
        N0  = H_cnt
        G1  = G01.unsqueeze(-1) - G0                 # [K1, n, 32]
        N1  = N01.unsqueeze(-1) - N0

        V0f = G0 / (N0 + L2e)                        # [K1, n, 32]
        V1f = G1 / (N1 + L2e)                        # [K1, n, 32]
        S   = (G0 * V0f) + (G1 * V1f)                # [K1, n, 32]

        # Bit-cast float32 -> int32 for tie-consistent compares (CPU: round-trip via NumPy)
        S_bits = torch.from_numpy(S.contiguous().cpu().numpy().view(np.int32)).to(device=device)  # [K1,n,32] int32

        # k-index for lexicographic tie (prefer earlier k)
        k_idx = torch.arange(K1, device=device, dtype=torch.int64).view(K1, 1, 1).expand(K1, nodes, lanes)
        lane_ids = torch.arange(lanes, device=device, dtype=torch.int32).view(1, lanes)
        min_i64 = torch.iinfo(torch.int64).min
        nodes_ar = torch.arange(nodes, device=device, dtype=torch.long)

        # Output views for this tree_set
        V_ts = V[tree_set]  # [K0, 2*nodes]
        I_ts = I[tree_set]  # [K0, nodes]

        # Robust view of F as unsigned for indexing (supports uint16 or int16 storage)
        F_row_u = (F[tree_set].to(torch.int64) & 0xFFFF)  # [32*K1] int64 holding u16 values

        # Iterate over folds (K0 usually small)
        for f in range(K0):
            # ks that route to this fold at each leaf
            Mf = (tf_all == f)  # [K1, n]
            if not Mf.any().item():
                continue
            Mf3 = Mf.unsqueeze(-1).expand(K1, nodes, lanes)  # [K1, n, 32]

            # Lexicographic key:
            #   high 32 bits = S_bits (score compare)
            #   low  32 bits = (0xFFFFFFFF - k)  -> earlier k wins ties
            key_hi = S_bits.to(torch.int64)
            key_lo = (0xFFFFFFFF - k_idx)  # fits in int64
            key    = (key_hi << 32) | key_lo
            key    = torch.where(Mf3, key, torch.full_like(key, min_i64))

            # Per-lane max over k (earliest k on ties)
            key_lane_max, argk_lane = key.max(dim=0)                  # [n, 32], [n, 32]
            Sbits_lane = (key_lane_max >> 32).to(torch.int32)         # [n, 32]

            # Lane tie-break: highest lane among equals (matches CUDA ballot winner)
            Smax_per_leaf, _ = Sbits_lane.max(dim=1, keepdim=True)    # [n, 1]
            winners = (Sbits_lane == Smax_per_leaf)                   # [n, 32]
            chosen_lane = (winners.to(torch.int32) * lane_ids).amax(dim=1)  # [n]

            # Chosen k for each leaf
            k_star = argk_lane[nodes_ar, chosen_lane]                 # [n]

            # Gather stats for (k*, lane*)
            G0_sel  = H_sum[k_star, nodes_ar, chosen_lane]            # [n]
            N0_sel  = H_cnt[k_star, nodes_ar, chosen_lane]            # [n]
            G01_sel = H0_sum[f, nodes_ar]                             # [n]
            N01_sel = H0_cnt[f, nodes_ar]                             # [n]

            L2_sel = L2_eff                                          # [n]
            qs_sel = qscale                                           # [n]

            V0_sel = G0_sel / (N0_sel + L2_sel)
            V1_sel = (G01_sel - G0_sel) / ((N01_sel - N0_sel) + L2_sel)

            # Quantize to int32 (CUDA uses trunc toward zero)
            V_ts[f, 2*nodes_ar    ] = torch.trunc(qs_sel * V0_sel).to(torch.int32)
            V_ts[f, 2*nodes_ar + 1] = torch.trunc(qs_sel * V1_sel).to(torch.int32)

            # Feature indices: I = F[tree_set, 32*k* + lane*] (store as uint16)
            feat_pos = (k_star.to(torch.int64) * lanes + chosen_lane.to(torch.int64))  # [n]
            # new — gather as int64, mask to u16 range, cast once, then row-copy
            vals_u16 = (F_row_u.index_select(0, feat_pos) & 0xFFFF).to(torch.uint16)  # [nodes]
            I_ts[f].copy_(vals_u16)  # write the entire row; avoids index_put on UInt16

        return V, I


    def advance_and_predict(self,
                            P: torch.Tensor,     # [N], int32 (in/out)
                            X: torch.Tensor,     # [R, M], uint32/int32
                            L_old: torch.Tensor, # [K0, Dm, N], uint8/uint16
                            L_new: torch.Tensor, # [K0, Dm, N], same dtype
                            V: torch.Tensor,     # [rounds, K0, 2*nodes], int32
                            I: torch.Tensor,     # [rounds, K0, nodes], uint16 (or legacy int16)
                            tree_set: int):

        use_cuda = any(t.is_cuda for t in (P, X, L_old, L_new, V, I))
        K0, Dm, N = L_old.shape
        nodes = I.shape[2]
        depths = min(tree_set + 1, Dm + 1)

        if use_cuda and torch.cuda.is_available():
            from packboost.cuda import kernels
            kernels.advance_and_predict(
                P.contiguous(), X.contiguous(),
                L_old.contiguous(), L_new.contiguous(),
                V.contiguous(), I.contiguous(), int(tree_set)
            )
            return P, L_new

        # -------- CPU vectorized path --------
        assert P.dtype == torch.int32 and X.dtype in (torch.int32, torch.uint32)
        assert V.dtype == torch.int32
        assert I.dtype in (torch.uint16, torch.int16), "I must be uint16 or int16"

        device = P.device

        # Use int64 proxy for bit ops on X
        X_ix = X.to(torch.int64) if (X.device.type == "cpu" and X.dtype == torch.uint32) else X.to(torch.int64, copy=False)

        # >>> NEW: make an int64 view of I row(s) to allow gather on CPU, keep unsigned semantics
        # One-time cast (copy) is fine for CPU test path.
        I_ix = I.to(torch.int64) & 0xFFFF  # shape [rounds, K0, nodes], values treated as unsigned 16-bit

        k = torch.arange(N, device=device, dtype=torch.long)
        word_idx = (k >> 5)
        bit_off  = (k & 31).to(torch.int64)

        for depth in range(depths):
            for f in range(K0):
                if depth == 0:
                    leaf_prev = torch.zeros(N, dtype=torch.int64, device=device)
                else:
                    leaf_prev = L_old[f, depth - 1].to(torch.int64)

                lo = leaf_prev + ((1 << depth) - 1)

                # >>> CHANGED: gather from I_ix (int64), not I (uint16), to avoid unsupported gather
                li = I_ix[tree_set, f].gather(0, lo.to(torch.long))  # int64

                words = X_ix[li, word_idx]                      # int64
                x     = ((words >> bit_off) & 1).to(torch.int64)

                leaf_new = (leaf_prev << 1) | x
                if depth < Dm:
                    if L_new.dtype == torch.uint8:
                        L_new[f, depth] = leaf_new.to(torch.uint8)
                    elif L_new.dtype == torch.uint16:
                        L_new[f, depth] = leaf_new.to(torch.uint16)
                    else:
                        raise AssertionError("L_new must be uint8 or uint16")

                add_idx = (2 * lo + 1 + x).to(torch.long)
                P.add_(V[tree_set, f].gather(0, add_idx))

        return P, L_new
    
        # -------- DES: H0 per-era (parents) --------
    def h0_des(self,
               G:  torch.Tensor,   # [N] int16
               LE: torch.Tensor,   # [nfolds, N] (u16/u32/u64)
               max_depth: int,
               era_bounds: torch.Tensor | None = None  # [nE+1] int32/int64 (device-matched)
               ) -> torch.Tensor:
        """
        Returns H0e: [nfolds, nE, 2**D, 2] int64
        (sum,count) per fold, per era, per parent node.
        """
        nfolds, N = int(LE.size(0)), int(LE.size(1))
        if era_bounds is None:
            era_bounds = torch.tensor([0, N], device=LE.device, dtype=torch.int64)
        else:
            era_bounds = era_bounds.to(device=LE.device, dtype=torch.int64).contiguous()
        nE = int(era_bounds.numel() - 1)

        # ---- GPU path ----
        use_cuda = (G.is_cuda or LE.is_cuda or era_bounds.is_cuda) and torch.cuda.is_available()
        if use_cuda:
            return kernels.h0_des(
                G.contiguous(), LE.contiguous(), era_bounds.contiguous(), int(max_depth)
            ).contiguous()

        # ---- CPU fallback ----
        # Support nE==1 cheaply by reusing existing CPU h0() and adding era dim.
        if nE == 1:
            H0 = self.h0(G, LE, max_depth)  # [nfolds, 2**D, 2]
            return H0.unsqueeze(1).contiguous()  # [nfolds, 1, 2**D, 2]
        raise NotImplementedError("CPU multi-era H0 (DES) not implemented; pass era_bounds with nE==1 or use CUDA.")


    # -------- DES: H per-era (left child stats per candidate) --------
    def h_des(self,
              XS: torch.Tensor,   # [K1(=nfeatsets), 32*M] uint32/int32
              Y:  torch.Tensor,   # [N] int16
              LF: torch.Tensor,   # [K1, N] (u16/u32/u64)
              max_depth: int,
              era_bounds: torch.Tensor | None = None  # [nE+1] (contiguous eras)
              ) -> torch.Tensor:
        """
        Returns He: [K1, nE, (2**D)-1, 2, 32] int64
        channel 0=sum(y*v), 1=count(v); last dim is lane (0..31).
        """
        K1, cols_32M = int(XS.size(0)), int(XS.size(1))
        N = int(Y.size(0))
        if era_bounds is None:
            era_bounds = torch.tensor([0, N], device=XS.device, dtype=torch.int64)
        else:
            era_bounds = era_bounds.to(device=XS.device, dtype=torch.int64).contiguous()

        # ---- GPU path ----
        use_cuda = (XS.is_cuda or Y.is_cuda or LF.is_cuda or era_bounds.is_cuda) and torch.cuda.is_available()
        if use_cuda:
            return kernels.h_des(
                XS.contiguous(), Y.contiguous(), LF.contiguous(),
                era_bounds.contiguous(), int(max_depth)
            ).contiguous()

        # ---- CPU fallback ----
        # Single-era: reuse existing CPU H() and add era dim.
        if int(era_bounds.numel()) == 2:
            H_all = self.H(XS, Y, LF, max_depth)  # [K1, nodes, 2, 32]
            return H_all.unsqueeze(1).contiguous()  # [K1, 1, nodes, 2, 32]
        raise NotImplementedError("CPU multi-era H (DES) not implemented; pass single-era bounds or use CUDA.")


    # -------- DES: directional cut (tie-break by classic score), per spec --------
    def cut_des(self,
                F:  torch.Tensor,   # [rounds, 32*K1] uint16 (feature ids per lane)
                FST:torch.Tensor,   # [rounds, K1, D] uint8 (fold id per candidate/depth)
                H:  torch.Tensor,   # [K1, nE, nodes, 2, 32] int64  (left stats per era)
                H0: torch.Tensor,   # [K0, nE, nodes, 2]     int64  (parent stats per era)
                V:  torch.Tensor,   # [rounds, K0, 2*nodes]  int32  (OUT)
                I:  torch.Tensor,   # [rounds, K0, nodes]    uint16 (OUT)
                *,
                tree_set: int,
                L2: float,
                lr: float,
                qgrad_bits: int,
                max_depth: int):
        """
        Directional-only DES:
          score = |mean_e sign(v_R - v_L)| over eras with both children non-empty.
        Tie-break = classic (S0 + S1) as int-bits.
        """
        use_cuda = any(t.is_cuda for t in (F, FST, H, H0, V, I)) and torch.cuda.is_available()
        if use_cuda:
            kernels.cut_cuda_des(
                F.contiguous(), FST.contiguous(),
                H.contiguous(), H0.contiguous(),
                V, I,
                int(tree_set), float(L2), float(lr),
                int(qgrad_bits), int(max_depth)
            )
            return V, I

        # CPU fallback not provided (DES selection is CUDA-optimized).
        raise NotImplementedError("CPU cut_des not implemented. Use CUDA path.")
