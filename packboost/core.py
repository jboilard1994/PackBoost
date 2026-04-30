import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor
import torch.nn.functional as Fn
import os

print('Installing kernels...')
from packboost.cuda import kernels
print('kernels successfully Installed!')
from packboost.callback import EarlyStoppingCallback


class PackBoost(BaseEstimator, RegressorMixin):
    def __init__(self, device='cuda',comment=""):
        self.device = device
        self.nfeatsets = 32
        self.stop_training = False
        self.feature_name = None
        self.comment = comment

    @classmethod
    def from_params(cls, V, I, device='cuda'):
        """
        Create a PackBoost instance from pre-trained parameters.
        
        Parameters
        ----------
        V : torch.Tensor or np.ndarray
            The tree node values with shape (rounds, nfolds, 2*nodes)
        I : torch.Tensor or np.ndarray
            The tree node split indices with shape (rounds, nfolds, nodes)
        device : str, optional
            The device to use ('cuda' or 'cpu'), default is 'cuda'
        
        Returns
        -------
        PackBoost
            A PackBoost instance initialized with the given parameters
        """
        instance = cls(device=device)
        
        # Convert to torch tensors if needed
        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)
        if isinstance(I, np.ndarray):
            I = torch.from_numpy(I)
        
        # Move to specified device
        device_obj = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        instance.V = V.to(device=device_obj, dtype=torch.int32)
        instance.I = I.to(device=device_obj, dtype=torch.uint16)
        
        # Infer metadata from parameter shapes
        if V.ndim == 3 and I.ndim == 3:
            rounds, nfolds, double_nodes = V.shape
            _, _, nodes = I.shape
            instance.nfolds = nfolds
            instance.tree_set = rounds
            # Infer max_depth from nodes count: nodes = 2^D - 1
            instance.max_depth = int(np.log2(nodes + 1))
        
        return instance

    def fit(self,
            X: np.ndarray, y: np.ndarray,
            Xv: np.ndarray = None, Yv: np.ndarray = None,
            nfolds: int = 8,
            rounds: int = 10_000,
            max_depth: int = 7,
            callbacks: list = None,
            feature_name: list = None,
            *,
            lr: float = 0.07,
            L2: float = 100_000.0,
            min_child_weight: float = 20.0,
            min_split_gain: float = 0.0,
            nfeatsets: int = 32,
            qgrad_bits: int = 12,
            seed: int = 42,
            encode_cut_device: str = "cuda",
            era_ids: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,
            max_delta_step: float = 0.0,
            profile_every: int = 0):
        """
        Fit the PackBoost model.

        Parameters
        ----------
        X : np.ndarray, shape (N, F), dtype int8
            Quantized feature matrix.
        y : np.ndarray, shape (N,), dtype float32
            Target values.
        Xv : np.ndarray, optional
            Validation feature matrix (same dtype/shape convention as X).
        Yv : np.ndarray, optional
            Validation targets.
        nfolds : int
            Number of ensemble folds (parallel trees).
        rounds : int
            Maximum number of boosting rounds.
        max_depth : int
            Maximum tree depth (supports up to 8 with the SMEM kernel).
        callbacks : list, optional
            List of callback objects called after each round.
        feature_name : list, optional
            Names for each feature column, used for reporting.
        lr : float
            Learning rate (shrinkage applied to each tree's leaf values).
        L2 : float
            L2 regularization on leaf values (in the same units as
            ``sample_weight``; scale accordingly if weights are not mean~1).
        min_child_weight : float
            Minimum sum of sample weights required in a leaf to allow a split.
            Interpreted in the same units as ``sample_weight``.
        min_split_gain : float
            Minimum gain required to accept a split.
        nfeatsets : int
            Number of random feature subsets sampled per round.
        qgrad_bits : int
            Bit-width used to quantize gradients to int16 before histogram
            accumulation.
        seed : int
            Random seed for feature-set sampling.
        encode_cut_device : str
            Device used for cut-encoding pre-processing (``"cuda"`` or
            ``"cpu"``).
        era_ids : np.ndarray, shape (N,), optional
            Era/group IDs for era-aware gradient normalization.
        sample_weight : np.ndarray, shape (N,), optional
            Per-sample weights (float, must be >= 0). Fractional values are
            supported: internally weights are scaled by 1024 before rounding
            to int16, giving a resolution of 1/1024 ≈ 0.001. The maximum
            representable weight is 32767/1024 ≈ 31.999; a ``ValueError`` is
            raised if any weight exceeds this. Values below ~0.00049 round to
            0 and are effectively excluded. L2 and min_child_weight are scaled
            by the same factor internally so their user-facing units remain
            unchanged (mean weight ~1). Passing ``np.ones(N)`` is equivalent
            to not passing sample_weight.
        max_delta_step : float, optional
            If > 0, clips each leaf value to ``[-max_delta_step, +max_delta_step]``
            before quantization. Useful when using sample weights to prevent
            high-weight leaves from producing extreme gradient steps. Default 0
            (disabled). Equivalent to XGBoost's ``max_delta_step``.
        profile_every : int, optional
            If > 0, collect internal diagnostics every N rounds and store
            them in ``self.profile_`` as a ``pandas.DataFrame``. Columns:
            ``round, G_min, G_max, G_mean_abs, V_min, V_max, V_mean_abs,
            V_nonzero, P_min, P_max, P_mean, Pv_min, Pv_max, Pv_mean``.
            Default 0 (disabled). Overhead is negligible (a few GPU
            reductions on already-resident tensors).
        """
        assert X.dtype == np.int8 and y.dtype == np.float32
        device = torch.device(self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu")
        encode_device = torch.device(
            encode_cut_device if (encode_cut_device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        callbacks = [] if callbacks is None else callbacks

        # ---------- meta ----------
        self.feature_name = feature_name
        self.nfeatsets = int(nfeatsets)
        self.nfolds    = int(nfolds)
        self.max_depth = int(max_depth)
        self.min_child_weight = float(min_child_weight)
        self.min_split_gain   = float(min_split_gain)

        D   = self.max_depth
        Dm  = max(D - 1, 0)
        nodes  = (1 << D) - 1
        lanes  = 32
        leaf_dtype = (torch.uint8 if D <= 8 else torch.uint16)

        # ---------- eras -> era_ends ----------
        N, F = X.shape
        self.train_N = int(N)
        if era_ids is None:
            ends_np = np.array([N], dtype=np.int32)   # single era
            use_des = False
        else:
            e_np = era_ids if isinstance(era_ids, np.ndarray) else np.asarray(era_ids)
            assert e_np.shape[0] == N
            change = np.flatnonzero(e_np[1:] != e_np[:-1]) + 1
            ends_np = np.concatenate((change, [N])).astype(np.int32)
            use_des = True
        era_ends = torch.from_numpy(ends_np).to(device=device, dtype=torch.int32).contiguous()

        # ---------- encode_cuts(X) ----------
        X_t = torch.from_numpy(X).to(device=encode_device, dtype=torch.int8)
        XB  = self.encode_cuts(X_t).contiguous().to(device=device)              # [4F, M] uint32
        bF, M = XB.shape
        Np = 32 * M
        del X_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # labels/preds (length N, unpadded)
        yq30  = (y * (1 << 30)).astype(np.int64)
        Y_i32 = torch.from_numpy(yq30[:N].astype(np.int32)).to(device=device)
        P     = torch.zeros(N, dtype=torch.int32, device=device)

        # ---------- sample weights (int16, mean ~ 1) ----------
        # Fractional weights are supported via fixed-point scaling: w=1.0 maps to
        # W_i16=1024, giving 1/1024 resolution. L2 and min_child_weight are scaled
        # by the same factor so the effective leaf value formula stays:
        #   V = sum(G*w) / (sum(w) + L2).
        # Default (no weights): W_i16=1, w_scale=1 (unchanged behavior).
        _W_FRAC_SCALE = 1024
        _W_MAX = 32767.0 / _W_FRAC_SCALE   # ~31.999
        if sample_weight is None:
            W_i16 = torch.ones(N, dtype=torch.int16, device=device)
            _w_scale = 1
        else:
            assert sample_weight.shape[0] == N, "sample_weight length must equal N"
            w_np = np.asarray(sample_weight, dtype=np.float32)
            if np.any(w_np < 0.0):
                raise ValueError(
                    f"sample_weight contains negative values (min={w_np.min():.6g}). "
                    "Negative weights are not supported."
                )
            w_max = float(w_np.max())
            if w_max > _W_MAX:
                raise ValueError(
                    f"sample_weight contains values exceeding the maximum representable weight "
                    f"of {_W_MAX:.4f} (got {w_max:.6g}). "
                    f"Normalize weights so that max(sample_weight) <= {_W_MAX:.4f}, "
                    f"e.g. sample_weight = sample_weight / sample_weight.max() * {_W_MAX:.4f}."
                )
            w_q  = np.rint(w_np * _W_FRAC_SCALE).astype(np.int64)
            w_q  = np.clip(w_q, 0, 32767).astype(np.int16)
            W_i16 = torch.from_numpy(w_q).to(device=device)
            _w_scale = _W_FRAC_SCALE
        self.W_i16 = W_i16

        self.Y_i32 = Y_i32
        self.P_    = P
        self.dY    = Y_i32
        self.dP    = P

        # leaves (length N)
        L_old = torch.zeros((nfolds, Dm, N), dtype=leaf_dtype, device=device)
        L_new = torch.zeros_like(L_old)

        # ---------- schedules ----------
        rng = np.random.RandomState(seed)
        if bF >= (1 << 16):
            raise ValueError(
                f"encode_cuts produced {bF} bitplanes (4*F). "
                f"Schedule dtype is uint16; reduce F or extend to uint32."
            )
        
        '''
        Fsch_cpu = torch.from_numpy(
            rng.randint(0, bF, size=(rounds, lanes * nfeatsets), dtype=np.uint16)
        ).contiguous()
        self.Fsch = Fsch_cpu.to(device=device, dtype=torch.uint16).contiguous()
        '''
        K1 = lanes * nfeatsets

        if bF < K1:
            raise ValueError(f"Need bF >= 32*nfeatsets for sampling without replacement; got bF={bF}, K1={K1}")

        Fsch_np = np.empty((rounds, K1), dtype=np.uint16)
        for t in range(rounds):
            # distinct within round t
            Fsch_np[t, :] = rng.choice(bF, size=K1, replace=False).astype(np.uint16, copy=False)

        Fsch_cpu = torch.from_numpy(Fsch_np).contiguous()
        self.Fsch = Fsch_cpu.to(device=device, dtype=torch.uint16).contiguous()


        base = torch.arange(nfolds, dtype=torch.uint8, device=device)
        rep  = (nfeatsets + nfolds - 1) // nfolds
        row  = base.repeat(rep)[:nfeatsets]
        FST  = torch.empty((rounds, nfeatsets, D), dtype=torch.uint8, device=device)
        for s in range(rounds):
            for d in range(D):
                perm = torch.randperm(nfeatsets, device=device)
                FST[s, :, d] = row[perm]
        FST = FST.contiguous()

        # ---------- outputs ----------
        self.V = torch.zeros((rounds, nfolds, 2 * nodes), dtype=torch.int32,  device=device)
        self.I = torch.zeros((rounds, nfolds,     nodes), dtype=torch.uint16, device=device)

        # ---------- optional validation ----------
        use_val = (Xv is not None) and (Yv is not None)
        if use_val:
            assert Xv.dtype == np.int8 and Yv.dtype == np.float32
            Nv = int(Xv.shape[0]); self.val_N = Nv
            Xv_t = torch.from_numpy(Xv).to(device=encode_device, dtype=torch.int8)
            XBv  = self.encode_cuts(Xv_t).contiguous().to(device=device)
            Pv   = torch.zeros(Nv, dtype=torch.int32, device=device)
            yvq30  = (Yv * (1 << 30)).astype(np.int64)
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

        # ---------- boosting loop ----------
        lr_per_fold = lr / float(nfolds)
        self.tree_set = 0

        # ---------- profiling setup ----------
        do_profile = (profile_every > 0)
        _prof_rows = [] if do_profile else None

        for t in range(rounds):
            
            # early stopping check
            if self.stop_training:
                self.stop_training = False
                break

            # (a) feature sampling -> XS [nfeatsets, 32*M] uint32
            if XB.is_cuda and torch.cuda.is_available():
                XS = self.et_sample_1b(XB, self.Fsch, t).contiguous()
            else:
                XS = self.et_sample_1b(XB, Fsch_cpu, t).to(device)

            # (b) pack leaves & gradients (length N)
            LE, G = self.prep_vars(L_old, Y_i32, P)  # LE: u16/u32/u64 ; G: int16

            # --- profile: gradient stats (before they're consumed) ---
            if do_profile and (t % profile_every == 0):
                _g_min = int(G.min().item())
                _g_max = int(G.max().item())
                _g_mabs = float(G.float().abs().mean().item())

            # (c) repack by feature schedule -> LF [nfeatsets, N]
            LF = self.repack(FST, LE, t).contiguous()

            # (d) histograms & (e) choose cuts
            if use_des:
                # DES path (multi-era aware). Kernel API shapes:
                #   h0_des -> [K0, 2**D, E, 2]
                #   h_des  -> [K1, E, (2**D-1), 2, 32]
                #   cut_des expects H0 in nodes-major: [K0, nodes, E, 2]
                if int(era_ends[-1].item()) != N:
                    raise ValueError(
                        f"era_ends[-1] must equal N ({N}), got {int(era_ends[-1].item())}"
                    )

                H0_all = self.h0_des(G, W_i16, LE, int(max_depth), era_ends)     # [K0, 2**D, E, 2]
                H0e    = H0_all[:, :nodes, :, :].contiguous()                    # [K0, nodes, E, 2]
                He     = self.h_des(XS, G, W_i16, LF, int(max_depth), era_ends)  # [K1, E, nodes, 2, 32]

                self.cut_des(
                    self.Fsch, FST, He, H0e, self.V, self.I,
                    tree_set=t, L2=L2 * _w_scale, lr=lr_per_fold,
                    qgrad_bits=qgrad_bits, max_depth=D,
                    min_child_weight=min_child_weight * _w_scale,
                    min_split_gain=min_split_gain,
                    max_delta_step=max_delta_step,
                )

                del He, H0_all, H0e
            else:
                # Non-DES path (single era)
                H  = self.H (XS, G, W_i16, LF, D).contiguous()                   # [K1, nodes, 2, 32]
                H0 = self.h0(G, W_i16, LE, D).contiguous()                       # [K0, 2**D, 2]
                self.cut(
                    self.Fsch, FST, H, H0[:, : H.size(1), :].contiguous(),       # -> [K0, nodes, 2]
                    self.V, self.I,
                    tree_set=t, L2=L2 * _w_scale, lr=lr_per_fold,
                    qgrad_bits=qgrad_bits, max_depth=D,
                    min_child_weight=min_child_weight * _w_scale,
                    min_split_gain=min_split_gain,
                    max_delta_step=max_delta_step,
                )
                del H, H0

            # (f) advance + predict
            self.advance_and_predict(P, XB, L_old, L_new, self.V, self.I, tree_set=t)
            L_old, L_new = L_new, L_old

            # (g) validation (optional)
            if use_val:
                self.advance_and_predict(Pv, XBv, Lv, Lvn, self.V, self.I, tree_set=t)
                Lv, Lvn = Lvn, Lv

            self.tree_set = t + 1

            # --- profile: leaf value & prediction stats ---
            if do_profile and (t % profile_every == 0):
                Vt = self.V[t].float()
                Vt_nz = Vt[Vt != 0]
                _v_nz = int(Vt_nz.numel())
                if _v_nz > 0:
                    _v_min = int(Vt_nz.min().item())
                    _v_max = int(Vt_nz.max().item())
                    _v_mabs = float(Vt_nz.abs().mean().item())
                else:
                    _v_min = _v_max = 0
                    _v_mabs = 0.0

                _p_min = float(P.min().item())
                _p_max = float(P.max().item())
                _p_mean = float(P.float().mean().item())

                if use_val:
                    _pv_min = float(Pv.min().item())
                    _pv_max = float(Pv.max().item())
                    _pv_mean = float(Pv.float().mean().item())
                else:
                    _pv_min = _pv_max = _pv_mean = float('nan')

                _prof_rows.append({
                    'round': t,
                    'G_min': _g_min, 'G_max': _g_max, 'G_mean_abs': _g_mabs,
                    'V_min': _v_min, 'V_max': _v_max, 'V_mean_abs': _v_mabs,
                    'V_nonzero': _v_nz,
                    'P_min': _p_min, 'P_max': _p_max, 'P_mean': _p_mean,
                    'Pv_min': _pv_min, 'Pv_max': _pv_max, 'Pv_mean': _pv_mean,
                })

            # (h) callbacks
            for cb in callbacks:
                try:
                    cb(self)
                except Exception:
                    pass

            # free big temporaries ASAP
            del XS, LE, G, LF
            if device.type == "cuda" and (t % 256 == 255):
                torch.cuda.empty_cache()

        
        # ---------- restore best model if early stopping was used ----------
        for cb in callbacks:
            if isinstance(cb, EarlyStoppingCallback) and cb.keep_best:
                cb.restore_best(self)
                break

        # ---------- assemble profile DataFrame ----------
        if do_profile and _prof_rows:
            import pandas as _pd
            self.profile_ = _pd.DataFrame(_prof_rows).set_index('round')
        elif do_profile:
            self.profile_ = None

        # ---------- stash for inference ----------
        self.FST  = FST
        self.X_packed_ = XB
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


    def h0(self, G: torch.Tensor, W: torch.Tensor, LE: torch.Tensor, max_depth: int) -> torch.Tensor:
        nfolds, N = LE.shape
        nodes = 1 << max_depth
        if (G.is_cuda or LE.is_cuda) and torch.cuda.is_available():
            return kernels.h0_sm_butterfly(G.contiguous(), W.contiguous(), LE.contiguous(), int(max_depth))

        # --- CPU vectorized fallback ---
        H0 = torch.zeros((nfolds, nodes, 2), dtype=torch.int64, device=LE.device)
        H0_sum = H0[..., 0]  # [nfolds, nodes]
        H0_w   = H0[..., 1]  # [nfolds, nodes]

        le64 = LE.to(torch.int64, copy=False).contiguous()   # [nfolds, N]
        g64  = G.to(torch.int64,  copy=False).contiguous()   # [N]
        w64  = W.to(torch.int64,  copy=False).contiguous()   # [N]
        SRC  = g64.unsqueeze(0).expand(nfolds, N).contiguous()
        WSRC = w64.unsqueeze(0).expand(nfolds, N).contiguous()
        WGSRC = (g64 * w64).unsqueeze(0).expand(nfolds, N).contiguous()  # weighted gradient

        for d in range(max_depth):
            if d == 0:
                idx = torch.zeros((nfolds, N), dtype=torch.long, device=LE.device)
            else:
                s = (d * (d - 1)) // 2
                base = (1 << d) - 1
                mask = (1 << d) - 1
                idx = (((le64 >> s) & mask) + base).to(torch.long)
            H0_sum.scatter_add_(1, idx, WGSRC)
            H0_w.scatter_add_(1, idx, WSRC)

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
    

    def H(self, XS: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, LF: torch.Tensor, max_depth: int) -> torch.Tensor:
        """
        XS : [nfeatsets, 32*M] (uint32/int32), packed features
        Y  : [N]               (int16), gradient
        W  : [N]               (int16), per-sample weights
        LF : [nfeatsets, N]    (uint16/uint32/uint64), path codes (Murky parity)
        max_depth: int (<= 7 for the SMEM variant on GPU)
        Returns:
        H: [nfeatsets, (1<<max_depth)-1, 2, 32] int64
            channel 0 = sum(y*v), 1 = sum(w*v)
        """
        nfeatsets, cols_32M = XS.shape
        N = int(Y.shape[0])
        nodes_tot = (1 << max_depth) - 1

        # --- GPU fast path ---
        if (XS.is_cuda or Y.is_cuda or LF.is_cuda) and torch.cuda.is_available():
            return kernels.h_sm(XS.contiguous(), Y.contiguous(), W.contiguous(), LF.contiguous(), int(max_depth))

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
        W64  = W.to(TORCH_INT64,  copy=False).contiguous()                     # [N]
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
        W_b  = W64.view(1, 1, N)                      # [1,1,N]
        WY_b = W_b * Y_b                              # [1,1,N] weighted gradient

        # -------- depth = 0 (node 0) --------
        H_sum[:, 0, :] = (V * WY_b).sum(dim=2)        # [F, 32]
        H_cnt[:, 0, :] = (V * W_b).sum(dim=2)         # [F, 32]

        # -------- depth = 1 (nodes 1,2) --------
        tk1 = (LF64 & 1)                               # [F, N]
        m0  = (tk1 == 0).to(TORCH_INT64).unsqueeze(1)  # [F,1,N]
        m1  = 1 - m0                                   # [F,1,N]

        H_sum[:, 1, :] = (V * (WY_b * m0)).sum(dim=2)  # -> node 1
        H_cnt[:, 1, :] = (V * (W_b * m0)).sum(dim=2)
        H_sum[:, 2, :] = (V * (WY_b * m1)).sum(dim=2)  # -> node 2
        H_cnt[:, 2, :] = (V * (W_b * m1)).sum(dim=2)

        # -------- depth = 2 (nodes 3..6) via one-hot --------
        tk2 = ((LF64 >> 1) & 3).to(torch.long)         # [F, N]
        oh2 = Fn.one_hot(tk2, num_classes=4).to(TORCH_INT64)  # [F, N, 4]
        oh2 = oh2.unsqueeze(1)                         # [F,1,N,4]
        V4  = V.unsqueeze(-1)                          # [F,32,N,1]
        sum2 = (V4 * (WY_b.unsqueeze(-1) * oh2)).sum(dim=2)  # [F,32,4]
        cnt2 = (V4 * (W_b.unsqueeze(-1) * oh2)).sum(dim=2)   # [F,32,4]
        H_sum[:, 3:7, :] = sum2.permute(0, 2, 1).contiguous()  # [F,4,32]
        H_cnt[:, 3:7, :] = cnt2.permute(0, 2, 1).contiguous()

        # -------- depths >= 3 (nodes 7..(2^D-2)) via scatter_add --------
        if max_depth > 3:
            F32 = nfeatsets * 32
            V_flat  = V.view(F32, N)                               # [F*32, N]
            Y_flat  = Y64.view(1, N).expand(F32, N)               # [F*32, N]
            W_flat  = W64.view(1, N).expand(F32, N)               # [F*32, N]
            WY_flat = (Y64 * W64).view(1, N).expand(F32, N)       # [F*32, N] weighted grad

            for d in range(3, max_depth):
                s     = (d * (d - 1)) // 2
                base  = (1 << d) - 1
                maskd = (1 << d) - 1

                idx_fn = (((LF64 >> s) & maskd) + base).to(torch.long).contiguous()  # [F,N]
                idx_b  = idx_fn.repeat_interleave(32, dim=0)                          # [F*32, N]

                out_sum = torch.zeros((F32, nodes_tot), dtype=TORCH_INT64, device=dev)
                out_cnt = torch.zeros_like(out_sum)

                out_sum.scatter_add_(1, idx_b, (V_flat * WY_flat))
                out_cnt.scatter_add_(1, idx_b, (V_flat * W_flat))

                # reshape back to [F, 32, nodes] -> [F, nodes, 32]
                H_sum[:, :, :] += out_sum.view(nfeatsets, 32, nodes_tot).permute(0, 2, 1)
                H_cnt[:, :, :] += out_cnt.view(nfeatsets, 32, nodes_tot).permute(0, 2, 1)

        # stack channels: [F, nodes, 2, 32]
        H = torch.stack([H_sum, H_cnt], dim=2).contiguous()
        return H

    def cut(
        self,
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
        max_depth: int,
        min_child_weight: float = 20.0,
        min_split_gain: float = 0.0,
        max_delta_step: float = 0.0,
    ):
        use_cuda = (
            F.is_cuda or FST.is_cuda or H.is_cuda or
            H0.is_cuda or V.is_cuda or I.is_cuda
        )
        if use_cuda and torch.cuda.is_available():
            from packboost.cuda import kernels
            kernels.cut_cuda(
                F.contiguous(), FST.contiguous(),
                H.contiguous(), H0.contiguous(),
                V, I,
                int(tree_set), float(L2), float(lr),
                int(qgrad_bits), int(max_depth),
                float(min_child_weight), float(min_split_gain),
                float(max_delta_step),
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

        use_min_child = (min_child_weight > 0.0)
        use_min_gain  = (min_split_gain  > 0.0)

        # depth(leaf) = floor(log2(leaf+1))
        leaves = torch.arange(nodes, device=device, dtype=torch.float32) + 1.0
        depth  = torch.floor(torch.log2(leaves)).to(torch.long)  # [nodes]

        # L2 scaling & qscale (match CUDA)
        L2_eff = torch.tensor(L2, dtype=torch.float32, device=device) * torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=device),
            5.0 - depth.float()
        )  # [nodes]
        shift  = 31 - int(qgrad_bits)  # matches CUDA kernel
        qscale = torch.tensor(lr, dtype=torch.float32, device=device) * float(1 << shift) * torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=device),
            -(float(max_depth) - depth.float())
        )  # [nodes]

        # fold routing per (k, leaf)
        FST_ts = FST[tree_set]  # [K1, D]
        tf_all = FST_ts.gather(
            1,
            depth.unsqueeze(0).expand(K1, nodes)
        ).to(torch.long)  # [K1, nodes]

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
        N0  = H_cnt                                  # [K1, n, 32]
        G1  = G01.unsqueeze(-1) - G0                 # [K1, n, 32]
        N1  = N01.unsqueeze(-1) - N0                 # [K1, n, 32]

        V0f = G0 / (N0 + L2e)                        # [K1, n, 32]
        V1f = G1 / (N1 + L2e)                        # [K1, n, 32]
        S   = (G0 * V0f) + (G1 * V1f)                # [K1, n, 32]  gain per lane

        # Validity mask for min_child_weight / min_split_gain
        if use_min_child or use_min_gain:
            valid = torch.ones_like(S, dtype=torch.bool, device=device)
            if use_min_child:
                thr = float(min_child_weight)
                valid &= (N0 >= thr) & (N1 >= thr)
            if use_min_gain:
                thr_g = float(min_split_gain)
                valid &= (S >= thr_g)
        else:
            valid = torch.ones_like(S, dtype=torch.bool, device=device)

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

            Mf3    = Mf.unsqueeze(-1).expand(K1, nodes, lanes)   # [K1, n, 32]
            valid3 = valid & Mf3                                 # only candidates that both route here AND satisfy constraints

            # If no valid candidate anywhere for this fold, skip
            if not valid3.any().item():
                continue

            # Which leaves have at least one valid candidate for this fold? [nodes] bool
            valid_leaf = valid3.any(dim=0).any(dim=1)  # any over (k, lane) for each leaf

            # Lexicographic key:
            #   high 32 bits = S_bits (score compare)
            #   low  32 bits = (0xFFFFFFFF - k)  -> earlier k wins ties
            key_hi = S_bits.to(torch.int64)
            key_lo = (0xFFFFFFFF - k_idx)  # fits in int64
            key    = (key_hi << 32) | key_lo
            key    = torch.where(valid3, key, torch.full_like(key, min_i64))

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

            # Clip leaf values if max_delta_step > 0 (mirrors CUDA kernel)
            if max_delta_step > 0.0:
                V0_sel = V0_sel.clamp(-max_delta_step, max_delta_step)
                V1_sel = V1_sel.clamp(-max_delta_step, max_delta_step)

            # Only write leaves that had at least one valid candidate
            mask = valid_leaf
            if not mask.any().item():
                continue

            idx = nodes_ar[mask]  # [num_valid]

            # Quantize to int32 (CUDA uses trunc toward zero)
            v0_q = torch.trunc(qs_sel * V0_sel).to(torch.int32)
            v1_q = torch.trunc(qs_sel * V1_sel).to(torch.int32)

            V_ts[f, 2 * idx    ] = v0_q[mask]
            V_ts[f, 2 * idx + 1] = v1_q[mask]

            # Feature indices: I = F[tree_set, 32*k* + lane*] (store as uint16)
            feat_pos      = (k_star.to(torch.int64) * lanes + chosen_lane.to(torch.int64))  # [n]
            vals_i32_all  = (F_row_u.index_select(0, feat_pos) & 0xFFFF).to(torch.int32)    # [n], stay int32
            vals_i32      = vals_i32_all[idx]                                              # [num_valid]

            # Work in int32 scratch row, then cast to uint16 once
            row_i32 = I_ts[f].to(torch.int32)        # [nodes]
            row_i32[idx] = vals_i32                  # write only valid leaves
            I_ts[f].copy_(row_i32.to(torch.uint16))  # final cast, no indexing on uint16

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
        max_idx= V.shape[2]-1

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

                add_idx = (2 * lo + 1 - x).to(torch.long)
                add_idx=torch.clamp(add_idx, max=max_idx)
                P.add_(V[tree_set, f].gather(0, add_idx))

        return P, L_new
    
        # -------- DES: H0 per-era (parents) --------
    def h0_des(self,
            G:  torch.Tensor,   # [N] int16
            W:  torch.Tensor,   # [N] int16
            LE: torch.Tensor,   # [nfolds, N] (u16/u32/u64)
            max_depth: int,
            era_bounds: torch.Tensor | None = None  # [E] int32 (exclusive ends; last==N)
            ) -> torch.Tensor:
        """
        Returns H0e: [nfolds, 2**D, E, 2] int64  (parent stats per fold & era).
        NOTE: This raw layout matches your CUDA launcher. In fit() you permute/slice to [nfolds, E, nodes, 2].
        """
        nfolds, N = int(LE.size(0)), int(LE.size(1))

        # normalize era_ends: ends-only, dtype=int32, on LE.device
        if era_bounds is None:
            era_ends = torch.tensor([N], device=LE.device, dtype=torch.int32)
        else:
            era_ends = era_bounds.to(device=LE.device, dtype=torch.int32, copy=False).contiguous()

        if int(era_ends[-1].item()) != N:
            raise ValueError(f"era_ends[-1] must equal N ({N}), got {int(era_ends[-1].item())}")

        use_cuda = (G.is_cuda or LE.is_cuda or era_ends.is_cuda) and torch.cuda.is_available()
        if use_cuda:
            return kernels.h0_des(G.contiguous(), W.contiguous(), LE.contiguous(), era_ends.contiguous(), int(max_depth)).contiguous()

        # -------- CPU parity (multi-era) --------
        E = int(era_ends.numel())
        nodes_with_last = (1 << max_depth)  # your CPU h0 returns 2**D nodes (includes last row)
        H0e = torch.zeros((nfolds, nodes_with_last, E, 2),
                        dtype=torch.int64, device=LE.device)

        # era starts from [0] + era_ends[:-1]
        era_starts = torch.empty_like(era_ends)
        if E > 0:
            era_starts[0] = 0
            if E > 1:
                era_starts[1:] = era_ends[:-1]

        for ei in range(E):
            e0 = int(era_starts[ei].item())
            e1 = int(era_ends[ei].item())
            if e1 <= e0:
                continue  # empty era segment; leave zeros

            # Slice to this era and reuse your CPU h0()
            G_e  = G[e0:e1].to(device=LE.device, dtype=G.dtype, copy=False).contiguous()
            W_e  = W[e0:e1].to(device=LE.device, dtype=W.dtype, copy=False).contiguous()
            LE_e = LE[:, e0:e1].to(device=LE.device, dtype=LE.dtype, copy=False).contiguous()
            H0_slice = self.h0(G_e, W_e, LE_e, max_depth)  # [nfolds, 2**D, 2] (int64)

            # Place into era dimension
            H0e[:, :, ei, :] = H0_slice

        return H0e.contiguous()


    def h_des(self,
            XS: torch.Tensor,   # [K1, 32*M] uint32/int32  (padded to Np=32*M)
            Y:  torch.Tensor,   # [N] int16
            W:  torch.Tensor,   # [N] int16
            LF: torch.Tensor,   # [K1, N] (u16/u32/u64)
            max_depth: int,
            era_bounds: torch.Tensor | None = None  # [E] int32 (exclusive ends; last==N)
            ) -> torch.Tensor:
        """
        Returns He: [K1, E, (2**D)-1, 2, 32] int64  (left stats per era).
        CPU parity masks XS per-era with 32-bit word masks and reuses your H() math.
        """
        K1, cols_32M = int(XS.size(0)), int(XS.size(1))
        N = int(Y.size(0))
        if era_bounds is None:
            era_ends = torch.tensor([N], device=XS.device, dtype=torch.int32)
        else:
            era_ends = era_bounds.to(device=XS.device, dtype=torch.int32, copy=False).contiguous()

        if int(era_ends[-1].item()) != N:
            raise ValueError(f"era_ends[-1] must equal N ({N}), got {int(era_ends[-1].item())}")

        use_cuda = (XS.is_cuda or Y.is_cuda or LF.is_cuda or era_ends.is_cuda) and torch.cuda.is_available()
        if use_cuda:
            return kernels.h_des(
                XS.contiguous(), Y.contiguous(), W.contiguous(), LF.contiguous(),
                era_ends.contiguous(), int(max_depth)
            ).contiguous()

        # -------- CPU parity (multi-era) --------
        # Shapes & constants
        TORCH_INT64 = torch.int64
        TORCH_INT32 = torch.int32
        nodes = (1 << max_depth) - 1
        M = (cols_32M + 31) // 32  # padded blocks
        assert 32 * M == cols_32M, "XS second dim must be a multiple of 32"

        # Prepare base casted views (no copy if already correct)
        XS64 = XS.to(TORCH_INT64, copy=False).view(K1, M, 32)  # [K1, M, 32]
        Y64  = Y.to(TORCH_INT64,  copy=False)                  # [N]
        W64  = W.to(TORCH_INT64,  copy=False)                  # [N]
        LF64 = LF.to(TORCH_INT64, copy=False)                  # [K1, N]

        # era starts from [0] + era_ends[:-1]
        E = int(era_ends.numel())
        era_starts = torch.empty_like(era_ends)
        if E > 0:
            era_starts[0] = 0
            if E > 1:
                era_starts[1:] = era_ends[:-1]

        # Output: [K1, E, nodes, 2, 32]
        He = torch.zeros((K1, E, nodes, 2, 32), dtype=TORCH_INT64, device=XS.device)

        # Common lane-bit constants for unpack
        bit_ids = torch.arange(32, dtype=TORCH_INT64, device=XS.device)
        bit_masks = (1 << bit_ids).view(1, 1, 1, 32)  # broadcast over [K1,M,32,32]

        # Precompute block starts (s_b = 32*b)
        b_idx = torch.arange(M, dtype=TORCH_INT64, device=XS.device)
        block_start = (b_idx * 32)  # [M]

        for ei in range(E):
            e0 = int(era_starts[ei].item())
            e1 = int(era_ends[ei].item())
            if e1 <= e0:
                continue  # empty era

            # ----- build per-block 32-bit masks for this era -----
            # k0 = clamp(e0 - s_b, 0, 32); k1 = clamp(e1 - s_b, 0, 32); len = clamp(k1-k0, 0, 32)
            k0 = torch.clamp(e0 - block_start, min=0, max=32).to(TORCH_INT64)
            k1 = torch.clamp(e1 - block_start, min=0, max=32).to(TORCH_INT64)
            span = torch.clamp(k1 - k0,       min=0, max=32).to(TORCH_INT64)
            # mask_b = ((1 << span) - 1) << k0
            ones = torch.ones_like(span, dtype=TORCH_INT64)
            mask_base = torch.bitwise_left_shift(ones, span) - 1  # (2^span - 1)
            mask_words = torch.bitwise_left_shift(mask_base, k0).view(1, M, 1)  # [1,M,1], int64

            # ----- apply mask to XS words and compute V (same as H() CPU path) -----
            XS_masked = (XS64 & mask_words)                        # [K1, M, 32]
            V_bits    = ((XS_masked[..., None] & bit_masks) != 0).to(TORCH_INT64)  # [K1,M,32,32]
            V         = V_bits.permute(0, 2, 1, 3).reshape(K1, 32, 32 * M)         # [K1,32,Np]
            if V.size(2) > N:   # trim padded tail
                V = V[:, :, :N].contiguous()                      # [K1,32,N]

            # Broadcasts
            Y_b  = Y64.view(1, 1, N)                      # [1,1,N]
            W_b  = W64.view(1, 1, N)                      # [1,1,N]

            # Allocate per-era accumulators (match H() layout then assign into He[:, ei])
            H_sum = torch.zeros((K1, nodes, 32), dtype=TORCH_INT64, device=XS.device)
            H_cnt = torch.zeros_like(H_sum)

            # depth 0
            H_sum[:, 0, :] = (V * Y_b).sum(dim=2)
            H_cnt[:, 0, :] = (V * W_b).sum(dim=2)

            # depth 1
            tk1 = (LF64 & 1)                             # [K1, N]
            m0  = (tk1 == 0).to(TORCH_INT64).unsqueeze(1)  # [K1,1,N]
            m1  = 1 - m0
            H_sum[:, 1, :] = (V * (Y_b * m0)).sum(dim=2)
            H_cnt[:, 1, :] = (V * (W_b * m0)).sum(dim=2)
            H_sum[:, 2, :] = (V * (Y_b * m1)).sum(dim=2)
            H_cnt[:, 2, :] = (V * (W_b * m1)).sum(dim=2)

            # depth 2
            tk2 = ((LF64 >> 1) & 3).to(torch.long)       # [K1, N]
            oh2 = torch.nn.functional.one_hot(tk2, num_classes=4).to(TORCH_INT64)  # [K1,N,4]
            oh2 = oh2.unsqueeze(1)                       # [K1,1,N,4]
            V4  = V.unsqueeze(-1)                        # [K1,32,N,1]
            sum2 = (V4 * (Y_b.unsqueeze(-1) * oh2)).sum(dim=2)   # [K1,32,4]
            cnt2 = (V4 * (W_b.unsqueeze(-1) * oh2)).sum(dim=2)   # [K1,32,4]
            H_sum[:, 3:7, :] = sum2.permute(0, 2, 1).contiguous()  # [K1,4,32]
            H_cnt[:, 3:7, :] = cnt2.permute(0, 2, 1).contiguous()

            # depths >= 3
            if max_depth > 3:
                F32 = K1 * 32
                V_flat = V.view(F32, N)
                Y_flat = Y64.view(1, N).expand(F32, N)
                W_flat = W64.view(1, N).expand(F32, N)
                for d in range(3, max_depth):
                    s = (d * (d - 1)) // 2
                    base = (1 << d) - 1
                    maskd = (1 << d) - 1
                    idx_fn = (((LF64 >> s) & maskd) + base).to(torch.long)        # [K1, N]
                    idx_b  = idx_fn.repeat_interleave(32, dim=0)                  # [K1*32, N]

                    out_sum = torch.zeros((F32, nodes), dtype=TORCH_INT64, device=XS.device)
                    out_cnt = torch.zeros_like(out_sum)
                    out_sum.scatter_add_(1, idx_b, (V_flat * Y_flat))
                    out_cnt.scatter_add_(1, idx_b, (V_flat * W_flat))

                    H_sum += out_sum.view(K1, 32, nodes).permute(0, 2, 1)
                    H_cnt += out_cnt.view(K1, 32, nodes).permute(0, 2, 1)

            # stack into He (add channel dim)
            He[:, ei, :, 0, :] = H_sum
            He[:, ei, :, 1, :] = H_cnt

        return He.contiguous()

    def cut_des(
        self,
        F:  torch.Tensor,   # [rounds, 32*K1] uint16
        FST:torch.Tensor,   # [rounds, K1, D] uint8
        H:  torch.Tensor,   # [K1, E, nodes, 2, 32] (lane-resolved, ERA-MAJOR)
        H0: torch.Tensor,   # [K0, nodes, E, 2]     (parent totals per era)
        V:  torch.Tensor,   # [rounds, K0, 2*nodes] int32
        I:  torch.Tensor,   # [rounds, K0, nodes]   uint16
        *,
        tree_set: int,
        L2: float,
        lr: float,
        qgrad_bits: int,
        max_depth: int,
        min_child_weight: float = 20.0,
        min_split_gain: float = 0.0,
        max_delta_step: float = 0.0,
    ):
        use_cuda = any(t.is_cuda for t in (F, FST, H, H0, V, I)) and torch.cuda.is_available()
        if use_cuda:
            from packboost.cuda import kernels
            kernels.cut_cuda_des(
                F.contiguous(), FST.contiguous(),
                H.contiguous(), H0.contiguous(),     # NOTE: no permute; kernel expects [K0,nodes,E,2]
                V, I,
                int(tree_set), float(L2), float(lr),
                int(qgrad_bits), int(max_depth),
                float(min_child_weight), float(min_split_gain),
                float(max_delta_step),
            )
            return V, I

        # ---------------- CPU fallback (bit-for-bit tie logic with warp emulation) ----------------
        assert H.dtype == torch.int64 and H0.dtype == torch.int64
        K1, E, nodes, two, lanes = H.shape
        K0 = int(H0.shape[0])
        assert two == 2 and lanes == 32, "H must be [K1,E,nodes,2,32]"
        assert H0.shape == (K0, nodes, E, 2), "H0 must be [K0,nodes,E,2]"
        assert V.shape == (F.shape[0], K0, 2*nodes)
        assert I.shape == (F.shape[0], K0, nodes)

        device = H.device
        D = int(FST.shape[2])
        assert nodes == (1 << max_depth) - 1 and D >= max_depth, "depth mismatch"

        use_min_child = (min_child_weight > 0.0)
        use_min_gain  = (min_split_gain  > 0.0)

        # Only the selected tree_set row of F matters here
        K1x32 = int(F.shape[1])
        assert K1x32 == 32 * K1
        F_row = (F[int(tree_set)].to(torch.int64) & 0xFFFF)  # [32*K1] as unsigned

        # depth per leaf
        leaf_ids = torch.arange(nodes, device=device, dtype=torch.long)
        depth = torch.floor(
            leaf_ids.to(torch.float32) + 1.0
        ).log2().to(torch.long)  # [nodes]

        # per-leaf scalars
        L2_eff = (
            torch.tensor(L2, dtype=torch.float32, device=device)
            * torch.pow(torch.tensor(2.0, device=device), 5.0 - depth.to(torch.float32))
        )
        qscale = (
            torch.tensor(lr, dtype=torch.float32, device=device)
            * float(1 << (31 - int(qgrad_bits)))
            * torch.pow(
                torch.tensor(2.0, device=device),
                -(float(max_depth) - depth.to(torch.float32)),
            )
        )

        # fold assignment per (k, node): FST[tree_set, k, depth[node]]
        tf_all = FST[int(tree_set)][:, depth].to(torch.long)  # [K1, nodes]

        for n in range(nodes):
            d   = int(depth[n])
            L2n = float(L2_eff[n].item())
            qsn = float(qscale[n].item())

            # lane-local best per fold
            best_dir   = torch.full(
                (32, K0), float("-inf"), dtype=torch.float32, device=device
            )
            best_sbits = torch.full(
                (32, K0), torch.iinfo(torch.int32).min, dtype=torch.int32, device=device
            )
            best_vl    = torch.zeros((32, K0), dtype=torch.int32, device=device)
            best_vr    = torch.zeros((32, K0), dtype=torch.int32, device=device)
            best_k     = torch.full((32, K0), -1, dtype=torch.int32, device=device)

            for lane in range(32):
                # IMPORTANT: each lane evaluates ALL k (matches CUDA), not striped k
                for k in range(K1):
                    f = int(tf_all[k, n])
                    if not (0 <= f < K0):
                        continue

                    # Per-era stats for this lane
                    G0 = H[k, :, n, 0, lane].to(torch.float32)  # [E]
                    N0 = H[k, :, n, 1, lane].to(torch.float32)  # [E]
                    GT = H0[f, n, :, 0].to(torch.float32)       # [E]
                    HT = H0[f, n, :, 1].to(torch.float32)       # [E]

                    G1 = GT - G0
                    N1 = HT - N0

                    # Directional score (per-era sign of V1 - V0; only eras with N0>0,N1>0)
                    valid = (N0 > 0.0) & (N1 > 0.0)
                    eras_used = int(valid.sum().item())
                    if eras_used > 0:
                        V0 = G0 / (N0 + L2n)
                        V1 = G1 / (N1 + L2n)
                        sum_dir = torch.sign(V1 - V0)[valid].sum().abs().item()
                        dir_score = float(sum_dir) / float(eras_used)
                    else:
                        dir_score = 0.0

                    # classic pooled gain S using lane-local left and parent pooled across eras
                    G0_tot = G0.sum().item()
                    N0_tot = N0.sum().item()
                    GT_tot = GT.sum().item()
                    HT_tot = HT.sum().item()
                    G1_tot = GT_tot - G0_tot
                    N1_tot = HT_tot - N0_tot

                    V0f = (G0_tot / (N0_tot + L2n)) if N0_tot > 0.0 else 0.0
                    V1f = (G1_tot / (N1_tot + L2n)) if N1_tot > 0.0 else 0.0
                    S    = (G0_tot * V0f) + (G1_tot * V1f)

                    # min_child_weight / min_split_gain gating (match CUDA semantics)
                    if use_min_child:
                        if (N0_tot < float(min_child_weight)) or (N1_tot < float(min_child_weight)):
                            continue
                    if use_min_gain:
                        if S < float(min_split_gain):
                            continue

                    # float32 bit-pattern tie value
                    S_bits = (
                        torch.tensor([S], dtype=torch.float32, device=device)
                        .view(torch.int32)[0]
                        .item()
                    )

                    # (dir, S_bits) lexicographic compare
                    if (
                        dir_score > best_dir[lane, f].item()
                        or (
                            dir_score == best_dir[lane, f].item()
                            and S_bits > best_sbits[lane, f].item()
                        )
                    ):
                        best_dir  [lane, f] = dir_score
                        best_sbits[lane, f] = S_bits
                        cv0 = max(-max_delta_step, min(max_delta_step, V0f)) if max_delta_step > 0.0 else V0f
                        cv1 = max(-max_delta_step, min(max_delta_step, V1f)) if max_delta_step > 0.0 else V1f
                        best_vl   [lane, f] = int(
                            torch.trunc(torch.tensor(qsn * cv0, device=device)).item()
                        )
                        best_vr   [lane, f] = int(
                            torch.trunc(torch.tensor(qsn * cv1, device=device)).item()
                        )
                        best_k    [lane, f] = k

            # reduce across lanes per fold: max(dir) -> max(S_bits) -> highest lane id
            for f in range(K0):
                dir_col = best_dir[:, f]
                if not torch.isfinite(dir_col).any():
                    continue

                max_dir = torch.nan_to_num(dir_col, nan=-float("inf")).max().item()
                m_dir = (best_dir[:, f] == max_dir)

                sbits_masked = torch.where(
                    m_dir,
                    best_sbits[:, f],
                    torch.full_like(best_sbits[:, f], torch.iinfo(torch.int32).min),
                )
                max_sbits = sbits_masked.max().item()
                winners = m_dir & (best_sbits[:, f] == max_sbits)
                if not winners.any():
                    continue

                winner_lane = int(
                    torch.nonzero(winners, as_tuple=False).flatten().max().item()
                )
                k_star = int(best_k[winner_lane, f].item())
                if k_star < 0:
                    continue

                V[int(tree_set), f, 2 * n    ] = int(best_vl[winner_lane, f].item())
                V[int(tree_set), f, 2 * n + 1] = int(best_vr[winner_lane, f].item())

                feat_pos = k_star * 32 + winner_lane
                I[int(tree_set), f, n] = (F_row[feat_pos] & 0xFFFF).to(torch.uint16)

        return V, I
                            
    def save(self, path):
        """Save with PyTorch's native ZIP compression."""
        if not path.endswith('.pth'):
            path = path + '.pth'
        
        state = {
            "V": self.V.cpu(),
            "I": self.I.cpu(),
            "FST": self.FST.cpu(),
            "tree_set": self.tree_set,
            "max_depth": self.max_depth,
            "nfolds": self.nfolds,
            "nfeatsets": self.nfeatsets,
            "device": self.device,
            "feature_name" : self.feature_name,
            "comment" : self.comment,
            "train_N": getattr(self, 'train_N', None),
        }
        
        torch.save(state, path, _use_new_zipfile_serialization=True)
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Saved PackBoost model to {path} ({file_size_mb:.2f} MB)")
    
    def load(self, path, device=None):
        """Load PyTorch compressed format."""
        if not path.endswith('.pth') and os.path.exists(path + '.pth'):
            path = path + '.pth'
        
        device = device or self.device
        state = torch.load(path, map_location=device)
        
        self.V         = state["V"]
        self.I         = state["I"]
        self.FST       = state["FST"]
        self.tree_set  = state["tree_set"]
        self.max_depth = state["max_depth"]
        self.nfolds    = state["nfolds"]
        self.nfeatsets = state["nfeatsets"]
        self.device    = device
        self.feature_name  = state.get("feature_name",None)
        self.comment = state.get("comment","")
        self.train_N   = state.get("train_N", None)
        
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Loaded PackBoost model from {path} ({file_size_mb:.2f} MB)")
        return self

