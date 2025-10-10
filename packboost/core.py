import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor
from packboost.cuda import kernels

class PackBoost(BaseEstimator, RegressorMixin):
    def __init__(self, device='cuda'):
        self.device = device
        self.nfeatsets = 32
    
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
        max_depth = Dm

        if   max_depth > 8:
            le_dtype = torch.int64
        elif max_depth > 6:
            le_dtype = torch.int32
        else:
            le_dtype = torch.int16

        device = L.device
        d = torch.arange(1, max_depth+1, device=device)
        offsets = (d * (d - 1)) // 2                      # [Dm], int64
        weights = (torch.ones_like(offsets, dtype=le_dtype) << offsets)  # [Dm] le_dtype

        L_bits = (L & 1).to(le_dtype)
        LE = (L_bits * weights.view(1, max_depth, 1)).sum(dim=1, dtype=le_dtype)

        g = (Y.to(torch.int32) - P.to(torch.int32)) >> 19
        G = g.clamp_(-32767, 32767).to(torch.int16)

        return LE.contiguous(), G.contiguous()


    def h0(self, G: torch.Tensor, LE: torch.Tensor, max_depth: int) -> torch.Tensor:
        nfolds, N = LE.shape
        nodes = 1 << max_depth
        if (G.is_cuda or LE.is_cuda) and torch.cuda.is_available():
            return kernels.h0_sm(G.contiguous(), LE.contiguous(), int(max_depth))

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
