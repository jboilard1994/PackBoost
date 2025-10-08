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
        XB = words.permute(2, 0, 1).reshape(4 * F, M).contiguous()                       # [4F, M]
        return XB


    def et_sample_1b(self, X : Tensor, Fsch : Tensor, round : int) -> Tensor:
        if X.device.type != 'cpu' and torch.cuda.is_available():
            XS = torch.empty((self.nfeatsets, X.shape[1]*32), dtype=torch.uint32, device=X.device)
            return kernels.et_sample_1b(X.contiguous(), XS.contiguous(), Fsch.contiguous(), round)
        M = X.shape[1]
        nfeatsets = self.nfeatsets
        Fs = Fsch[round].view(nfeatsets, 32).to(dtype=torch.long, device=X.device)
        return X.to(torch.int32)[Fs, :].transpose(1, 2).contiguous().view(nfeatsets, M*32).to(torch.uint32)

        