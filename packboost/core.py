import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor
from packboost.cuda import kernels

class PackBoost(BaseEstimator, RegressorMixin):
    def __init__(self, device='cuda'):
        self.device = device

    def encode_cuts(self, X : Tensor) -> Tensor:
        if X.device != 'cpu' and torch.cuda.is_available():
            return kernels.encode_cuts(X.contiguous())
        N, F = X.shape
        M = (N + 31) // 32
        Np = M*32

        if Np != N:
            pad = torch.zeroes((Np - N, F), dtype=torch.int8)
            X = torch.cat([X, pad], dim=0)
        
        X = X.T.copy()
        thresholds = torch.arange(4, dtype=torch.int8).view(1, 1, 1, 4)
        bitplanes = (X.view(F, M, 32, 1) > thresholds).to(torch.uint32)
        weights = (1 << torch.arange(32, dtype=torch.uint32)).view(1, 1, 32, 1)
        words = (bitplanes * weights).sum(2, dtype=torch.uint64).to(torch.uint32)
        XB = words.permute(2, 0, 1).reshape(4*F, M)
        return XB
