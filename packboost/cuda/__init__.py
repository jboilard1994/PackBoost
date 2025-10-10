from __future__ import annotations

from pathlib import Path
from types import ModuleType
import torch
from torch.utils.cpp_extension import load

__all__ = ["kernels"]


def _load_extension() -> ModuleType:
    src_dir = Path(__file__).resolve().parent
    sources = [
        str(src_dir / "kernels.cpp"),
        str(src_dir / "encode_cuts.cu"),
        str(src_dir / "et_sample_1b.cu"),
        str(src_dir / "prep_vars.cu"),
        str(src_dir / "h0.cu"),
        str(src_dir / "repack.cu"),
    ]
    return load(
        name="packboost_cuda_kernels",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


if torch.cuda.is_available():
    kernels = _load_extension()
else:
    class _CudaUnavailable(ModuleType):
        def __init__(self) -> None:
            super().__init__("packboost.cuda.kernels")

        def __getattr__(self, name: str):  # pragma: no cover - error path
            raise RuntimeError(
                "packboost CUDA kernels are unavailable because torch.cuda.is_available() is False"
            )


    kernels = _CudaUnavailable()
