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
    bF, M, rounds = 19, 7, 3

    X_cpu = torch.randint(0, 2 ** 32, (bF, M), dtype=torch.int64).to(torch.uint32)
    Fsch = torch.randint(0, bF, (rounds, 32 * pack_cpu.nfeatsets), dtype=torch.int16).to(
        torch.uint16
    )
    round_idx = 2

    ref = pack_cpu.et_sample_1b(X_cpu, Fsch, round_idx)
    result = pack_gpu.et_sample_1b(X_cpu.to("cuda"), Fsch.to("cuda"), round_idx)
    torch.cuda.synchronize()

    torch.testing.assert_close(result.cpu(), ref)
