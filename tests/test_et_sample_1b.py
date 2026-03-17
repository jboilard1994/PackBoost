import torch

from packboost.core import PackBoost


def test_et_sample_1b_respects_round_selection():
    pack = PackBoost(device="cpu")
    pack.nfeatsets = 1

    bF, M = 64, 4
    X = torch.arange(bF * M, dtype=torch.int64).reshape(bF, M).to(torch.uint32)

    # round 0 picks feature 0 for all lanes, round 1 picks feature 1 for all lanes
    Fsch = torch.zeros((2, 32), dtype=torch.uint16)
    Fsch[1, :] = 1

    out0 = pack.et_sample_1b(X, Fsch, 0)
    out1 = pack.et_sample_1b(X, Fsch, 1)

    # At word m=0, each lane should equal the selected row index because X[row, 0] == row*4.
    assert torch.all(out0[0, :32] == 0)
    assert torch.all(out1[0, :32] == 4)
    assert not torch.equal(out0, out1)



