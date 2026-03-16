import torch

from packboost.core import PackBoost


def _encode_cuts_reference_words(X: torch.Tensor) -> torch.Tensor:
    """Reference packer for X[N, F] int8 -> XB[4*F, M] uint32."""
    assert X.dtype == torch.int8
    N, F = X.shape
    M = (N + 31) // 32

    out = torch.zeros((4 * F, M), dtype=torch.uint32)
    for n in range(N):
        w = n // 32
        b = n % 32
        bit = 1 << b
        for f in range(F):
            v = int(X[n, f].item())
            for t in range(4):
                if v > t:
                    out[4 * f + t, w] = (out[4 * f + t, w].to(torch.int64) | bit).to(torch.uint32)
    return out


def _build_test_input(n: int, f: int) -> torch.Tensor:
    X = torch.empty((n, f), dtype=torch.int8)
    X[:, 0] = torch.arange(n, dtype=torch.int16).remainder(5).to(torch.int8)
    X[:, 1] = (4 - torch.arange(n, dtype=torch.int16).remainder(5)).to(torch.int8)
    X[:, 2] = torch.full((n,), 2, dtype=torch.int8)
    return X


def _assert_last_word_padding_is_zero(encoded: torch.Tensor, m: int, rows: int, label: str) -> None:
    # Word m-1 corresponds to samples 96..127 when N=100, so only bits 0..3 are valid.
    for row in range(rows):
        last_word = int(encoded[row, m - 1].item())
        assert (last_word >> 4) == 0, (
            f"{label} row {row}: expected upper 28 padded bits to be zero, got 0x{last_word:08x}"
        )


def test_encode_cuts_cpu_word_extraction_and_padding_n100_m4():
    # N=100 gives M=4 words; last word has 4 valid bits and 28 padded bits.
    N, F = 100, 3
    M = (N + 31) // 32
    assert M == 4

    X = _build_test_input(N, F)

    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    assert out_cpu.dtype == torch.uint32
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)
    _assert_last_word_padding_is_zero(out_cpu, M, 4 * F, "CPU")


def test_encode_cuts_cuda_word_extraction_and_padding_n100_m4():
    # Intentionally fail if CUDA is unavailable so this test explicitly enforces CUDA presence.
    assert torch.cuda.is_available(), "CUDA is required for this test but is not available"

    N, F = 100, 3
    M = (N + 31) // 32
    assert M == 4

    X = _build_test_input(N, F)
    expected = _encode_cuts_reference_words(X)

    pack_gpu = PackBoost(device="cuda")
    out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

    assert out_gpu.shape == (4 * F, M)
    assert out_gpu.dtype == torch.uint32
    torch.testing.assert_close(out_gpu, expected, rtol=0, atol=0)
    _assert_last_word_padding_is_zero(out_gpu, M, 4 * F, "CUDA")


# ==================== Edge Case Tests ====================


def test_encode_cuts_single_sample():
    """Test with N=1: single sample with maximum padding (31 bits)."""
    N, F = 1, 5
    M = (N + 31) // 32
    assert M == 1

    X = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int8)
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)

    # Verify only bit 0 is set, bits 1-31 should be padding (zero)
    for row in range(4 * F):
        word_val = int(out_cpu[row, 0].item())
        # Only bit 0 can be set, so value should be 0 or 1
        assert word_val in [0, 1], f"Row {row}: expected only bit 0 possible, got 0x{word_val:08x}"


def test_encode_cuts_exact_word_boundary():
    """Test with N=32: exactly one word, no padding needed."""
    N, F = 32, 4
    M = (N + 31) // 32
    assert M == 1

    X = torch.zeros((N, F), dtype=torch.int8)
    X[:, 0] = torch.arange(N, dtype=torch.int8) % 5
    X[:, 1] = 2  # All threshold 2
    X[:, 2] = 4  # All threshold 4
    X[:, 3] = 0  # All zeros

    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)


def test_encode_cuts_minimal_padding():
    """Test with N=33: minimal padding (31 bits in last word)."""
    N, F = 33, 3
    M = (N + 31) // 32
    assert M == 2

    X = _build_test_input(N, F)
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)

    # Verify that only bit 0 is used in the last word
    for row in range(4 * F):
        last_word = int(out_cpu[row, M - 1].item())
        assert (last_word >> 1) == 0, f"Row {row}: expected only bit 0 in last word, got 0x{last_word:08x}"


def test_encode_cuts_single_feature():
    """Test with F=1: single feature, multiple samples."""
    N, F = 100, 1
    M = (N + 31) // 32

    X = torch.arange(N, dtype=torch.int8).unsqueeze(1) % 5
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    assert out_cpu.shape == (4, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)


def test_encode_cuts_all_zeros():
    """Test with all samples having value 0: no bits should be set."""
    N, F = 64, 5
    M = (N + 31) // 32

    X = torch.zeros((N, F), dtype=torch.int8)
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    # All zeros means v > t is always false for t in [0,1,2,3]
    # So all output should be zero
    torch.testing.assert_close(out_cpu, torch.zeros_like(out_cpu), rtol=0, atol=0)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)


def test_encode_cuts_all_max_value():
    """Test with all samples having value 4: all threshold bits should be set."""
    N, F = 64, 5
    M = (N + 31) // 32

    X = torch.full((N, F), 4, dtype=torch.int8)
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    # All 4s means v > t is true for all t in [0,1,2,3]
    # For N=64, M=2, all 32 bits should be set in each word
    expected_word = 0xFFFFFFFF  # All 32 bits set
    for f in range(F):
        for t in range(4):
            for w in range(M):
                assert out_cpu[4 * f + t, w] == expected_word, \
                    f"Feature {f}, threshold {t}, word {w}: expected all bits set"
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)


def test_encode_cuts_threshold_boundaries():
    """Test values exactly at each threshold (0, 1, 2, 3, 4)."""
    N, F = 160, 1  # 5 words, 32 samples per threshold value
    M = (N + 31) // 32
    assert M == 5

    X = torch.zeros((N, F), dtype=torch.int8)
    # Word 0: all zeros (value 0)
    X[0:32, 0] = 0
    # Word 1: all ones (value 1)
    X[32:64, 0] = 1
    # Word 2: all twos (value 2)
    X[64:96, 0] = 2
    # Word 3: all threes (value 3)
    X[96:128, 0] = 3
    # Word 4: all fours (value 4)
    X[128:160, 0] = 4

    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)

    # Verify expected bit patterns
    # Threshold t=0: v > 0 → true for v in [1,2,3,4]
    assert out_cpu[0, 0] == 0x00000000  # v=0: false
    assert out_cpu[0, 1] == 0xFFFFFFFF  # v=1: true
    assert out_cpu[0, 2] == 0xFFFFFFFF  # v=2: true
    assert out_cpu[0, 3] == 0xFFFFFFFF  # v=3: true
    assert out_cpu[0, 4] == 0xFFFFFFFF  # v=4: true

    # Threshold t=1: v > 1 → true for v in [2,3,4]
    assert out_cpu[1, 0] == 0x00000000  # v=0: false
    assert out_cpu[1, 1] == 0x00000000  # v=1: false
    assert out_cpu[1, 2] == 0xFFFFFFFF  # v=2: true
    assert out_cpu[1, 3] == 0xFFFFFFFF  # v=3: true
    assert out_cpu[1, 4] == 0xFFFFFFFF  # v=4: true

    # Threshold t=2: v > 2 → true for v in [3,4]
    assert out_cpu[2, 0] == 0x00000000  # v=0: false
    assert out_cpu[2, 1] == 0x00000000  # v=1: false
    assert out_cpu[2, 2] == 0x00000000  # v=2: false
    assert out_cpu[2, 3] == 0xFFFFFFFF  # v=3: true
    assert out_cpu[2, 4] == 0xFFFFFFFF  # v=4: true

    # Threshold t=3: v > 3 → true for v in [4]
    assert out_cpu[3, 0] == 0x00000000  # v=0: false
    assert out_cpu[3, 1] == 0x00000000  # v=1: false
    assert out_cpu[3, 2] == 0x00000000  # v=2: false
    assert out_cpu[3, 3] == 0x00000000  # v=3: false
    assert out_cpu[3, 4] == 0xFFFFFFFF  # v=4: true


def test_encode_cuts_alternating_pattern():
    """Test alternating pattern to verify bit indexing within words."""
    N, F = 64, 2
    M = (N + 31) // 32

    X = torch.zeros((N, F), dtype=torch.int8)
    # Feature 0: alternating 0 and 4
    X[::2, 0] = 0
    X[1::2, 0] = 4
    # Feature 1: alternating 4 and 0
    X[::2, 1] = 4
    X[1::2, 1] = 0

    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)

    # For feature 0: odd bits set (0xAAAAAAAA pattern for all 4 thresholds)
    # For feature 1: even bits set (0x55555555 pattern for all 4 thresholds)
    odd_bits = 0xAAAAAAAA
    even_bits = 0x55555555

    for t in range(4):
        assert out_cpu[4 * 0 + t, 0] == odd_bits
        assert out_cpu[4 * 0 + t, 1] == odd_bits
        assert out_cpu[4 * 1 + t, 0] == even_bits
        assert out_cpu[4 * 1 + t, 1] == even_bits


def test_encode_cuts_large_n():
    """Test with large N to verify scalability."""
    N, F = 10000, 10
    M = (N + 31) // 32

    X = _build_test_input(N, F)
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)

    # Verify padding in last word
    samples_in_last_word = N % 32
    if samples_in_last_word > 0:
        for row in range(4 * F):
            last_word = int(out_cpu[row, M - 1].item())
            # Bits beyond samples_in_last_word should be zero
            mask = (1 << samples_in_last_word) - 1
            assert (last_word & ~mask) == 0, \
                f"Row {row}: padding bits not zero in last word, got 0x{last_word:08x}"


def test_encode_cuts_large_f():
    """Test with large F (many features)."""
    N, F = 200, 100
    M = (N + 31) // 32

    X = _build_test_input(N, F)
    expected = _encode_cuts_reference_words(X)

    pack_cpu = PackBoost(device="cpu")
    out_cpu = pack_cpu.encode_cuts(X)

    assert out_cpu.shape == (4 * F, M)
    torch.testing.assert_close(out_cpu, expected, rtol=0, atol=0)


def test_encode_cuts_cpu_vs_cuda_single_sample():
    """CPU vs CUDA parity for single sample edge case."""
    if not torch.cuda.is_available():
        return  # Skip silently if CUDA not available

    N, F = 1, 10
    X = torch.randint(0, 5, (N, F), dtype=torch.int8)

    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    out_cpu = pack_cpu.encode_cuts(X)
    out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu, rtol=0, atol=0)


def test_encode_cuts_cpu_vs_cuda_exact_boundary():
    """CPU vs CUDA parity for exact word boundary (N=32, 64, 96)."""
    if not torch.cuda.is_available():
        return  # Skip silently if CUDA not available

    for N in [32, 64, 96]:
        F = 7
        X = torch.randint(0, 5, (N, F), dtype=torch.int8)

        pack_cpu = PackBoost(device="cpu")
        pack_gpu = PackBoost(device="cuda")

        out_cpu = pack_cpu.encode_cuts(X)
        out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

        torch.testing.assert_close(out_cpu, out_gpu, rtol=0, atol=0,
                                 msg=f"CPU vs GPU mismatch for N={N}")


def test_encode_cuts_cpu_vs_cuda_all_zeros():
    """CPU vs CUDA parity for all zeros."""
    if not torch.cuda.is_available():
        return  # Skip silently if CUDA not available

    N, F = 100, 5
    X = torch.zeros((N, F), dtype=torch.int8)

    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    out_cpu = pack_cpu.encode_cuts(X)
    out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu, rtol=0, atol=0)
    torch.testing.assert_close(out_cpu, torch.zeros_like(out_cpu), rtol=0, atol=0)


def test_encode_cuts_cpu_vs_cuda_all_max():
    """CPU vs CUDA parity for all max values."""
    if not torch.cuda.is_available():
        return  # Skip silently if CUDA not available

    N, F = 100, 5
    X = torch.full((N, F), 4, dtype=torch.int8)

    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    out_cpu = pack_cpu.encode_cuts(X)
    out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu, rtol=0, atol=0)


def test_encode_cuts_cpu_vs_cuda_threshold_boundaries():
    """CPU vs CUDA parity for threshold boundary values."""
    if not torch.cuda.is_available():
        return  # Skip silently if CUDA not available

    N, F = 160, 5
    X = torch.zeros((N, F), dtype=torch.int8)
    for f in range(F):
        X[0:32, f] = 0
        X[32:64, f] = 1
        X[64:96, f] = 2
        X[96:128, f] = 3
        X[128:160, f] = 4

    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    out_cpu = pack_cpu.encode_cuts(X)
    out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu, rtol=0, atol=0)


def test_encode_cuts_cpu_vs_cuda_large():
    """CPU vs CUDA parity for large inputs."""
    if not torch.cuda.is_available():
        return  # Skip silently if CUDA not available

    N, F = 10000, 50
    X = torch.randint(0, 5, (N, F), dtype=torch.int8)

    pack_cpu = PackBoost(device="cpu")
    pack_gpu = PackBoost(device="cuda")

    out_cpu = pack_cpu.encode_cuts(X)
    out_gpu = pack_gpu.encode_cuts(X.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu, rtol=0, atol=0)
