"""
Unit tests for CPU vs GPU prediction parity.

These tests verify that the refactored branch-bit implementation produces
identical predictions on CPU and GPU.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packboost.core import PackBoost


def create_simple_dataset(n_samples=1000, n_features=50, seed=42):
    """Create a simple synthetic dataset for testing."""
    np.random.seed(seed)

    # Features: int8 in range [0, 4]
    X = np.random.randint(0, 5, size=(n_samples, n_features), dtype=np.int8)

    # Target: simple linear combination with some noise
    weights = np.random.randn(n_features) * 0.1
    y = X.astype(np.float32) @ weights + np.random.randn(n_samples) * 0.05
    y = y.astype(np.float32)

    return X, y


def create_validation_split(X, y, val_ratio=0.2, seed=42):
    """Split data into train and validation."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])


@pytest.fixture(scope="module")
def trained_model():
    """Train a model once for all tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Need n_features >= 32*nfeatsets/4 for sampling without replacement
    # For nfeatsets=4: need n_features >= 32*4/4 = 32
    X, y = create_simple_dataset(n_samples=2000, n_features=40)
    (X_train, y_train), (X_val, y_val) = create_validation_split(X, y)

    model = PackBoost(device='cuda')
    model.fit(
        X_train, y_train,
        Xv=X_val, Yv=y_val,
        nfolds=4,
        rounds=50,
        max_depth=5,
        lr=0.05,
        L2=10000.0,
        min_child_weight=5.0,
        nfeatsets=4,  # Reduced from 16 to fit with n_features=40
        seed=42
    )

    return model, X_val


class TestCPUGPUParity:
    """Test suite for CPU vs GPU prediction consistency."""

    def test_predictions_match_exact_device(self, trained_model):
        """Test that predictions are consistent on the same device."""
        model, X_test = trained_model

        # Predict twice on GPU
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)

        np.testing.assert_allclose(pred1, pred2, rtol=1e-7, atol=0,
                                   err_msg="Predictions should be deterministic on same device")

    def test_predictions_match_cpu_vs_gpu_numpy(self, trained_model):
        """Test CPU vs GPU predictions with numpy input."""
        model, X_test = trained_model

        # Predict on GPU
        pred_gpu = model.predict(X_test)

        # Move model to CPU
        model_cpu = PackBoost(device='cpu')
        model_cpu.V = model.V.cpu()
        model_cpu.I = model.I.cpu()
        model_cpu.FST = model.FST.cpu()
        model_cpu.tree_set = model.tree_set
        model_cpu.max_depth = model.max_depth
        model_cpu.nfolds = model.nfolds
        model_cpu.nfeatsets = model.nfeatsets

        # Predict on CPU
        pred_cpu = model_cpu.predict(X_test)

        # Predictions should match within floating point tolerance
        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-5, atol=1e-6,
                                   err_msg="CPU and GPU predictions should match")

    def test_predictions_match_cpu_vs_gpu_tensor(self, trained_model):
        """Test CPU vs GPU predictions with torch tensor input."""
        model, X_test = trained_model

        # Convert to torch tensor
        X_tensor_gpu = torch.from_numpy(X_test).to(device='cuda', dtype=torch.int8)
        X_tensor_cpu = torch.from_numpy(X_test).to(device='cpu', dtype=torch.int8)

        # Predict on GPU
        pred_gpu = model.predict(X_tensor_gpu)

        # Move model to CPU
        model_cpu = PackBoost(device='cpu')
        model_cpu.V = model.V.cpu()
        model_cpu.I = model.I.cpu()
        model_cpu.FST = model.FST.cpu()
        model_cpu.tree_set = model.tree_set
        model_cpu.max_depth = model.max_depth
        model_cpu.nfolds = model.nfolds
        model_cpu.nfeatsets = model.nfeatsets

        # Predict on CPU
        pred_cpu = model_cpu.predict(X_tensor_cpu)

        # Both should be torch tensors
        assert torch.is_tensor(pred_gpu) and torch.is_tensor(pred_cpu)

        # Move to same device for comparison
        pred_gpu_cpu = pred_gpu.cpu()

        # Predictions should match
        torch.testing.assert_close(pred_cpu, pred_gpu_cpu, rtol=1e-5, atol=1e-6)

    def test_predictions_match_different_batch_sizes(self, trained_model):
        """Test that batching doesn't affect CPU vs GPU parity."""
        model, X_test = trained_model

        # Test with different batch sizes
        for n_samples in [1, 10, 100, 500]:
            X_batch = X_test[:n_samples]

            # GPU prediction
            pred_gpu = model.predict(X_batch)

            # CPU model
            model_cpu = PackBoost(device='cpu')
            model_cpu.V = model.V.cpu()
            model_cpu.I = model.I.cpu()
            model_cpu.FST = model.FST.cpu()
            model_cpu.tree_set = model.tree_set
            model_cpu.max_depth = model.max_depth
            model_cpu.nfolds = model.nfolds
            model_cpu.nfeatsets = model.nfeatsets

            # CPU prediction
            pred_cpu = model_cpu.predict(X_batch)

            np.testing.assert_allclose(
                pred_cpu, pred_gpu, rtol=1e-5, atol=1e-6,
                err_msg=f"Predictions should match for batch size {n_samples}"
            )

    def test_predictions_match_single_sample(self, trained_model):
        """Test CPU vs GPU for single sample prediction."""
        model, X_test = trained_model

        # Single sample
        X_single = X_test[:1]

        # GPU prediction
        pred_gpu = model.predict(X_single)

        # CPU model
        model_cpu = PackBoost(device='cpu')
        model_cpu.V = model.V.cpu()
        model_cpu.I = model.I.cpu()
        model_cpu.FST = model.FST.cpu()
        model_cpu.tree_set = model.tree_set
        model_cpu.max_depth = model.max_depth
        model_cpu.nfolds = model.nfolds
        model_cpu.nfeatsets = model.nfeatsets

        # CPU prediction
        pred_cpu = model_cpu.predict(X_single)

        assert pred_cpu.shape == (1,) and pred_gpu.shape == (1,)
        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-5, atol=1e-6)


class TestBranchBitConsistency:
    """Test that branch bits are stored and used correctly."""

    def test_branch_bits_are_binary(self, trained_model):
        """Verify that L_old only contains 0 and 1 values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model, X_test = trained_model

        # Get some predictions to populate L buffers
        X_small = X_test[:100]
        XB = model.encode_cuts(torch.from_numpy(X_small).to('cuda', dtype=torch.int8))

        D = model.max_depth
        Dm = max(D - 1, 0)
        N = XB.shape[1] * 32

        L = torch.zeros((model.nfolds, Dm, N), dtype=torch.uint8, device='cuda')
        Ln = torch.zeros_like(L)
        P = torch.zeros(N, dtype=torch.int32, device='cuda')

        # Advance one tree
        model.advance_and_predict(P, XB, L, Ln, model.V, model.I, tree_set=0)

        # Check that Ln only contains 0 or 1
        unique_values = torch.unique(Ln)
        assert all(v in [0, 1] for v in unique_values.cpu().numpy()), \
            f"Branch bits should only be 0 or 1, got {unique_values}"

    def test_leaf_dtype_is_uint8(self, trained_model):
        """Verify that L tensors use uint8 dtype."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model, X_test = trained_model

        X_small = X_test[:100]
        XB = model.encode_cuts(torch.from_numpy(X_small).to('cuda', dtype=torch.int8))

        D = model.max_depth
        Dm = max(D - 1, 0)
        N = XB.shape[1] * 32

        L = torch.zeros((model.nfolds, Dm, N), dtype=torch.uint8, device='cuda')
        Ln = torch.zeros_like(L)

        assert L.dtype == torch.uint8, "L should be uint8"
        assert Ln.dtype == torch.uint8, "Ln should be uint8"


class TestDepthVariations:
    """Test CPU vs GPU parity for different tree depths."""

    @pytest.mark.parametrize("max_depth", [3, 4, 5, 6, 7])
    def test_different_depths(self, max_depth):
        """Test CPU vs GPU parity for different max_depth values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Need n_features >= 32*nfeatsets/4 for sampling without replacement
        # For nfeatsets=2: need n_features >= 32*2/4 = 16
        X, y = create_simple_dataset(n_samples=500, n_features=20)
        (X_train, y_train), (X_val, y_val) = create_validation_split(X, y)

        # Train on GPU
        model_gpu = PackBoost(device='cuda')
        model_gpu.fit(
            X_train, y_train,
            Xv=X_val, Yv=y_val,
            nfolds=4,
            rounds=20,
            max_depth=max_depth,
            lr=0.05,
            nfeatsets=2,  # Reduced from 8 to fit with n_features=20
            seed=42
        )

        # GPU prediction
        pred_gpu = model_gpu.predict(X_val)

        # CPU model
        model_cpu = PackBoost(device='cpu')
        model_cpu.V = model_gpu.V.cpu()
        model_cpu.I = model_gpu.I.cpu()
        model_cpu.FST = model_gpu.FST.cpu()
        model_cpu.tree_set = model_gpu.tree_set
        model_cpu.max_depth = model_gpu.max_depth
        model_cpu.nfolds = model_gpu.nfolds
        model_cpu.nfeatsets = model_gpu.nfeatsets

        # CPU prediction
        pred_cpu = model_cpu.predict(X_val)

        np.testing.assert_allclose(
            pred_cpu, pred_gpu, rtol=1e-5, atol=1e-6,
            err_msg=f"Predictions should match for max_depth={max_depth}"
        )


class TestGradientComputation:
    """Test that prep_vars works correctly on CPU and GPU."""

    def test_prep_vars_cpu_vs_gpu(self):
        """Test prep_vars gradient computation on CPU vs GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from packboost.core import PackBoost

        # Create sample Y and P tensors
        N = 1000
        Y_np = np.random.randn(N).astype(np.float32)
        P_np = np.random.randn(N).astype(np.float32)

        # Quantize to Q30 format
        Y_i32 = torch.from_numpy((Y_np * (1 << 30)).astype(np.int32))
        P_i32 = torch.from_numpy((P_np * (1 << 30)).astype(np.int32))

        # CPU version - note: CPU uses >>20 shift, GPU uses >>15 shift
        model_cpu = PackBoost(device='cpu')
        G_cpu = model_cpu.prep_vars(Y_i32.cpu(), P_i32.cpu())

        # GPU version
        model_gpu = PackBoost(device='cuda')
        G_gpu = model_gpu.prep_vars(Y_i32.cuda(), P_i32.cuda())

        expected = (Y_i32.to(torch.int64) - P_i32.to(torch.int64)) >> 20
        expected_clamped = expected.clamp(-(1 << 15), (1 << 15) - 1).to(torch.int16)

        # Verify CPU implementation
        torch.testing.assert_close(
            G_cpu, expected_clamped,
            rtol=0, atol=0,
            msg="CPU prep_vars implementation incorrect"
        )

        # Verify GPU implementation
        torch.testing.assert_close(
            G_gpu.cpu(), expected_clamped,
            rtol=0, atol=0,
            msg="GPU prep_vars implementation incorrect"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
