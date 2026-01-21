"""
Fold aggregation methods for PackBoost.

This module contains various methods for computing fold weights
based on validation performance.
"""

import torch
from typing import Optional, Dict, Any


def compute_ridge_weights(per_fold_preds: torch.Tensor,
                         targets: torch.Tensor,
                         current_preds: torch.Tensor,
                         alpha: float,
                         device: torch.device) -> torch.Tensor:
    """
    Compute optimal fold weights using ridge regression.
    
    Parameters
    ----------
    per_fold_preds : torch.Tensor [K0, N]
        Per-fold predictions for current round
    targets : torch.Tensor [N]
        Target values (quantized as int32)
    current_preds : torch.Tensor [N]
        Current predictions (int32)
    alpha : float
        Ridge regularization parameter
    device : torch.device
        Device to perform computation on
        
    Returns
    -------
    torch.Tensor [K0]
        Normalized weights that sum to 1
    """
    K0, N = per_fold_preds.shape
    
    # Prepare regression matrices
    X_reg = per_fold_preds.to(torch.float32).t()  # [N, K0]
    y_reg = (targets.to(torch.float32) - current_preds.to(torch.float32))  # [N] residuals
    
    # Ridge regression closed form using Cholesky: w = (X^T X + alpha*I)^(-1) X^T y
    XtX = X_reg.t() @ X_reg  # [K0, K0]
    Xty = X_reg.t() @ y_reg  # [K0]
    I_reg = torch.eye(K0, device=device, dtype=torch.float32)
    
    # Use Cholesky decomposition for efficient solving
    L = torch.linalg.cholesky(XtX + alpha * I_reg)  # [K0, K0]
    w_opt = torch.cholesky_solve(Xty.unsqueeze(-1), L).squeeze(-1)  # [K0]
    
    # Normalize weights so absolute values sum to 1
    w_opt = w_opt / (torch.abs(w_opt).sum() + 1e-10)
    
    return w_opt


def compute_lasso_weights(per_fold_preds: torch.Tensor,
                         targets: torch.Tensor,
                         current_preds: torch.Tensor,
                         alpha: float,
                         device: torch.device,
                         max_iter: int = 1000,
                         tol: float = 1e-4) -> torch.Tensor:
    """
    Compute optimal fold weights using lasso regression (L1 regularization).
    Uses ISTA (Iterative Soft Thresholding Algorithm).
    
    Parameters
    ----------
    per_fold_preds : torch.Tensor [K0, N]
        Per-fold predictions for current round
    targets : torch.Tensor [N]
        Target values (quantized as int32)
    current_preds : torch.Tensor [N]
        Current predictions (int32)
    alpha : float
        Lasso regularization parameter
    device : torch.device
        Device to perform computation on
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    torch.Tensor [K0]
        Normalized weights that sum to 1
    """
    K0, N = per_fold_preds.shape
    
    # Prepare regression matrices
    X_reg = per_fold_preds.to(torch.float32).t()  # [N, K0]
    y_reg = (targets.to(torch.float32) - current_preds.to(torch.float32))  # [N] residuals
    
    # Normalize X for stability
    X_mean = X_reg.mean(dim=0, keepdim=True)
    X_std = X_reg.std(dim=0, keepdim=True) + 1e-8
    X_norm = (X_reg - X_mean) / X_std
    
    # ISTA parameters
    XtX = X_norm.t() @ X_norm
    Xty = X_norm.t() @ y_reg
    L = torch.linalg.matrix_norm(XtX, ord=2).item()  # Lipschitz constant
    step_size = 1.0 / (L + 1e-8)
    
    # Initialize weights
    w = torch.zeros(K0, device=device, dtype=torch.float32)
    
    # Soft thresholding function
    def soft_threshold(x, lambda_val):
        return torch.sign(x) * torch.maximum(torch.abs(x) - lambda_val, torch.zeros_like(x))
    
    # ISTA iterations
    for _ in range(max_iter):
        w_old = w.clone()
        
        # Gradient step
        grad = -Xty + XtX @ w
        w = w - step_size * grad
        
        # Proximal step (soft thresholding)
        w = soft_threshold(w, alpha * step_size)
        
        # Check convergence
        if torch.norm(w - w_old) / (torch.norm(w_old) + 1e-10) < tol:
            break
    
    # Un-normalize weights
    w = w / (X_std.squeeze() + 1e-10)
    
    # Normalize so absolute values sum to 1 (allow negative weights)
    w = w / (torch.abs(w).sum() + 1e-10)
    
    return w


def compute_pairwise_correlation_penalty_weights(per_fold_preds: torch.Tensor,
                                                 gamma: float,
                                                 alpha: float,
                                                 device: torch.device) -> torch.Tensor:
    """
    Compute fold weights using pairwise correlation penalty.
    Penalizes folds that are highly correlated with other folds, promoting diversity.
    
    Algorithm:
    - For each fold k, compute average pairwise correlation with all other folds:
      ρ̄_k = (1/(K-1)) * Σ_{j≠k} corr(p_k, p_j)
    - Compute scores: s_k = (1 - ρ̄_k)^γ
    - Compute weights: w_k = s_k^α / Σ_j s_j^α
    
    Parameters
    ----------
    per_fold_preds : torch.Tensor [K0, N]
        Per-fold predictions for current round
    gamma : float
        Exponent for diversity score (higher = more aggressive diversity penalty)
    alpha : float
        Exponent for weight normalization (higher = more aggressive selection)
    device : torch.device
        Device to perform computation on
        
    Returns
    -------
    torch.Tensor [K0]
        Weights normalized so absolute values sum to 1
    """
    K0, N = per_fold_preds.shape
    
    if K0 == 1:
        # Only one fold, return weight of 1
        return torch.ones(1, dtype=torch.float32, device=device)
    
    # Convert to float for correlation computation
    preds_float = per_fold_preds.to(torch.float32)
    
    # Compute all pairwise correlations efficiently using correlation matrix
    # Standardize predictions: [K0, N]
    pred_mean = preds_float.mean(dim=1, keepdim=True)  # [K0, 1]
    pred_centered = preds_float - pred_mean  # [K0, N]
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-10  # [K0, 1]
    pred_standardized = pred_centered / pred_std  # [K0, N]
    
    # Correlation matrix: [K0, K0]
    # corr[i,j] = corr(pred_i, pred_j)
    corr_matrix = (pred_standardized @ pred_standardized.t()) / N  # [K0, K0]
    
    # For each fold k, compute average correlation with all other folds
    # ρ̄_k = (1/(K-1)) * Σ_{j≠k} corr(p_k, p_j)
    # This is equivalent to: (sum of row k - diagonal element k) / (K-1)
    corr_sum_per_fold = corr_matrix.sum(dim=1)  # [K0]
    corr_self = torch.diagonal(corr_matrix)  # [K0]
    avg_corr = (corr_sum_per_fold - corr_self) / (K0 - 1)  # [K0]
    
    # Compute diversity scores: s_k = (1 - ρ̄_k)^γ
    # Higher score means lower average correlation (more diverse)
    diversity_score = (1.0 - avg_corr) ** gamma  # [K0]
    
    # Clip to ensure non-negative
    diversity_score = torch.maximum(diversity_score, torch.zeros_like(diversity_score))
    
    # Apply temperature softmax: w_i = s_i^(1/T) / Σ_j s_j^(1/T)
    # Stabilize scores first
    eps = 1e-8
    diversity_score_stable = torch.maximum(diversity_score, torch.tensor(eps, device=device))
    
    # Temperature softmax: w = s^(1/T) / sum(s^(1/T))
    # When T=1: standard normalization, T<1: sharper, T>1: flatter
    temperature = alpha  # Use alpha as temperature parameter
    w_unnorm = diversity_score_stable ** (1.0 / temperature)
    
    # Normalize so absolute values sum to 1
    w = w_unnorm / (torch.abs(w_unnorm).sum() + 1e-10)
    
    return w


def compute_orthogonality_weights(per_fold_preds: torch.Tensor,
                                  temperature: float,
                                  device: torch.device) -> torch.Tensor:
    """
    Compute fold weights using orthogonality scores with temperature softmax.
    
    For each fold i, computes how much of its predictions cannot be explained
    by the span of all other folds' predictions, then applies temperature softmax.
    
    Algorithm:
    - For each fold i: compute projection of e_i onto space spanned by E_{-i}
    - Orthogonality score: ||e_i - projection|| / ||e_i|| ∈ [0, 1]
    - Apply temperature softmax: w_i = s_i^(1/T) / Σ_j s_j^(1/T)
    
    Parameters
    ----------
    per_fold_preds : torch.Tensor [K0, N]
        Per-fold predictions for current round
    temperature : float
        Temperature for softmax (T=1: baseline, T<1: sharper, T>1: flatter)
    device : torch.device
        Device to perform computation on
        
    Returns
    -------
    torch.Tensor [K0]
        Weights normalized so absolute values sum to 1
    """
    K0, N = per_fold_preds.shape
    
    if K0 == 1:
        # Only one fold, return weight of 1
        return torch.ones(1, dtype=torch.float32, device=device)
    
    preds_float = per_fold_preds.float()
    
    # ========== Step 1: Compute Gram matrix ==========
    # Gram matrix G = E^T @ E where E is [N, K0]
    # This is the expensive O(K0² × N) operation, fully GPU-accelerated
    gram_matrix = preds_float @ preds_float.t()  # [K0, K0]
    
    # ========== Step 2: Extract submatrices for each fold ==========
    # For each fold i, we need:
    # - gram_minus[i]: Gram matrix with row i and column i removed [K0-1, K0-1]
    # - Ete[i]: Cross products E_{-i}^T @ e_i [K0-1]
    
    # Create index matrix: indices_per_fold[i] contains all indices except i
    idx = torch.arange(K0, device=gram_matrix.device).unsqueeze(0).expand(K0, K0)
    mask = ~torch.eye(K0, dtype=torch.bool, device=gram_matrix.device)
    indices_per_fold = idx[mask].reshape(K0, K0-1)  # [K0, K0-1]
    
    # Extract all K0 submatrices in one operation using advanced indexing
    gram_minus_batched = gram_matrix[indices_per_fold.unsqueeze(1), indices_per_fold.unsqueeze(2)]  # [K0, K0-1, K0-1]
    
    # Extract cross-product vectors for all folds
    fold_indices = torch.arange(K0, device=gram_matrix.device).unsqueeze(1)
    Ete_batched = gram_matrix[indices_per_fold, fold_indices]  # [K0, K0-1]
    
    # ========== Step 3: Solve linear systems ==========
    # For each fold i, solve: (E_{-i}^T E_{-i}) @ coeffs = E_{-i}^T @ e_i
    # Add regularization for numerical stability
    reg = torch.eye(K0-1, device=gram_matrix.device).unsqueeze(0) * 1e-6
    gram_minus_batched = gram_minus_batched + reg
    
    # Batch solve all K0 systems in parallel
    coeffs_batched = torch.linalg.solve(gram_minus_batched, Ete_batched.unsqueeze(-1)).squeeze(-1)  # [K0, K0-1]
    
    # ========== Step 4: Compute orthogonality scores ==========
    # For each fold i, compute: ||e_i - projection||² / ||e_i||²
    
    # Compute squared norm of projections: ||ê_i||² = coeffs^T @ G @ coeffs
    norm_proj_sq = torch.bmm(
        coeffs_batched.unsqueeze(1),  # [K0, 1, K0-1]
        torch.bmm(gram_minus_batched, coeffs_batched.unsqueeze(-1))  # [K0, K0-1, 1]
    ).squeeze(-1).squeeze(-1)  # [K0]
    norm_proj_sq = norm_proj_sq - 1e-6 * (coeffs_batched ** 2).sum(dim=1)  # Remove reg contribution
    
    # Get squared norms of original vectors: ||e_i||²
    norm_ei_sq = torch.diagonal(gram_matrix)  # [K0]
    
    # Orthogonality score = ||e_i - ê_i|| / ||e_i||
    orth_scores = torch.sqrt(torch.clamp(norm_ei_sq - norm_proj_sq, min=0.0)) / (torch.sqrt(norm_ei_sq) + 1e-10)
    
    # ========== Step 5: Apply temperature softmax and normalize ==========
    # w_i = (s_i^(1/T)) / sum_j(s_j^(1/T))
    eps = 1e-8
    orth_scores_stable = torch.maximum(orth_scores, torch.tensor(eps, device=device))
    w_unnorm = orth_scores_stable ** (1.0 / temperature)
    w = w_unnorm / (torch.abs(w_unnorm).sum() + 1e-10)
    
    return w


# Wrapper functions that handle argument extraction from fold_aggregation_method_args

def aggregate_mean(per_fold_preds: torch.Tensor,
                   targets: Optional[torch.Tensor],
                   current_preds: Optional[torch.Tensor],
                   era_ids: Optional[torch.Tensor],
                   device: torch.device,
                   args: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Mean aggregation - equal weights for all folds."""
    K0 = per_fold_preds.shape[0]
    return torch.ones(K0, device=device, dtype=torch.float32) / K0


def aggregate_ridge_regression(per_fold_preds: torch.Tensor,
                                targets: Optional[torch.Tensor],
                                current_preds: Optional[torch.Tensor],
                                era_ids: Optional[torch.Tensor],
                                device: torch.device,
                                args: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Ridge regression aggregation."""
    if args is None:
        args = {}
    alpha = args.get('alpha', 1e-4)
    return compute_ridge_weights(per_fold_preds, targets, current_preds, alpha, device)


def aggregate_lasso_regression(per_fold_preds: torch.Tensor,
                                targets: Optional[torch.Tensor],
                                current_preds: Optional[torch.Tensor],
                                era_ids: Optional[torch.Tensor],
                                device: torch.device,
                                args: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Lasso regression aggregation."""
    if args is None:
        args = {}
    alpha = args.get('alpha', 1e-4)
    max_iter = args.get('max_iter', 1000)
    tol = args.get('tol', 1e-4)
    return compute_lasso_weights(per_fold_preds, targets, current_preds, alpha, device, max_iter, tol)


def aggregate_pairwise_correlation_penalty(per_fold_preds: torch.Tensor,
                                           targets: Optional[torch.Tensor],
                                           current_preds: Optional[torch.Tensor],
                                           era_ids: Optional[torch.Tensor],
                                           device: torch.device,
                                           args: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Pairwise correlation penalty aggregation - promotes diversity."""
    if args is None:
        args = {}
    gamma = args.get('gamma', 1.0)
    alpha = args.get('alpha', 2.0)
    return compute_pairwise_correlation_penalty_weights(per_fold_preds, gamma, alpha, device)


def aggregate_orthogonality_weighting(per_fold_preds: torch.Tensor,
                                      targets: Optional[torch.Tensor],
                                      current_preds: Optional[torch.Tensor],
                                      era_ids: Optional[torch.Tensor],
                                      device: torch.device,
                                      args: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Orthogonality weighting with temperature softmax - weights by unique contribution."""
    if args is None:
        args = {}
    temperature = args.get('temperature', 1.0)
    return compute_orthogonality_weights(per_fold_preds, temperature, device)


# Dictionary mapping method names to aggregation functions
AGGREGATION_METHODS = {
    'mean': aggregate_mean,
    'ridge_regression': aggregate_ridge_regression,
    'lasso_regression': aggregate_lasso_regression,
    'pairwise_correlation_penalty': aggregate_pairwise_correlation_penalty,
    'orthogonality_weighting': aggregate_orthogonality_weighting,
}
