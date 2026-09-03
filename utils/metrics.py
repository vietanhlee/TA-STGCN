import numpy as np
import torch


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 1.0) -> float:
    """Mean Absolute Percentage Error (excluding true values < null_val)"""
    mask = y_true >= null_val
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error"""
    denom = np.sum(np.abs(y_true))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes a comprehensive dictionary of overall and horizon-wise evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth array of shape (N_samples, Horizon, N_nodes, C_out).
        y_pred (np.ndarray): Predicted array of shape (N_samples, Horizon, N_nodes, C_out).

    Returns:
        dict: Evaluation results dictionary.
    """
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    wape = compute_wape(y_true, y_pred)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'wape': wape,
    }

    horizon = y_true.shape[1]
    for h in range(horizon):
        yt_h = y_true[:, h, :, :]
        yp_h = y_pred[:, h, :, :]
        metrics[f'mae_h{h+1}'] = compute_mae(yt_h, yp_h)
        metrics[f'rmse_h{h+1}'] = compute_rmse(yt_h, yp_h)
        metrics[f'wape_h{h+1}'] = compute_wape(yt_h, yp_h)

    return metrics


def print_metrics_table(metrics: dict, horizon: int = 6, step_minutes: int = 5):
    """
    Formats and prints a formatted metrics summary table.
    """
    print("\n" + "=" * 50)
    print(" 🏆 TA-STGCN PERFORMANCE EVALUATION METRICS 🏆")
    print("=" * 50)
    print(f"  Overall MAE  : {metrics['mae']:.4f}")
    print(f"  Overall RMSE : {metrics['rmse']:.4f}")
    print(f"  Overall MAPE : {metrics['mape']:.2f}%")
    print(f"  Overall WAPE : {metrics['wape']:.2f}%")
    print("-" * 50)
    print("  Horizon Breakdown:")
    for h in range(horizon):
        step_min = (h + 1) * step_minutes
        mae_h = metrics.get(f'mae_h{h+1}', 0.0)
        rmse_h = metrics.get(f'rmse_h{h+1}', 0.0)
        wape_h = metrics.get(f'wape_h{h+1}', 0.0)
        print(f"   ├─ t+{h+1} ({step_min:>2}m): MAE={mae_h:.4f} | RMSE={rmse_h:.4f} | WAPE={wape_h:.2f}%")
    print("=" * 50 + "\n")
