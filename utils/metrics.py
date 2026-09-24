import numpy as np
import torch


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.5) -> float:
    """Mean Absolute Percentage Error (excluding true values < null_val)"""
    mask = y_true >= null_val
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-5))) * 100.0)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error"""
    denom = np.sum(np.abs(y_true))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes a comprehensive dictionary of overall and horizon-wise evaluation metrics.
    For multi-target flow (Cars & Motorcycles), computes both the aggregate Total Volume
    metrics (matching Table 14 of the paper) and category-decomposed errors (Table 17).

    Args:
        y_true (np.ndarray): Ground truth array of shape (N_samples, Horizon, N_nodes, C_out).
        y_pred (np.ndarray): Predicted array of shape (N_samples, Horizon, N_nodes, C_out).

    Returns:
        dict: Evaluation results dictionary.
    """
    mae_macro = compute_mae(y_true, y_pred)
    horizon = y_true.shape[1]

    if y_true.ndim == 4 and y_true.shape[-1] >= 2:
        # Aggregate total traffic volume across vehicle categories: y_total = y_car + y_bike
        y_true_tot = y_true.sum(axis=-1)
        y_pred_tot = y_pred.sum(axis=-1)

        mae_tot = compute_mae(y_true_tot, y_pred_tot)
        rmse_tot = compute_rmse(y_true_tot, y_pred_tot)
        mape_tot = compute_mape(y_true_tot, y_pred_tot, null_val=0.5)
        wape_tot = compute_wape(y_true_tot, y_pred_tot)

        mae_car = compute_mae(y_true[..., 0], y_pred[..., 0])
        mae_bike = compute_mae(y_true[..., 1], y_pred[..., 1])

        metrics = {
            'mae': mae_tot,              # Primary: Total Traffic Volume MAE (matches Table 14)
            'rmse': rmse_tot,            # Primary: Total Traffic Volume RMSE
            'mape': mape_tot,            # Primary: Total Traffic Volume MAPE
            'wape': wape_tot,
            'mae_car': mae_car,          # Category-specific Car MAE (matches Table 17)
            'mae_bike': mae_bike,        # Category-specific Motorcycle MAE (matches Table 17)
            'mae_macro': mae_macro,      # Unweighted channel macro-average
        }

        for h in range(horizon):
            yt_h = y_true_tot[:, h, :]
            yp_h = y_pred_tot[:, h, :]
            metrics[f'mae_h{h+1}'] = compute_mae(yt_h, yp_h)
            metrics[f'rmse_h{h+1}'] = compute_rmse(yt_h, yp_h)
            metrics[f'wape_h{h+1}'] = compute_wape(yt_h, yp_h)
    else:
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
        for h in range(horizon):
            yt_h = y_true[:, h, ...]
            yp_h = y_pred[:, h, ...]
            metrics[f'mae_h{h+1}'] = compute_mae(yt_h, yp_h)
            metrics[f'rmse_h{h+1}'] = compute_rmse(yt_h, yp_h)
            metrics[f'wape_h{h+1}'] = compute_wape(yt_h, yp_h)

    return metrics


def print_metrics_table(metrics: dict, horizon: int = 6, step_minutes: int = 5):
    """
    Formats and prints a formatted metrics summary table.
    """
    print("\n" + "=" * 60)
    print(" 🏆 TA-STGCN PERFORMANCE EVALUATION METRICS 🏆")
    print("=" * 60)
    print(f"  Overall Total Volume MAE  : {metrics['mae']:.4f}")
    print(f"  Overall Total Volume RMSE : {metrics['rmse']:.4f}")
    print(f"  Overall Total Volume MAPE : {metrics['mape']:.2f}%")
    if 'mae_car' in metrics and 'mae_bike' in metrics:
        print(f"  ├── Passenger Car MAE     : {metrics['mae_car']:.4f}")
        print(f"  └── Motorcycle MAE        : {metrics['mae_bike']:.4f}")
        print(f"  (Unweighted Macro-Avg MAE : {metrics.get('mae_macro', 0.0):.4f})")
    print("-" * 60)
    print("  Multi-Horizon Breakdown (Total Traffic Volume):")
    for h in range(horizon):
        step_min = (h + 1) * step_minutes
        mae_h = metrics.get(f'mae_h{h+1}', 0.0)
        rmse_h = metrics.get(f'rmse_h{h+1}', 0.0)
        print(f"   ├─ t+{h+1} ({step_min:>2}m): MAE = {mae_h:.4f} | RMSE = {rmse_h:.4f}")
    print("=" * 60 + "\n")
