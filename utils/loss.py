import torch
import torch.nn as nn


class HuberSmoothLoss(nn.Module):
    """
    Robust Huber Loss (Smooth L1) with optional temporal-smoothness regularizer.
    
    Combines L2 smoothness for small errors with L1 robustness for large outliers.
    Optionally penalizes consecutive prediction step variations across the horizon.
    """
    def __init__(self, delta: float = 1.0, smooth_weight: float = 0.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)
        self.smooth_weight = smooth_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, x_last: torch.Tensor = None) -> torch.Tensor:
        """
        Forward loss pass.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, Horizon, N, C).
            target (torch.Tensor): Ground truth tensor of shape (B, Horizon, N, C).
            x_last (torch.Tensor, optional): Last historical timestep input.

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        base_loss = self.loss_fn(pred, target)
        if self.smooth_weight > 0 and pred.shape[1] > 1:
            diff = pred[:, 1:] - pred[:, :-1]
            smooth_loss = torch.mean(diff ** 2)
            return base_loss + self.smooth_weight * smooth_loss
        return base_loss
