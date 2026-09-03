import torch
import torch.nn as nn


class TemporalConvLayer(nn.Module):
    """
    Temporal Gated Convolution Layer (TCN block in STGCN).
    
    Applies 1D Causal Convolution along the temporal dimension combined with a Gated Linear Unit (GLU).
    
    Mathematical Formulation:
        [P, Q] = Conv1D(X)
        Output = P \\odot \\sigma(Q)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (B, in_channels, N, T).

        Returns:
            torch.Tensor: Gated temporal features of shape (B, out_channels, N, T).
        """
        x = self.conv(x)
        p, q = torch.chunk(x, 2, dim=1)
        return p * torch.sigmoid(q)
