import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_conv import TemporalConvLayer
from .cheb_conv import ChebConvLayer


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block (STGCN Block).
    
    Pipeline:
        1. Temporal Gated Conv 1 (TCN)
        2. Spatial Graph Conv (ChebConv, K-order) + ReLU
        3. Temporal Gated Conv 2 (TCN)
        4. Residual Addition + LayerNorm + Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        cheb_K: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.tconv1 = TemporalConvLayer(in_channels, out_channels, kernel_size=3)
        self.sconv = ChebConvLayer(out_channels, out_channels, cheb_K)
        self.tconv2 = TemporalConvLayer(out_channels, out_channels, kernel_size=3)
        self.ln = nn.LayerNorm([num_nodes, out_channels])
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (B, in_channels, N, T).
            L_tilde (torch.Tensor): Scaled Chebyshev Laplacian of shape (N, N).

        Returns:
            torch.Tensor: Block output of shape (B, out_channels, N, T).
        """
        res = self.residual(x)

        h = self.tconv1(x)

        B, C, N, T = h.shape
        h_s = h.permute(0, 3, 2, 1).reshape(B * T, N, C)
        h_s = self.sconv(h_s, L_tilde)
        h = h_s.view(B, T, N, C).permute(0, 3, 2, 1)
        h = F.relu(h)

        h = self.tconv2(h)

        h = h + res
        h = h.permute(0, 3, 2, 1)
        h = self.ln(h)
        h = h.permute(0, 3, 2, 1)
        h = self.dropout(h)

        return h
