import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Multi-Head Temporal Self-Attention Module.
    
    Observes the complete temporal window (T_in historical timesteps) across spatial nodes,
    computing adaptive attention weights over historical timesteps.
    
    Structure:
        - Multi-Head Attention (nn.MultiheadAttention)
        - Residual Connection 1 + LayerNorm
        - Feed-Forward Network (Linear -> GELU -> Dropout -> Linear -> Dropout)
        - Residual Connection 2 + LayerNorm
    """
    def __init__(self, in_channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(in_channels)
        self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (B * N, T, C).

        Returns:
            torch.Tensor: Attention-enhanced features of shape (B * N, T, C).
        """
        try:
            with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
                attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        except Exception:
            attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)

        self.last_attn_weights = attn_weights
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)
