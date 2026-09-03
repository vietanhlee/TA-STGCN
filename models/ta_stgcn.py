import torch
import torch.nn as nn

from .stgcn_block import STGCNBlock
from .temporal_attention import TemporalAttention


class TASTGCNModel(nn.Module):
    """
    TA-STGCN: Temporal Attention-Guided Spatio-Temporal Graph Convolutional Network.
    
    Combines spatial Chebyshev graph convolutions with Temporal Self-Attention
    to achieve noise-robust multi-horizon traffic forecasting across urban camera networks.
    
    Args:
        num_nodes (int): Number of spatial nodes (cameras/intersections).
        in_feat (int): Input feature dimension per node (e.g. 5: Car, Bike, Hour sin/cos, Hour norm).
        block_hidden (int): Channel capacity for hidden STGCN blocks (e.g. 80).
        num_blocks (int): Number of STGCN blocks (default: 2).
        T_in (int): Number of historical observation steps (default: 24 for 120m).
        cheb_K (int): Chebyshev polynomial filter order (default: 3).
        horizon (int): Number of forecasting steps into the future (default: 6 for 30m).
        output_feat (int): Output feature dimension per node (default: 2 for Car, Bike counts).
        L_tilde (torch.Tensor, optional): Precomputed Scaled Chebyshev Laplacian of shape (N, N).
        dropout (float): Dropout probability for STGCN blocks (default: 0.25).
        use_temporal_attention (bool): Whether to activate Temporal Self-Attention (default: True).
        attn_num_heads (int): Number of heads in Multi-Head Attention (default: 4).
        attn_dropout (float): Dropout probability for Attention layer (default: 0.1).
        attn_position (str): Placement position for Temporal Attention: 'middle', 'end', or 'before' (default: 'middle').
    """
    def __init__(
        self,
        num_nodes: int,
        in_feat: int = 5,
        block_hidden: int = 80,
        num_blocks: int = 2,
        T_in: int = 24,
        cheb_K: int = 3,
        horizon: int = 6,
        output_feat: int = 2,
        L_tilde: torch.Tensor = None,
        dropout: float = 0.25,
        use_temporal_attention: bool = True,
        attn_num_heads: int = 4,
        attn_dropout: float = 0.1,
        attn_position: str = 'middle'
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.output_feat = output_feat
        self.use_temporal_attention = use_temporal_attention
        self.attn_position = attn_position.lower()

        if use_temporal_attention and self.attn_position == 'before':
            self.in_proj = nn.Conv2d(in_feat, block_hidden, kernel_size=1)
            c_in = block_hidden
        else:
            self.in_proj = None
            c_in = in_feat

        blocks = []
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(c_in, block_hidden, num_nodes, cheb_K, dropout))
            c_in = block_hidden
        self.blocks = nn.ModuleList(blocks)

        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(block_hidden, attn_num_heads, attn_dropout)
        else:
            self.temporal_attn = None

        # Time-dimension reduction convolution: T_in -> 1
        self.final_conv = nn.Conv1d(block_hidden, horizon * output_feat, kernel_size=T_in)

        if L_tilde is None:
            self.register_buffer('L_tilde', torch.eye(num_nodes))
        else:
            if not isinstance(L_tilde, torch.Tensor):
                L_tilde = torch.tensor(L_tilde, dtype=torch.float32)
            self.register_buffer('L_tilde', L_tilde.float())

    def apply_attn(self, h: torch.Tensor) -> torch.Tensor:
        """
        Applies Temporal Self-Attention across time dimension.
        h shape: (B, C, N, T) -> reshaped to (B * N, T, C) -> attention -> (B, C, N, T)
        """
        B, C, N, T = h.shape
        h_seq = h.permute(0, 2, 3, 1).reshape(B * N, T, C)
        h_seq = self.temporal_attn(h_seq)
        return h_seq.reshape(B, N, T, C).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T_in, N, in_feat).

        Returns:
            torch.Tensor: Multi-step prediction of shape (B, horizon, N, output_feat).
        """
        h = x.permute(0, 3, 2, 1)  # (B, in_feat, N, T_in)

        if self.use_temporal_attention:
            if self.attn_position == 'before':
                h = self.in_proj(h)
                h = self.apply_attn(h)

            mid_idx = max(1, len(self.blocks) // 2)
            for i, block in enumerate(self.blocks):
                h = block(h, self.L_tilde)
                if self.attn_position == 'middle' and i == mid_idx - 1:
                    h = self.apply_attn(h)

            if self.attn_position == 'end':
                h = self.apply_attn(h)
        else:
            for block in self.blocks:
                h = block(h, self.L_tilde)

        B, C, N, T = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * N, C, T)

        out = self.final_conv(h)  # (B * N, horizon * output_feat, 1)
        out = out.squeeze(-1)
        out = out.view(B, N, self.horizon, self.output_feat)
        y_pred = out.permute(0, 2, 1, 3)  # (B, horizon, N, output_feat)

        return y_pred


# Alias for backward compatibility
STGCN_Model = TASTGCNModel
