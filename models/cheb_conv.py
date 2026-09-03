import torch
import torch.nn as nn


class ChebConvLayer(nn.Module):
    """
    Chebyshev Spectral Graph Convolution Layer (Defferrard et al., NIPS 2016; Yu et al., IJCAI 2018).
    
    Computes spatial graph convolutions using Chebyshev polynomial approximation up to order K.
    
    Mathematical Formulation:
        Y = \\sum_{k=0}^{K-1} T_k(\\tilde{L}) X \\Theta_k
        where:
        - T_0(\\tilde{L}) X = X
        - T_1(\\tilde{L}) X = \\tilde{L} X
        - T_k(\\tilde{L}) X = 2 \\tilde{L} T_{k-1}(\\tilde{L}) X - T_{k-2}(\\tilde{L}) X  (k >= 2)
    """
    def __init__(self, in_feats: int, out_feats: int, K: int = 3):
        super().__init__()
        self.K = K
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.linears = nn.ModuleList([nn.Linear(in_feats, out_feats) for _ in range(K)])

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Feature tensor of shape (B*T, N, in_feats).
            L_tilde (torch.Tensor): Scaled Chebyshev Laplacian of shape (N, N).

        Returns:
            torch.Tensor: Convolved spatial output of shape (B*T, N, out_feats).
        """
        T_prev = x
        out = self.linears[0](T_prev)

        if self.K > 1:
            T_curr = torch.einsum('ij,bjf->bif', L_tilde, x)
            out = out + self.linears[1](T_curr)

            for k in range(2, self.K):
                T_next = 2.0 * torch.einsum('ij,bjf->bif', L_tilde, T_curr) - T_prev
                out = out + self.linears[k](T_next)
                T_prev, T_curr = T_curr, T_next

        return out
