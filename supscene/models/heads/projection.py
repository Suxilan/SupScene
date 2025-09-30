from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead

def trunc_normal_(tensor, mean=0.0, std=1.0, a: float = -2.0, b: float = 2.0):
    """
    Truncated normal initialization.
    """
    def norm_cdf(x):
        return (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * torch.sqrt(torch.tensor(2.0)))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

# Code adapted from SALAD, Apache 2.0 license
# https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
class DinoProjection(BaseHead):
    """
    MLP -> L2 Norm -> linear (weight normalized)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = True,
        mlp_bias: bool = True,
    ):
        super().__init__(in_dim, out_dim)
        
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(
            nlayers, in_dim, bottleneck_dim, 
            hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias
        )
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        with torch.no_grad():
            weight_norm = torch.norm(self.last_layer.weight.data, dim=1, keepdim=True)
            self.last_layer.weight.data.div_(weight_norm)
        
        self.apply(self._init_weights)
    
    def _build_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
        """build MLP layers"""
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim, bias=bias)
        
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in] 
            node_mask: [B, N] 
        Returns:
            z: [B, N, D_out] 
        """
        B, N, D = x.shape
        
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)
        
        # Reshape for BatchNorm
        x_flat = x.reshape(B * N, D)
        
        # MLP + L2 Norm
        x_flat = self.mlp(x_flat)
        eps = 1e-6 if x_flat.dtype == torch.float16 else 1e-12
        x_flat = F.normalize(x_flat, dim=-1, p=2, eps=eps)
        
        # last layer (linear + weight norm)
        z_flat = self.last_layer(x_flat)
        z = z_flat.reshape(B, N, -1)
        
        return z

# Code adapted from MocoV3
# https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
class MoCoProjection(BaseHead):
    """
    MoCo/SimCLR style projection head: multi-layer MLP + optional last layer BN (no affine)
    """

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int = 2048,
        nlayers: int = 3, 
        last_bn: bool = False
    ):
        super().__init__(in_dim, out_dim)
        self.head = self._build_mlp(nlayers, in_dim, hidden_dim, out_dim, last_bn=last_bn)
        
    def _build_mlp(self, nlayers, in_dim, hidden_dim, out_dim, last_bn=True):
        """构建MLP"""
        mlp = []
        for l in range(nlayers):
            dim1 = in_dim if l == 0 else hidden_dim
            dim2 = out_dim if l == nlayers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < nlayers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # SimCLR uses a BatchNorm without affine transformation
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
        
    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in]   
            node_mask: [B, N] 
        Returns:
            z: [B, N, D_out] 
        """
        B, N, D = x.shape
        
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)
        
        # Reshape for BatchNorm
        x_flat = x.reshape(B * N, D)
        z_flat = self.head(x_flat)
        z = z_flat.reshape(B, N, -1)
        
        return z
