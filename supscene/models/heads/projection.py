"""
投影头（Projection Heads）
- DinoProjection: 深层MLP + 权重归一化最后层（DINO风格）
- MoCoProjection: 标准MLP + 可选最后层BN（MoCo/SimCLR风格）
两者输入/输出统一：
  x: [B, N, D_in]  ->  z: [B, N, D_out]
可选 node_mask: [B, N]
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead

def trunc_normal_(tensor, mean=0.0, std=1.0, a: float = -2.0, b: float = 2.0):
    """截断正态分布初始化"""
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


class DinoProjection(BaseHead):
    """
    DINO 风格投影头：深层 MLP -> L2 Norm -> 线性（权重行归一）
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
        
        # MLP主体
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(
            nlayers, in_dim, bottleneck_dim, 
            hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias
        )
        
        # 权重归一化的最后层 (DINO关键设计)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        with torch.no_grad():
            weight_norm = torch.norm(self.last_layer.weight.data, dim=1, keepdim=True)
            self.last_layer.weight.data.div_(weight_norm)
        
        self.apply(self._init_weights)
    
    def _build_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
        """构建MLP层"""
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
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in] 聚合后特征
            node_mask: [B, N] 有效节点掩码
        Returns:
            z: [B, N, D_out] 训练用投影特征
        """
        B, N, D = x.shape
        
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)
        
        # Reshape for BatchNorm
        x_flat = x.reshape(B * N, D)
        
        # MLP + 中间L2归一化
        x_flat = self.mlp(x_flat)
        eps = 1e-6 if x_flat.dtype == torch.float16 else 1e-12
        x_flat = F.normalize(x_flat, dim=-1, p=2, eps=eps)
        
        # 权重归一化的最后层
        z_flat = self.last_layer(x_flat)
        z = z_flat.reshape(B, N, -1)
        
        return z


class MoCoProjection(BaseHead):
    """
    MoCo/SimCLR 风格投影头：多层 MLP + 可选最后层 BN（无仿射）
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
                # SimCLR设计：最后层BN无仿射参数
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
        
    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in] 聚合后特征  
            node_mask: [B, N] 有效节点掩码
        Returns:
            z: [B, N, D_out] 训练用投影特征
        """
        B, N, D = x.shape
        
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)
        
        # Reshape for BatchNorm
        x_flat = x.reshape(B * N, D)
        z_flat = self.head(x_flat)
        z = z_flat.reshape(B, N, -1)
        
        return z
