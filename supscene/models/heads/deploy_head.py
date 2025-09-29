"""
轻量部署头 - 保持特征表达能力，最小化计算开销
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseHead
class DeployHead(BaseHead):
    """
    轻量部署头：仅做维度映射+L2归一化，保护特征表达能力
    适合实际检索部署，计算开销最小
    
    Args:
        in_dim: 输入特征维度 (NetVLAD输出)
        out_dim: 部署向量维度 (默认与输入相同)
        use_projection: 是否降维投影 (False时直接L2归一化)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        use_projection: bool = False,
        bias: bool = False
    ):
        super().__init__(in_dim, out_dim)
        self.use_projection = use_projection
        
        if use_projection or out_dim is not None:
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
            # Xavier初始化保持方差稳定
            nn.init.xavier_uniform_(self.proj.weight)
            if bias:
                nn.init.zeros_(self.proj.bias)
        else:
            self.out_dim = in_dim
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*N, D_in] 聚合后特征
        Returns:
            z: [B*N, D_out] L2归一化的部署向量
        """
        
        # 可选投影
        z = self.proj(x)
        
        # L2归一化 (部署必需)
        return self._maybe_norm(z)
