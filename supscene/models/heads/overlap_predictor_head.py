"""
OverlapPredictorHead: 轻量级重叠度预测头
输入: z [B, N, D]
输出: y_hat [B, N, N] (0~1), 可选激活
实现: 双线性打分 + 仿射，再经 Sigmoid
y_ij = sigmoid( z_i^T W z_j * alpha + beta )
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead
class OverlapPredictorHead(BaseHead):
    def __init__(
        self,
        in_dim: int,
        use_bias: bool = True,
        apply_sigmoid: bool = True,
    ):
        super().__init__(in_dim, in_dim)
        self.apply_sigmoid = apply_sigmoid

        # 双线性核 W
        self.bilinear = nn.Linear(in_dim, in_dim, bias=False)
        # 全局缩放与偏置
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0)) if use_bias else None

        # 初始化为接近恒等，便于稳定起步
        with torch.no_grad():
            self.bilinear.weight.copy_(torch.eye(in_dim))

    def forward(self, z: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: [B, N, D]
        node_mask: [B, N]，无效节点将被置零，输出也会mask
        return: y_hat [B, N, N]
        """
        B, N, D = z.shape
        if node_mask is not None:
            z = z * node_mask.unsqueeze(-1).to(z.dtype)

        # zW
        zW = self.bilinear(z.reshape(B * N, D)).reshape(B, N, D)
        # 矩阵乘: (zW) @ z^T -> [B, N, N]
        S = torch.matmul(zW, z.transpose(1, 2))

        # 对称性：理论上 S 已对称（若 W 对称且 zW=zW），此处不强制对称以保留表达力
        # 缩放与偏置
        S = self.alpha * S
        if self.beta is not None:
            S = S + self.beta
        if self.apply_sigmoid:
            S = torch.sigmoid(S)

        if node_mask is not None:
            m = node_mask.bool()
            S = S * (m.unsqueeze(1) & m.unsqueeze(2)).to(S.dtype)

        return S
