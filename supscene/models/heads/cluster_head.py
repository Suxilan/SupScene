import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead

class SimpleClusterHead(BaseHead):
    """
    低秩分配头：MLP -> logits -> (温度缩放) -> softmax 得到 P，适配低秩重叠回归 S≈PP^T。
    输入:  z: [B, N, D]  (ProjectionHead 输出)
    输出:  P: [B, N, K], logits: [B, N, K]
    特性:
      - 支持可学习温度 or 固定温度
      - 支持 node_mask（无效节点 logits=-inf，P=0）
      - Xavier 初始化
    """
    def __init__(
        self,
        in_dim: int,
        K: int,
        hidden: int = 512,
        dropout: float = 0.1,
        tau: float = 0.07,
        learnable_tau: bool = False,
        num_layers: int = 2,
        bias: bool = True,
    ):
        super().__init__(in_dim, K)
        self.learnable_tau = learnable_tau

        layers = []
        d_prev = in_dim
        for li in range(max(0, num_layers - 1)):
            layers += [
                nn.Linear(d_prev, hidden, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            d_prev = hidden
        layers += [nn.Linear(d_prev, K, bias=bias)]
        self.mlp = nn.Sequential(*layers)

        if learnable_tau:
            # 以 logit 形式学习缩放：scale = exp(s)；初值与 tau 对齐
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / max(tau, 1e-4))))
        else:
            self.register_buffer("inv_tau", torch.tensor(1.0 / max(tau, 1e-4)))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.learnable_tau:
            scale = torch.exp(self.logit_scale).clamp(1e-2, 1e4)
            return logits * scale
        else:
            return logits * self.inv_tau

    def forward(
        self,
        z: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        z: [B, N, D]
        node_mask: [B, N] (True=有效)
        返回:
          P: [B, N, K] (行归一)
          logits: [B, N, K] 或 None
        """
        assert z.dim() == 3, f"Expected [B,N,D], got {tuple(z.shape)}"
        B, N, D = z.shape

        logits = self.mlp(z)                   # [B, N, K]
        logits = self._scale_logits(logits)    # 温度缩放

        if node_mask is not None:
            # 将无效节点置为 -inf，softmax 后即 0 分布
            mask = (~node_mask).unsqueeze(-1)              # [B,N,1]
            logits = logits.masked_fill(mask, float("-inf"))

        P = F.softmax(logits, dim=-1)          # 行 softmax
        # 数值保障（极端 -inf 行，softmax 会给 NaN）
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        if return_logits:
            return (P, logits)
        return P
