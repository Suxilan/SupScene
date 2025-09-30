import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseHead
class DeployHead(BaseHead):
    """
    Deploy Head for feature projection and normalization.
    
    Args:
        in_dim: the input feature dimension
        out_dim: the deployment vector dimension (default is the same as input)
        use_projection: whether to use dimensionality reduction projection (if False, directly L2 normalize)
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
            nn.init.xavier_uniform_(self.proj.weight)
            if bias:
                nn.init.zeros_(self.proj.bias)
        else:
            self.out_dim = in_dim
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*N, D_in]
        Returns:
            z: [B*N, D_out]
        """
        
        z = self.proj(x)
        
        # L2 Normalization for deployment
        return self._maybe_norm(z)
