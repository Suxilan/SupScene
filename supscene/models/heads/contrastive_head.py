
import torch
import torch.nn as nn
from .projection import DinoProjection, MoCoProjection

from typing import Optional

class ContrastiveHead(nn.Module):
    """
    对比头：forward(z) -> q_contrast
    """
    def __init__(self, in_dim:int, style:str="dino", **kwargs):
        super().__init__()
        if style.lower() == "dino":
            self.impl = DinoProjection(in_dim, **kwargs)
        elif style.lower() == "moco":
            self.impl = MoCoProjection(in_dim, **kwargs)
        else:
            raise ValueError(f"Unknown contrastive style: {style}")

    def forward(self, z: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        return self.impl(z, node_mask)
