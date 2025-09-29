import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    @property
    def output_dim(self):
        return self.out_dim
    
    def _maybe_norm(self, x: torch.Tensor, do_norm: bool = True) -> torch.Tensor:
        if do_norm:
            # use small eps for stability across dtypes
            eps = 1e-6 if x.dtype in (torch.float16, torch.bfloat16) else 1e-12
            return F.normalize(x, dim=-1, p=2, eps=eps)
        return x