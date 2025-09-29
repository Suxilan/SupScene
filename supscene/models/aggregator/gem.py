import torch
import torch.nn.functional as F
import torch.nn as nn


class GeMPool(nn.Module):
    """
    Implementation of Generalized Mean (GeM) pooling
    
    GeM pooling computes the generalized mean of feature maps:
    f = (1/|X| * sum(x^p))^(1/p)
    
    Args:
        p (float): Power parameter for GeM pooling. p=1 gives average pooling, p=inf gives max pooling
        eps (float): Small constant to avoid numerical issues
        trainable (bool): Whether p parameter is trainable
        flatten (bool): Whether to flatten output
    """
    def __init__(
        self, 
        in_dim, 
        p=3.0, 
        eps=1e-6, 
        trainable=True, 
        flatten=True
    ):
        super().__init__()
        
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
            
        self.eps = eps
        self.flatten = flatten
        self.out_dim = in_dim

    def forward(self, x, return_maps=False):
        """
        Forward pass
        
        Args:
            x: Input feature maps [B, C, H, W]
            return_maps: Whether to return attention maps
            
        Returns:
            pooled: GeM pooled features [B, C] if flatten=True, else [B, C, 1, 1]
            maps: (optional) GeM attention maps [B, C, H, W]
        """
        # Calculate GeM weights (x^p normalized)
        x_clamped = x.clamp(min=self.eps)
        x_powered = x_clamped.pow(self.p)
        
        # GeM pooling: (1/N * sum(x^p))^(1/p)
        pooled = F.avg_pool2d(
            x_powered, 
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)
        
        if self.flatten:
            pooled = pooled.flatten(1)
            
        assert pooled.size(1) == self.out_dim, f"Output dimension mismatch: {pooled.size(1)} != {self.out_dim}"
        
        if return_maps:
            # Compute attention maps (normalized x^p)
            maps = x_powered / (x_powered.sum(dim=(2, 3), keepdim=True) + self.eps)
            return pooled, maps, None, None, None
        
        return pooled
    
    @property
    def output_dim(self):
        """Output feature dimension (same as input channel dimension)"""
        return self.out_dim
    
    def freeze(self):
        """Freeze parameters"""
        if isinstance(self.p, nn.Parameter):
            self.p.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze parameters"""
        if isinstance(self.p, nn.Parameter):
            self.p.requires_grad = True


class AdaptiveGeMPool(nn.Module):
    """
    Adaptive GeM pooling with learnable output size
    
    Args:
        output_size (tuple): Target output size (H, W)
        p (float): Power parameter for GeM pooling
        eps (float): Small constant to avoid numerical issues
        trainable (bool): Whether p parameter is trainable
        flatten (bool): Whether to flatten output
    """
    def __init__(self, in_dim, output_size=(1, 1), p=3.0, eps=1e-6, trainable=True, 
                 flatten=True):
        super().__init__()
        
        self.output_size = output_size
        
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
            
        self.eps = eps
        self.flatten = flatten
        self.out_dim = in_dim

    def forward(self, x):
        """
        Forward pass with adaptive pooling
        
        Args:
            x: Input feature maps [B, C, H, W]
            
        Returns:
            pooled: GeM pooled features
        """
        # Apply GeM power
        x_powered = x.clamp(min=self.eps).pow(self.p)
        
        # Adaptive average pooling
        pooled = F.adaptive_avg_pool2d(x_powered, self.output_size)
        
        # Apply inverse power
        pooled = pooled.pow(1.0 / self.p)
        
        if self.flatten:
            pooled = pooled.flatten(1)
            
        assert pooled.size(1) == self.out_dim, f"Output dimension mismatch: {pooled.size(1)} != {self.out_dim}"
            
        return pooled
    
    def freeze(self):
        """Freeze parameters"""
        if isinstance(self.p, nn.Parameter):
            self.p.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze parameters"""
        if isinstance(self.p, nn.Parameter):
            self.p.requires_grad = True


# if __name__ == "__main__":
#     import torch
#     from torch.autograd import gradcheck

#     # Use double precision to avoid gradient check warnings
#     x = torch.randn(4, 3, 5, 5, dtype=torch.double, requires_grad=True)
#     gem = GeMPool(in_dim=3)
#     # Convert model parameters to double precision
#     gem = gem.double()
    
#     y = gem(x)
#     y.backward(torch.ones_like(y))

#     assert gradcheck(gem, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)
    