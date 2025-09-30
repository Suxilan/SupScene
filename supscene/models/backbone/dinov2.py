import torch
import torch.nn as nn


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class DINOv2(nn.Module):
    """
    DINOv2 model with configurable trainable blocks
    
    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_cls_token (bool): If True, the forward pass returns both the feature map and the token.
        return_attn_maps (bool): If True, the forward pass returns the attention maps.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=True,
            return_cls_token=False,
            return_attn_maps=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        
        try:
            self.model = torch.hub.load('third_party/dinov2', model_name, source='local')
        except Exception as e:
            print(f"Warning: Could not load {model_name} from torch.hub: {e}")
            
        self.num_channels = self.model.embed_dim
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_cls_token = return_cls_token
        self.return_attn_maps = return_attn_maps
        
        # Configure trainable parameters
        self._configure_trainable_blocks()
        
    def _configure_trainable_blocks(self):
        """Configure which blocks are trainable"""
        # freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.num_frozen_blocks = len(self.model.blocks) - self.num_trainable_blocks
        if hasattr(self.model, 'blocks'):
            # First freeze all blocks
            for blk in self.model.blocks[:self.num_frozen_blocks]:
                for param in blk.parameters():
                    param.requires_grad = False
                    
            # Then make only the last num_trainable_blocks trainable
            if self.num_trainable_blocks > 0:
                for blk in self.model.blocks[self.num_frozen_blocks:]:
                    for param in blk.parameters():
                        param.requires_grad = True
                if self.norm_layer:
                    for param in self.model.norm.parameters():
                        param.requires_grad = True

    def forward(self, x):
        """
        Forward pass for DINOv2
        
        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.
        
        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_cls_token is True.
        """
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        if self.num_frozen_blocks > 0:
            with torch.no_grad():
                for blk in self.model.blocks[:self.num_frozen_blocks]:
                    x = blk(x)
            x = x.detach()

        # Last blocks are trained
        attn = None
        for i, blk in enumerate(self.model.blocks[self.num_frozen_blocks:]):
            if self.return_attn_maps and i == len(self.model.blocks[self.num_frozen_blocks:]) - 1:
                attn = blk(x, return_attention=True)
                x = blk(x)
            else:
                x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]  # CLS token
        f = x[:, 1:]  # Patch tokens

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
        
        if self.return_attn_maps and attn is not None:
            # Process attention maps
            cls_to_patch_attn = attn[:, :, 1:, 0]  
            patch_h, patch_w = H // 14, W // 14
            attn_maps = cls_to_patch_attn.reshape(B, attn.shape[1], patch_h, patch_w)
            return f, attn_maps

        if self.return_cls_token:
            return f, t
        return f
    
    @property
    def output_dim(self):
        """Output feature dimension"""
        return self.num_channels
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, num_trainable_blocks=None):
        """Unfreeze backbone parameters"""
        if num_trainable_blocks is not None:
            self.num_trainable_blocks = num_trainable_blocks
        self._configure_trainable_blocks()


# if __name__ == "__main__":
#     # Test DINOv2 output dimensions
#     model = DINOv2(model_name='dinov2_vitb14', return_cls_token=True).to("cuda")
#     x = torch.randn(2, 3, 322, 322).to("cuda")
    
#     # Test with CLS token
#     f, t = model(x)
#     print(f"Feature map shape: {f.shape}")  # Should be [2, 768, 16, 16]
#     print(f"CLS token shape: {t.shape}")    # Should be [2, 768]
    
#     # Test without CLS token
#     model.return_cls_token = False
#     f = model(x)
#     print(f"Feature map only shape: {f.shape}")  # Should be [2, 768, 16, 16]
