import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


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
            return_attn_maps=False,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_bias='none',
            lora_targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        
        self.model = torch.hub.load('third_party/dinov2', model_name, source='local')
            
        self.num_channels = self.model.embed_dim
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_cls_token = return_cls_token
        self.return_attn_maps = return_attn_maps
        self.patch_size = 14

        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_bias = str(lora_bias)
        self.lora_targets = tuple(lora_targets)
        
        # Configure trainable parameters
        self._configure_trainable_blocks()
        self._inject_lora()

    def _collect_target_module_names(self):
        block_indices = list(range(self.num_frozen_blocks, len(self.model.blocks)))
        prefixes = [f"blocks.{i}." for i in block_indices]

        full_names = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.startswith(p) for p in prefixes):
                continue
            if any(name.endswith(t) for t in self.lora_targets):
                full_names.append(name)
        return sorted(set(full_names))

    def _inject_lora(self):
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:
            raise ImportError(
                "PEFT is required for DINOv2 LoRA backbone. Please install `peft`."
            ) from exc

        target_modules = self._collect_target_module_names()
        if len(target_modules) == 0:
            raise ValueError(
                "No target modules found for LoRA injection. "
                f"num_trainable_blocks={self.num_trainable_blocks}, targets={self.lora_targets}"
            )

        peft_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            target_modules=target_modules,
        )
        _ = get_peft_model(self.model, peft_cfg)

        for name, param in self.model.named_parameters():
            param.requires_grad = ('lora_' in name)

        if self.norm_layer:
            for param in self.model.norm.parameters():
                param.requires_grad = True
        
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
        h_new = (H // self.patch_size) * self.patch_size
        w_new = (W // self.patch_size) * self.patch_size
        x = CenterCrop((h_new, w_new))(x)

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        if self.num_frozen_blocks > 0:
            with torch.no_grad():
                for blk in self.model.blocks[:self.num_frozen_blocks]:
                    x = blk(x)
            x = x.detach()

        # Last blocks are trainable (LoRA + optional norm)
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
        f = f.reshape((B, h_new // self.patch_size, w_new // self.patch_size, self.num_channels)).permute(0, 3, 1, 2)
        
        if self.return_attn_maps and attn is not None:
            # Process attention maps
            cls_to_patch_attn = attn[:, :, 1:, 0]  
            patch_h, patch_w = h_new // self.patch_size, w_new // self.patch_size
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
