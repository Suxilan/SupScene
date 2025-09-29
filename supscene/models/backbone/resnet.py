import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "resnext50": models.resnext50_32x4d,
    }

    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        num_trainable_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = model_name
        self.pretrained = pretrained
        self.num_trainable_blocks = num_trainable_blocks
        self.crop_last_block = crop_last_block

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {model_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[model_name](weights=weights)

        # Create backbone with only the necessary layers
        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )
        
        
        # Output channels
        if self.backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = resnet.layer3[-1].conv2.out_channels
        else:
            self.out_channels = resnet.layer3[-1].conv3.out_channels if crop_last_block else resnet.layer4[-1].conv3.out_channels

        # Configure trainable blocks
        self._configure_trainable_blocks()
    
    def _configure_trainable_blocks(self):
        """Configure which blocks are trainable"""
        nb_layers = len(self.model)
        
        # Validate num_trainable_blocks
        if not (isinstance(self.num_trainable_blocks, int) and 0 <= self.num_trainable_blocks <= nb_layers):
            raise ValueError(f"num_trainable_blocks must be an integer between 0 and {nb_layers} (inclusive)")
        
        if self.pretrained:
            # Freeze all parameters first
            for param in self.parameters():
                param.requires_grad = False
                
            # Unfreeze the last num_trainable_blocks layers
            if self.num_trainable_blocks > 0:
                for layer in self.model[-self.num_trainable_blocks:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            # When not pretrained, all parameters are trainable by default
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
    @property
    def output_dim(self):
        """Output feature dimension"""
        return self.out_channels
        
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, num_trainable_blocks: Optional[int] = None):
        """Unfreeze backbone parameters"""
        if num_trainable_blocks is not None:
            self.num_trainable_blocks = num_trainable_blocks
        self._configure_trainable_blocks()
    

