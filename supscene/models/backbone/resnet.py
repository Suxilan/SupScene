# adapted from Amar Ali-bey's Bag-of-Queries
# ----------------------------------------------------------------------------
# https://github.com/amaralibey/Bag-of-Queries


import torch.nn as nn
import torch
import torchvision
    
class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        # Create backbone with only the necessary layers
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )

        # Handle trainable/frozen layers
        nb_layers = len(self.net)
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Freeze required layers
            for layer in self.net[:nb_layers - unfreeze_n_blocks]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0

        # Output channels
        if self.crop_last_block:
            last_layer = resnet.layer3
        else:
            last_layer = resnet.layer4

        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = last_layer[-1].conv2.out_channels
        else:
            self.out_channels = last_layer[-1].conv3.out_channels

    def forward(self, x):
        return self.net(x)

    @property
    def output_dim(self):
        """Output feature dimension"""
        return self.out_channels
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, num_trainable_blocks=None):
        """Unfreeze backbone parameters"""
        if num_trainable_blocks is not None:
            self.num_trainable_blocks = num_trainable_blocks
        self._configure_trainable_blocks()