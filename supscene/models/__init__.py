from .backbone import DINOv2, ResNet
from .aggregator import NetVLAD, GeMPool, SCPP
from .heads import DeployHead, DinoProjection, MoCoProjection, DistillHead, ContrastiveHead 

__all__ = [
    "DINOv2","ResNet",
    "NetVLAD", "GeMPool", "SCPP",
    "DeployHead", "DinoProjection", "MoCoProjection", "DistillHead", "ContrastiveHead",
]
