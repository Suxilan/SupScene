from .deploy_head import DeployHead

# New projection heads
from .projection import DinoProjection, MoCoProjection

# Distill/Contrastive heads (thin wrappers over projection)
from .distill_head import DistillHead
from .contrastive_head import ContrastiveHead


__all__ = [
    "DinoProjection",
    "MoCoProjection",
    "DistillHead",
    "ContrastiveHead",
    "DeployHead",
]
