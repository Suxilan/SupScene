from .cluster_head import SimpleClusterHead
from .deploy_head import DeployHead

# New projection heads
from .projection import DinoProjection, MoCoProjection

# Distill/Contrastive heads (thin wrappers over projection)
from .distill_head import DistillHead
from .contrastive_head import ContrastiveHead

# Overlap predictor head
from .overlap_predictor_head import OverlapPredictorHead

__all__ = [
    # cluster
    "SimpleClusterHead",
    # projection
    "DinoProjection",
    "MoCoProjection",
    # distill/contrastive
    "DistillHead",
    "ContrastiveHead",
    # overlap predictor
    "OverlapPredictorHead",
    # deploy
    "DeployHead",
]
