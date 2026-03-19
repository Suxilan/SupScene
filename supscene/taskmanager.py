from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from itertools import chain
from accelerate import Accelerator
import torch
import torch.nn as nn
from utils import printf

from .models.heads import (
    DistillHead,
    ContrastiveHead,
)

from .losses import (
    MultiSimilarityLoss,
    DistillLoss,
)

@dataclass
class TaskConfig:
    name: str                    # task name, eg. "contrast", "distill"
    enabled: bool = True         # if the task is enabled
    head_name: Optional[str] = None  # task head name, eg. "contrast_head", "distill_head"
    loss_name: str = ""          # the loss function name, eg. "contrast_loss", "distill_loss"
    loss_weight: float = 1.0     # the loss weight


class TaskManager:
    """
    Core responsibilities:
    1. Initialization: read, assign, and validate task configs; initialize heads and losses.
    2. forward_heads: take features (B, N, D) and run them through the configured heads,
       returning an output dict.
    3. compute_loss: compute different task losses based on the outputs dict.
    """
    
    def __init__(self, 
                 in_dim: int,
                 task_cfgs: List[TaskConfig],
                 head_cfgs: Dict[str, Any],  
                 loss_cfgs: Dict[str, Any],  
                 ):
        self.in_dim = in_dim
        self.task_cfgs = task_cfgs
        self.head_cfgs = head_cfgs
        self.loss_cfgs = loss_cfgs
        self.heads = self._init_heads()
        self.losses = self._init_losses()
            
    def _init_heads(self) -> nn.ModuleDict:
        """Initialize heads based on head_cfgs."""
        heads = nn.ModuleDict()
        
        for head_name, cfg in self.head_cfgs.items():
            printf(f"[TaskManager] init head {head_name}")
            printf(f"head type: {cfg.get('type')}")
            printf(f"head params: {cfg.get('params')}")
            head_type = cfg.get('type')
            head_params = cfg.get('params', {})
            
            if head_type == 'SimpleClusterHead':
                raise NotImplementedError("SimpleClusterHead is not implemented in this version.")
                # heads[head_name] = SimpleClusterHead(
                #     in_dim=self.in_dim, 
                #     K=head_params.get('K', 64), 
                #     hidden=head_params.get('hidden', 512), 
                #     dropout=head_params.get('dropout', 0.1),
                #     tau=head_params.get('tau', 0.07), 
                #     learnable_tau=head_params.get('learnable_tau', False), 
                #     num_layers=head_params.get('num_layers', 2), 
                #     bias=head_params.get('bias', True)
                # )
            elif head_type == 'DistillHead':
                # Remove style from kwargs before passing to projection
                projection_kwargs = {k: v for k, v in head_params.items() if k != 'style'}
                heads[head_name] = DistillHead(
                    in_dim=self.in_dim,
                    style=head_params.get('style', 'moco'), 
                    **projection_kwargs)
            elif head_type == 'ContrastiveHead':
                # Remove style from kwargs before passing to projection
                projection_kwargs = {k: v for k, v in head_params.items() if k != 'style'}
                heads[head_name] = ContrastiveHead(
                    in_dim=self.in_dim,
                    style=head_params.get('style', 'moco'), 
                    **projection_kwargs)
            elif head_type == 'OverlapPredictorHead':
                raise NotImplementedError("OverlapPredictorHead is not implemented in this version.")
                # heads[head_name] = OverlapPredictorHead(
                #     in_dim=self.in_dim, 
                #     use_bias=head_params.get('use_bias', True),
                #     apply_sigmoid=head_params.get('apply_sigmoid', True)
                # )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
        
        printf(f"[TaskManager] Initialized heads: {list(heads.keys())}")
        return heads
    
    def _init_losses(self) -> nn.ModuleDict:
        """Initialize loss functions from loss_cfgs."""
        losses = nn.ModuleDict()
        
        for loss_name, cfg in self.loss_cfgs.items():
            loss_type = cfg.get('type')
            loss_params = cfg.get('params', {})
            
            if loss_type == 'MultiSimilarityLoss':
                losses[loss_name] = MultiSimilarityLoss(
                    pos_th=loss_params.get('pos_th', 0.25),
                    exclude_self=loss_params.get('exclude_self', True),
                    eps=loss_params.get('eps', 1e-8),
                    alpha=loss_params.get('alpha', 2.0),
                    beta=loss_params.get('beta', 50.0),
                    base=loss_params.get('base', 0.5),
                    rank_weight=loss_params.get('rank_weight', 10.0),
                    ov_margin=loss_params.get('ov_margin', 0.05),
                    sim_margin=loss_params.get('sim_margin', 0.05),
                )
            elif loss_type == 'DistillLoss':
                losses[loss_name] = DistillLoss(
                    distill_type=loss_params.get('distill_type', 'relation'),
                    tau=loss_params.get('tau', 4.0),
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        printf(f"[TaskManager] Initialized losses: {list(losses.keys())}")
        return losses
    
    def _validate_cfgs(self):
        for task_cfg in self.task_cfgs:
            if not task_cfg.enabled:
                continue
            if task_cfg.head_name and task_cfg.head_name not in self.head_cfgs:
                raise ValueError(f"Task {task_cfg.name} requires head {task_cfg.head_name} which is not defined in head_cfgs")
            if task_cfg.loss_name and task_cfg.loss_name not in self.loss_cfgs:
                raise ValueError(f"Task {task_cfg.name} requires loss {task_cfg.loss_name} which is not defined in loss_cfgs")
            if task_cfg.loss_weight <= 0:
                raise ValueError(f"Task {task_cfg.name} must have a positive loss_weight")
        printf(f"[TaskManager] Configuration validation passed")
    
    def prepare_with_accelerator(self, accelerator) -> "TaskManager":
        """Prepare heads and losses with the provided accelerator."""
        # Prepare heads (may be wrapped by accelerator for distributed training)
        prepared_heads = accelerator.prepare(self.heads)
        self.heads = prepared_heads
        return self
    
    def forward_heads(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all enabled task heads."""
        outputs = {}
        
        # Handle DDP-wrapped heads
        heads = self.heads
        if hasattr(self.heads, 'module'):  # object wrapped by DDP or similar
            heads = self.heads.module
        
        for task_cfg in self.task_cfgs:
            if not task_cfg.enabled:
                continue
            if task_cfg.head_name and task_cfg.head_name in heads:
                head = heads[task_cfg.head_name]
                outputs[task_cfg.name] = head(features)
            else:
                # If no head is specified for the task, use the input features directly
                outputs[task_cfg.name] = features
        return outputs
    
    def _pick_feat(self, name: str, outs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select the feature tensor for the given task name from outputs dict."""
        if name in outs:
            return outs[name]
        else:
            raise KeyError(f"Output for task {name} not found. Available outputs: {list(outs.keys())}")
    
    def compute_loss(self, 
                     outputs: Dict[str, torch.Tensor], 
                     overlap: torch.Tensor,
                     pair_mask: torch.Tensor,
                     node_mask: torch.Tensor,
                     teacher_features: torch.Tensor,
                     accelerator: Optional[Accelerator] = None
                     ) -> torch.Tensor:
        """Compute the total loss for all enabled tasks."""
        
        device = next(iter(outputs.values())).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses: Dict[str, torch.Tensor] = {}
        
        for task_cfg in self.task_cfgs:
            if not task_cfg.enabled or task_cfg.loss_name not in self.losses:
                continue
            try:
                
                task_out = self._pick_feat(task_cfg.name, outputs)
                loss_fn = self.losses[task_cfg.loss_name]
                
                # Compute the appropriate loss for each task
                if task_cfg.name == "contrast":
                    loss = loss_fn(task_out, overlap, pair_mask)
                elif task_cfg.name == "distill":
                    # distillation loss
                    loss = loss_fn(task_out, teacher_features, pair_mask, node_mask)
                elif task_cfg.name == "huber":
                    # huberpp loss
                    loss = loss_fn(task_out, overlap, pair_mask)
                elif task_cfg.name == "lowrank":
                    # regularization loss (may return extra regularization info)
                    loss, reg_dict = loss_fn(task_out, node_mask)
                    losses.update(reg_dict)
                if isinstance(loss, dict):
                    loss_main = loss.get("main_loss", None)
                    loss_aux = loss.get("aux_loss", None)
                    loss_total = loss.get("loss", None)
                    if loss_total is None:
                        raise ValueError(f"Loss dict for task {task_cfg.name} must include key 'loss'")

                    weighted_total = loss_total * task_cfg.loss_weight
                    losses[f"{task_cfg.name}_loss"] = weighted_total
                    if loss_main is not None:
                        losses[f"{task_cfg.name}_main_loss"] = loss_main * task_cfg.loss_weight
                    if loss_aux is not None:
                        losses[f"{task_cfg.name}_aux_loss"] = loss_aux * task_cfg.loss_weight
                    total_loss = total_loss + weighted_total
                else:
                    weighted_loss = loss * task_cfg.loss_weight
                    losses[f"{task_cfg.name}_loss"] = weighted_loss
                    total_loss = total_loss + weighted_loss
            except Exception as e:
                printf(f"[TaskManager] Failed to compute loss for task {task_cfg.name}: {e}")
                continue
        
        losses["total_loss"] = total_loss
        return losses
        
    # --------- Optimizer parameters (for Trainer) ----------
    def parameters(self):
        return chain(*(m.parameters() for m in self.heads.values()))

    # --------- Checkpoint save/load ----------
    def state_dict(self, *args, **kwargs):
        return {
            "heads": self.heads.state_dict()
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        if "heads" in state:
            self.heads.load_state_dict(state["heads"], strict=strict)

        
head_cfgs = {
    # 对比/蒸馏都用同一个投影头也可以，只要给不同 head_name
    "contrast_head": {
        "type": "ContrastiveHead",
        "params": {
            "style": "dino",      # 或 "dino"
            "out_dim": 256,
            "hidden_dim": 2048,
            "bottleneck_dim": 256,
            "nlayers": 3,
            "use_bn": True,
            "mlp_bias": True,
        }
    },
    "distill_head": {
        "type": "DistillHead",
        "params": {
            "style": "moco",      # 或 "moco"
            "out_dim": 256,
            "hidden_dim": 2048,
            "nlayers": 3, 
            "last_bn": False
        }
    },
    # "cluster_head": {
    #     "type": "SimpleClusterHead",
    #     "params": {
    #         "K": 64,
    #         "hidden": 512,
    #         "dropout": 0.1,
    #         "tau": 0.07,
    #         "learnable_tau": False,
    #         "num_layers": 2,
    #         "bias": True,
    #     }
    # },
    # "overlap_head": {
    #     "type": "OverlapPredictorHead",
    #     "params": {
    #         "use_bias": True,
    #         "apply_sigmoid": True
    #     }
    # }
}

loss_cfgs = {
    "contrast_loss": {
        "type": "MultiSimilarityLoss",
        "params": {
            "pos_th": 0.25,
            "exclude_self": True,
            "eps": 1e-8,
            "alpha": 2.0,
            "beta": 50.0,
            "base": 0.5,
            "rank_weight": 10.0,
            "ov_margin": 0.05,
            "sim_margin": 0.05,
        }
    },
    "distill_loss": {
        "type": "DistillLoss",
        "params": {
            "distill_type": "relation",   # "relation" | "cosine" | "mse" | "kl"
            "tau": 4.0
        }
    },
    # "huber_loss": {
    #     "type": "HuberPPLoss",
    #     "params": {
    #         "delta": 0.1,
    #         "weight_mode": "overlap",   # "none" | "posneg" | "overlap"
    #         "pos_weight": 2.0,
    #         "neg_weight": 1.5,
    #         "pos_th": 0.25,
    #         "gamma": 0.7
    #     }
    # },
    # "lowrank_loss": {
    #     "type": "LowRankRegularizer",
    #     "params": {
    #         "weights": (1.0, 0.1, 1.0)  # (balance, entropy, decor)
    #     }
    # }
}

task_cfgs = [
    TaskConfig(name="contrast",   enabled=True,  head_name="contrast_head", loss_name="contrast_loss", loss_weight=1.0),
    TaskConfig(name="huber",      enabled=True,  head_name="overlap_head",  loss_name="huber_loss",    loss_weight=0.2),
    # TaskConfig(name="lowrank",    enabled=True,  head_name="cluster_head",  loss_name="lowrank_loss",  loss_weight=0.1),
    # TaskConfig(name="distill",    enabled=False, head_name="distill_head",  loss_name="distill_loss",  loss_weight=0.2),
]

