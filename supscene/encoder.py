import os
from typing import Optional, Dict, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== import your modules ======
# backbone
from .models import (
    DINOv2, ResNet
)
# aggregator
from .models import (
    NetVLAD, GeMPool, SCPP
)
# heads
from .models import DeployHead

# =========================
# （AvgPool / CLS / Identity）
# =========================
class GlobalAvgPool(nn.Module):
    """Global average pooling aggregator: (B,C,H,W) → (B,C)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.output_dim = int(in_dim)

    def forward(self, x: torch.Tensor, token: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B,C)
        assert z.size(1) == self.output_dim, f"Output dim mismatch: {z.size(1)} != {self.output_dim}"
        return z

class CLSTokenAgg(nn.Module):
    """Use backbone CLS token as global descriptor.

    Requires backbone to return (feat, token) or token alone.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.output_dim = int(in_dim)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        X, t = x
        if t is None:
            raise RuntimeError("CLSTokenAgg requires `token` from backbone (return_cls_token=True).")
        if t.dim() != 2:
            t = t.view(t.size(0), -1)
        assert t.size(1) == self.output_dim, f"Output dim mismatch: {t.size(1)} != {self.output_dim}"
        return t

# =========================
# Factories
# =========================
def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """Backbone factory.

    Args:
        cfg: {"name": str, "args": dict}

    Returns:
        nn.Module: Backbone with `.output_dim`.
    """
    name = cfg.get("name", "").lower()
    args = cfg.get("args", {}) or {}
    if name in ("dinov2", "dino", "dinov2_vit"):
        bb = DINOv2(**args)
    elif name in ("resnet", "ResNet"):
        bb = ResNet(**args)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    # propagate out_dim to config for downstream components
    cfg.setdefault("args", {})
    cfg["args"]["out_dim"] = getattr(bb, "output_dim")
    return bb

def build_aggregator(cfg: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
    """Aggregator factory.

    Args:
        cfg: {"name": str, "args": dict} or None

    Returns:
        nn.Module | None
    """
    if cfg is None:
        return None
    name = cfg.get("name", "").lower()
    args = cfg.get("args", {}) or {}
    
    if name in ("netvlad"):
        agg = NetVLAD(**args)
    elif name in ("gem", "gempool", "gem_pool"):
        agg = GeMPool(**args)
    elif name in ("scpp"):
        agg = SCPP(**args)
    elif name in ("avg", "avgpool"):
        agg = GlobalAvgPool(**args)
    elif name in ("cls", "clstoken"):
        agg = CLSTokenAgg(**args)
    else:
        raise ValueError(f"Unknown aggregator: {name}")
    
    cfg.setdefault("args", {})
    cfg["args"]["out_dim"] = getattr(agg, "output_dim")
    return agg

def build_deploy_head(cfg: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
    """Deploy head factory (optional)."""
    if cfg is None:
        return None
    name = (cfg.get("name") or "").lower()
    args = dict(cfg.get("args", {}) or {})

    if DeployHead is not None and (name in ("mlp", "deploy_head", "")):
        return DeployHead(**args)
    raise ValueError(f"Unknown deploy head: {name}")

# =========================
# Optional: LoRA/PEFT injection
# =========================
# def maybe_apply_peft(model: nn.Module, peft_cfg: Optional[Dict[str, Any]]) -> nn.Module:
#     """Apply LoRA adapters if peft_cfg is provided and PEFT is installed.

#     Args:
#         model: Target module.
#         peft_cfg: Config with keys {enable, target_modules, r, alpha, dropout}.

#     Returns:
#         nn.Module: Wrapped model.
#     """
#     if not peft_cfg or not peft_cfg.get("enable", False):
#         return model

#     try:
#         from peft import LoraConfig, get_peft_model  # type: ignore
#     except Exception as e:  # pragma: no cover
#         print(f"[PEFT] peft not available, skip: {e}")
#         return model

#     tmods = peft_cfg.get("target_modules")
#     lcfg = LoraConfig(
#         r=int(peft_cfg.get("r", 8)),
#         lora_alpha=int(peft_cfg.get("alpha", 16)),
#         lora_dropout=float(peft_cfg.get("dropout", 0.0)),
#         target_modules=tmods,
#         bias="none",
#         task_type="FEATURE_EXTRACTION",
#     )
#     model = get_peft_model(model, lcfg)
#     print("[PEFT] LoRA adapters injected.")
#     return model



# =========================
# Composed model: SupSceneEncoder
# =========================
class SupSceneEncoder(nn.Module):  
    """Unified deployable encoder.

    Flow: images → backbone → aggregator → (optional head) → z (L2)

    Args:
        backbone: Feature extractor.
        aggregator: Global aggregator.
        deploy_head: Optional projection head.
        return_cls_token (bool): If True, enforce CLS token return on supported backbones.
    """
    def __init__(
        self,
        backbone: nn.Module,
        aggregator: nn.Module,
        deploy_head: Optional[nn.Module] = None,
        return_cls_token: bool = False,  
        use_bn: bool = True,
        whitening: bool = False,
        whitening_dim: int = 256,
        final_norm: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.deploy_head = deploy_head
        self.return_cls_token = return_cls_token
        self.use_bn = bool(use_bn)
        self.whitening = bool(whitening)
        self.final_norm = bool(final_norm)

        feat_dim = getattr(self.aggregator, "output_dim", None)
        if feat_dim is None:
            raise ValueError("Aggregator must expose `output_dim` to build BN/whitening.")

        if self.use_bn:
            self.bn = nn.BatchNorm1d(feat_dim, affine=True)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        else:
            self.bn = nn.Identity()

        if self.whitening:
            if whitening_dim > feat_dim:
                raise ValueError(
                    f"whitening_dim ({whitening_dim}) cannot be greater than aggregator output dim ({feat_dim})"
                )
            self.whitening_layer = nn.Linear(feat_dim, whitening_dim, bias=True)
            self._output_dim = int(whitening_dim)
        else:
            self.whitening_layer = nn.Identity()
            self._output_dim = int(feat_dim)

        # If CLS aggregator is used and backbone supports it, enable token output
        if isinstance(self.aggregator, CLSTokenAgg) and hasattr(self.backbone, "return_cls_token"):
            if not getattr(self.backbone, "return_cls_token"):
                setattr(self.backbone, "return_cls_token", True)

    @property
    def output_dim(self) -> Optional[int]:
        if self.deploy_head is not None and hasattr(self.deploy_head, "output_dim"):
            dim = getattr(self.deploy_head, "output_dim", None)
            if dim is not None:
                return int(dim)
        return int(self._output_dim)

    @torch.no_grad()
    def init_whitening(self, X: torch.Tensor, eps: float = 1e-4) -> None:
        if not self.whitening or not isinstance(self.whitening_layer, nn.Linear):
            return

        device = self.whitening_layer.weight.device
        d_in = self.whitening_layer.in_features
        d_out = self.whitening_layer.out_features

        X = X.to(device).float()
        if X.dim() != 2 or X.size(1) != d_in:
            raise ValueError(f"X should be (N, {d_in}), got {tuple(X.shape)}")

        n = X.size(0)
        X64 = X.double()
        mean = X64.mean(dim=0)
        Xc = X64 - mean

        cov = (Xc.t() @ Xc) / max(n - 1, 1)
        cov = cov + eps * torch.eye(d_in, device=device, dtype=torch.float64)

        S, U = torch.linalg.eigh(cov)
        idx = torch.argsort(S, descending=True)[:d_out]
        S = S[idx]
        U = U[:, idx]

        whitening_matrix = U * torch.rsqrt(S).unsqueeze(0)
        new_weight = whitening_matrix.t().float()
        new_bias = (-(mean @ whitening_matrix)).float()

        self.whitening_layer.weight.data.copy_(new_weight)
        self.whitening_layer.bias.data.copy_(new_bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            images (Tensor): (B,3,H,W)

        Returns:
            Tensor: z ∈ R^{B×D_out}, L2‑normalized.
        """
        # backbone forward
        out = self.backbone(images)

        # aggregator forward
        z = self.aggregator(out)

        z = self.bn(z)
        z = self.whitening_layer(z)

        # optional deploy head
        if self.deploy_head is not None:
            z = self.deploy_head(z)

        if self.final_norm:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def load_weights(
        self,
        weights_cfg: Dict[str, Any],
        map_location: Union[str, torch.device, None] = None,
    ):
        """Load weights for aggregator/backbone/head from various sources.

        Args:
            weights_cfg (dict): See docstring in create_encoder.
            map_location: torch.load map location.
        """
        weights_type = weights_cfg.get('type', 'file')

        if weights_type == 'file':
            self._load_file_weights(weights_cfg, map_location)
        else:
            raise ValueError(f"Unsupported weights type: {weights_type}")
    
    def _load_file_weights(
        self,
        weights_cfg: Dict[str, Any],
        map_location: Union[str, torch.device, None] = None,
    ):

        ckpt = weights_cfg.get('path')

        if not ckpt or not os.path.isfile(ckpt):
            print(f"[SupSceneEncoder] Weight file not found: {ckpt}")
            return

        print(f"[SupSceneEncoder] Loading weights from file: {ckpt}")
        try:
            state = torch.load(ckpt, map_location=map_location or torch.device('cpu'))
        except Exception as e:
            print(f"[SupSceneEncoder] Failed to load checkpoint '{ckpt}': {e}")
            return

        # support checkpoints that wrap state_dict or contain nested dicts
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        try:
            bb_state = {}
            agg_state = {}
            
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    bb_state[k.replace("backbone.", "")] = v
                elif k.startswith("aggregator."):
                    agg_state[k.replace("aggregator.", "")] = v
                else:
                    agg_state[k] = v
            
            if bb_state:
                inc = self.backbone.load_state_dict(bb_state, strict=False)
                bb_missing = getattr(inc, "missing_keys", [])
                bb_unexpected = getattr(inc, "unexpected_keys", [])
                print(f"[SupSceneEncoder] Loaded backbone weights (missing={len(bb_missing)}, unexpected={len(bb_unexpected)})")
            if agg_state:
                inc = self.aggregator.load_state_dict(agg_state, strict=False)
                agg_missing = getattr(inc, "missing_keys", [])
                agg_unexpected = getattr(inc, "unexpected_keys", [])
                print(f"[SupSceneEncoder] Loaded aggregator weights (missing={len(agg_missing)}, unexpected={len(agg_unexpected)})")
            
        except Exception as e:
            print(f"[SupSceneEncoder] Failed to load weights: {e}")
            raise

# =========================
# Config‑driven builder
# =========================

def create_encoder(cfg: Dict[str, Any]) -> SupSceneEncoder:
    """Build `SupSceneEncoder` from config.

    Example cfg:
    {
      "backbone": {"name": "dinov2", "args": {"model_name": "dinov2_vitb14", "num_trainable_blocks": 2, "return_cls_token": false}},
      "aggregator": {"name": "netvlad", "args": {"num_clusters": 64, "in_dim": 768}},
      "deploy_head": {"name": "mlp", "args": {"in_dim": 8192, "out_dim": 512, "hidden_dim": 1024}},
      "normalize": true,
      "peft": {"enable": false, "target": "backbone", "type": "lora", "r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["qkv", "proj"]},
      "weights": {"type": "file", "path": "path/to/agg.ckpt", "strict": false}
    }

    Notes:
        - If aggregator/head args lack `in_dim`, they are filled from upstream.
    """
    # 1) Backbone
    bb = build_backbone(cfg.get("backbone", {}))

    # 2) Aggregator
    agg_cfg = dict(cfg.get("aggregator") or {})
    if agg_cfg.get("args") is not None:
        a = agg_cfg["args"]
        if "in_dim" not in a:
            a["in_dim"] = getattr(bb, "output_dim")
    agg = build_aggregator(agg_cfg)

    # 3) Deploy head (optional; default path keeps projection disabled)
    head_cfg = dict(cfg.get("deploy_head") or {})
    head = None
    if head_cfg:
        hargs = head_cfg.get("args", {}) or {}
        use_projection = bool(hargs.get("use_projection", False))
        has_out_dim = hargs.get("out_dim", None) is not None
        if use_projection or has_out_dim:
            if "in_dim" not in hargs:
                hargs["in_dim"] = getattr(agg, "output_dim")
            head_cfg["args"] = hargs
            head = build_deploy_head(head_cfg)

    model_cfg = cfg or {}
    enc = SupSceneEncoder(
        backbone=bb,
        aggregator=agg,
        deploy_head=head,
        use_bn=bool(model_cfg.get("use_bn", True)),
        whitening=bool(model_cfg.get("whitening", False)),
        whitening_dim=int(model_cfg.get("whitening_dim", 256)),
        final_norm=bool(model_cfg.get("final_norm", True)),
    )

    # # 4) Optional PEFT
    # peft_cfg = cfg.get("peft")
    # if peft_cfg and peft_cfg.get("enable", False):
    #     target = (peft_cfg.get("target", "backbone") or "backbone").lower()
    #     if target == "backbone":
    #         enc.backbone = maybe_apply_peft(enc.backbone, peft_cfg)
    #     elif target == "aggregator":
    #         enc.aggregator = maybe_apply_peft(enc.aggregator, peft_cfg)
    #     elif target in ("head", "deploy_head") and enc.deploy_head is not None:
    #         enc.deploy_head = maybe_apply_peft(enc.deploy_head, peft_cfg)
    #     else:
    #         print(f"[PEFT] Unknown target '{target}', skip.")

    # 5) Optional weights
    weights_cfg = cfg.get("weights", {}) or {}
    if weights_cfg:
        enc.load_weights(
            weights_cfg=weights_cfg,
            map_location="cpu",
        )
    return enc
