# orbit/orbit.py
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
    NetVLAD, GeMPool, BoQ, GAP, MTP, SALAD, APA, AttnAGG3d
)
# heads
from .models import DeployHead

# =========================
# 小型内置聚合器（AvgPool / CLS / Identity）
# =========================
class GlobalAvgPool(nn.Module):
    """自带的轻量全局平均池化聚合器：B,C,H,W -> B,C"""
    def __init__(self, in_dim):
        super().__init__()
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor, token: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,C,H,W]
        z = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, C]
        assert z.size(1) == self.output_dim, f"Output dimension mismatch: {z.size(1)} != {self.output_dim}"
        return z

class CLSTokenAgg(nn.Module):
    """
    使用 backbone 的 CLS token 作为聚合向量。
    要求 backbone 在 forward 时能返回 (feat, token) 或单独 token。
    """
    def __init__(self, in_dim):
        super().__init__()
        self.output_dim = in_dim

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, token = x
        if token is None:
            raise RuntimeError("CLSTokenAgg requires `token` from backbone (return_cls_token=True).")
        if token.dim() != 2:
            token = token.view(token.size(0), -1)
        assert token.size(1) == self.output_dim, f"Output dimension mismatch: {token.size(1)} != {self.output_dim}"
        return token


# =========================
# Backbone 工厂
# =========================
def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg.get("name", "").lower()
    args = cfg.get("args", {}) or {}
    if name in ("dinov2", "dino", "dinov2_vit"):
        backbone = DINOv2(**args)
    elif name in ("resnet", "rn"):
        backbone = ResNet(**args)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    cfg["args"]["out_dim"] = backbone.output_dim
    return backbone

# =========================
# Aggregator 工厂
# =========================
def build_aggregator(cfg: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
    if cfg is None:
        return None
    name = cfg.get("name", "").lower()
    args = cfg.get("args", {}) or {}
    
    if name in ("netvlad", "vlad"):
        agg = NetVLAD(**args)
    elif name in ("gem", "gempool", "gem_pool"):
        agg = GeMPool(**args)
    elif name in ("boq", "bag_of_queries"):
        agg = BoQ(**args)
    elif name in ("salad", "salad_vl"):
        agg = SALAD(**args)
    elif name in ("avg", "avgpool"):
        agg = GlobalAvgPool(**args)
    elif name in ("cls", "clstoken"):
        agg = CLSTokenAgg(**args)
    elif name in ("gap", "gaussian_anchored_pool"):
        agg = GAP(**args)
    elif name in ("mtp", "moment_token_pool"):
        agg = MTP(**args)
    elif name in ("apa", "attention_pooling_aggregator"):
        agg = APA(**args)
    elif name in ("attn3d", "attn3d_pool"):
        agg = AttnAGG3d(**args)
    else:
        raise ValueError(f"Unknown aggregator: {name}")
    cfg["args"]["out_dim"] = agg.output_dim
    return agg

# =========================
# Deploy Head 工厂（可选）
# =========================
def build_deploy_head(cfg: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
    if cfg is None:
        return None
    name = (cfg.get("name") or "").lower()
    args = cfg.get("args", {}) or {}

    # 你可以在 heads/deploy_head.py 中定义多种 head，这里只给一个示例
    if DeployHead is not None and (name in ("mlp", "deploy_head", "")):
        return DeployHead(**args)
    else:
        raise ValueError(f"Unknown deploy head: {name}")


# =========================
# 可选：LoRA/PEFT 注入
# =========================
def maybe_apply_peft(model: nn.Module, peft_cfg: Optional[Dict[str, Any]]) -> nn.Module:
    """
    若传入 peft_cfg 且已安装 peft，则对 `model` 应用 LoRA 适配器。
    典型 peft_cfg:
    {
      "enable": true,
      "target": "backbone",   # "backbone" | "aggregator" | "head"
      "type": "lora",
      "r": 16, "alpha": 32, "dropout": 0.1,
      "target_modules": ["q_proj","k_proj","v_proj","out_proj"]  # 对 ViT/Transformer
    }
    """
    if not peft_cfg or not peft_cfg.get("enable", False):
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        print(f"[PEFT] peft not available, skip applying adapters: {e}")
        return model

    tmods = peft_cfg.get("target_modules", None)
    lcfg = LoraConfig(
        r=int(peft_cfg.get("r", 8)),
        lora_alpha=int(peft_cfg.get("alpha", 16)),
        lora_dropout=float(peft_cfg.get("dropout", 0.0)),
        target_modules=tmods,
        bias="none",
        task_type="FEATURE_EXTRACTION",  # 不产生分类头
    )
    model = get_peft_model(model, lcfg)
    print("[PEFT] LoRA adapters injected.")
    return model


# =========================
# 组合模型：OrbitEncoder
# =========================
class OrbitEncoder(nn.Module):
    """
    统一的可部署 Encoder：
      images -> backbone -> aggregator -> (optional deploy head) -> z (L2 norm)
    """
    def __init__(
        self,
        backbone: nn.Module,
        aggregator: nn.Module,
        deploy_head: nn.Module,
        return_cls_token: bool = False,  # 若 aggregator=CLS，需要 True
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.deploy_head = deploy_head
        self.return_cls_token = return_cls_token

        # 若使用 CLS 聚合且 backbone 可设置 return_cls_token，强制打开
        if isinstance(self.aggregator, CLSTokenAgg) and hasattr(self.backbone, "return_cls_token"):
            if not getattr(self.backbone, "return_cls_token"):
                setattr(self.backbone, "return_cls_token", True)

    @property
    def output_dim(self) -> Optional[int]:
        """获取最终输出维度，优先级：deploy_head > aggregator"""
        # 使用 deploy_head 的输出维度
        if hasattr(self.deploy_head, "output_dim"):
            dim = getattr(self.deploy_head, "output_dim", None)
            if dim is not None:
                return int(dim)
        return None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B,3,H,W]
        Returns:
            z: [B, D_out] (默认 L2-normalized)
        """
        # backbone forward
        out = self.backbone(images)
        if isinstance(out, tuple):
            feat, token = out
        else:
            feat, token = out, None  # 大多数聚合器用特征图

        # aggregator forward
        if isinstance(self.aggregator, CLSTokenAgg):
            z = self.aggregator((feat, token))  # 使用 token
        elif isinstance(self.aggregator, SALAD):
            z = self.aggregator((feat, token))
        elif isinstance(self.aggregator, APA):
            z = self.aggregator((feat, token))
        elif isinstance(self.aggregator, AttnAGG3d):
            z = self.aggregator((feat, token))
        else:
            # feat 可能是 [B,C,H,W] 或已聚合好的 [B,D]
            z = self.aggregator(feat)

        # optional deploy head
        z = self.deploy_head(z)
        return z

    # --------- 统一权重加载接口 ---------
    def load_weights(
        self,
        weights_cfg: Dict[str, Any],
        map_location: Union[str, torch.device, None] = None,
    ):
        """
        统一的权重加载接口，支持不同聚合器的加载方式
        
        Args:
            weights_cfg: 权重配置字典
                {
                    "type": "boq" | "netvlad" | "file",
                    "backbone": "resnet50" | "dinov2",  # for boq
                    "output_dim": 16384,  # for boq
                    "path": "/path/to/weights.pth",  # for file
                    "strict": False
                }
        """
        weights_type = weights_cfg.get('type', 'file')
        
        if weights_type == 'boq':
            self._load_boq_weights(weights_cfg, map_location)
        elif weights_type == 'salad':
            self._load_salad_weights(weights_cfg, map_location)
        elif weights_type == 'file':
            self._load_file_weights(weights_cfg, map_location)
        else:
            raise ValueError(f"Unsupported weights type: {weights_type}")
    
    def _load_boq_weights(
        self,
        weights_cfg: Dict[str, Any],
        map_location: Union[str, torch.device, None] = None,
    ):
        """加载 BoQ 预训练权重 - 包含backbone和aggregator"""
        # 定义模型URL映射
        MODEL_URLS = {
            "resnet50": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
            "dinov2": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
        }
        
        backbone_name = weights_cfg.get('backbone', 'resnet50')
        model_key = f"{backbone_name}"
        
        if model_key not in MODEL_URLS:
            raise ValueError(f"Unsupported BoQ model: {model_key}. Available: {list(MODEL_URLS.keys())}")
        
        print(f"[OrbitEncoder] Loading pretrained BoQ weights for {backbone_name}")
        
        try:
            # 直接从URL加载状态字典
            state_dict = torch.hub.load_state_dict_from_url(
                MODEL_URLS[model_key],
                map_location=map_location or torch.device('cpu')
            )
            # print("state_dict keys:", state_dict.keys())
            # 分离backbone和aggregator的权重
            backbone_state = {}
            aggregator_state = {}
            
            for key, value in state_dict.items():
                if key.startswith('backbone.dino'):
                    # 移除 'backbone.' 前缀
                    new_key = key.replace('backbone.dino', 'model')
                    backbone_state[new_key] = value
                elif key.startswith('backbone.net'):
                    # 移除 'backbone.' 前缀
                    new_key = key.replace('backbone.net', 'model')
                    backbone_state[new_key] = value
                elif key.startswith('aggregator.'):
                    # 移除 'aggregator.' 前缀
                    new_key = key.replace('aggregator.', '')
                    aggregator_state[new_key] = value
                else:
                    # 默认分配给aggregator
                    aggregator_state[key] = value
            
            # 加载backbone权重
            if backbone_state:
                backbone_incompatible = self.backbone.load_state_dict(backbone_state, strict=False)
                backbone_missing = getattr(backbone_incompatible, "missing_keys", [])
                backbone_unexpected = getattr(backbone_incompatible, "unexpected_keys", [])
                print(f"[OrbitEncoder] Loaded backbone weights (missing={len(backbone_missing)}, unexpected={len(backbone_unexpected)})")

            # 加载aggregator权重
            if aggregator_state:
                agg_incompatible = self.aggregator.load_state_dict(aggregator_state, strict=False)
                agg_missing = getattr(agg_incompatible, "missing_keys", [])
                agg_unexpected = getattr(agg_incompatible, "unexpected_keys", [])
                print(f"[OrbitEncoder] Loaded aggregator weights (missing={len(agg_missing)}, unexpected={len(agg_unexpected)})")
            
        except Exception as e:
            print(f"[OrbitEncoder] Failed to load BoQ weights: {e}")
            raise
    
    def _load_salad_weights(
        self,
        weights_cfg: Dict[str, Any],
        map_location: Union[str, torch.device, None] = None,
    ):
        """加载 SALAD 预训练权重 - 基于DINOv2的NetVLAD模型"""
        SALAD_URL = 'https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt'
        
        print(f"[OrbitEncoder] Loading pretrained SALAD weights")
        
        try:
            # 从URL加载状态字典
            state_dict = torch.hub.load_state_dict_from_url(
                SALAD_URL,
                map_location=map_location or torch.device('cpu')
            )
            
            # 分离backbone和aggregator的权重
            backbone_state = {}
            aggregator_state = {}
            
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    # 移除 'backbone.' 前缀，映射到model
                    new_key = key.replace('backbone.', '')
                    backbone_state[new_key] = value
                elif key.startswith('aggregator.'):
                    # 移除 'aggregator.' 前缀
                    new_key = key.replace('aggregator.', '')
                    aggregator_state[new_key] = value
                else:
                    # 默认分配给aggregator
                    aggregator_state[key] = value
            
            # 加载backbone权重
            if backbone_state:
                backbone_incompatible = self.backbone.load_state_dict(backbone_state, strict=False)
                backbone_missing = getattr(backbone_incompatible, "missing_keys", [])
                backbone_unexpected = getattr(backbone_incompatible, "unexpected_keys", [])
                print(f"[OrbitEncoder] Loaded backbone weights (missing={len(backbone_missing)}, unexpected={len(backbone_unexpected)})")

            # 加载aggregator权重
            if aggregator_state:
                agg_incompatible = self.aggregator.load_state_dict(aggregator_state, strict=False)
                agg_missing = getattr(agg_incompatible, "missing_keys", [])
                agg_unexpected = getattr(agg_incompatible, "unexpected_keys", [])
                print(f"[OrbitEncoder] Loaded aggregator weights (missing={len(agg_missing)}, unexpected={len(agg_unexpected)})")
            
        except Exception as e:
            print(f"[OrbitEncoder] Failed to load SALAD weights: {e}")
            raise
    
    def _load_file_weights(
        self,
        weights_cfg: Dict[str, Any],
        map_location: Union[str, torch.device, None] = None,
    ):
        """从文件加载权重"""
        agg_ckpt = weights_cfg.get('path')
        strict = weights_cfg.get('strict', False)
        
        if not agg_ckpt or not os.path.isfile(agg_ckpt):
            print(f"[OrbitEncoder] Weight file not found: {agg_ckpt}")
            return
            
        state = torch.load(agg_ckpt, map_location=map_location or "cpu")
        key = "state_dict" if isinstance(state, dict) and "state_dict" in state else None
        sd = state[key] if key else state
        
        incompatible = self.aggregator.load_state_dict(sd, strict=strict)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(f"[OrbitEncoder] Loaded aggregator weights: {agg_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")

# =========================
# 工厂：配置驱动构建
# =========================
def create_orbit(cfg: Dict[str, Any]) -> OrbitEncoder:
    """
    从配置构建 OrbitEncoder

    典型 cfg:
    {
      "backbone": {"name": "dinov2", "args": {"model_name": "dinov2_vitb14", "num_trainable_blocks": 2, "return_cls_token": false}},
      "aggregator": {"name": "netvlad", "args": {"num_clusters": 64, "dim": 768}},
      "deploy_head": {"name": "mlp", "args": {"in_dim": 8192, "out_dim": 512, "hidden_dim": 1024}},
      "normalize": true,
      "peft": { "enable": false, "target": "backbone", "type": "lora", "r": 16, "alpha": 32, "dropout": 0.1,
                "target_modules": ["qkv","proj"] },

      "init_ckpt": {
         "aggregator": "path/to/agg.ckpt",
         "head": "path/to/head.ckpt",
         "strict": false
      }
    }
    """
    # 1) backbone
    bb = build_backbone(cfg.get("backbone", {}))
   
    # 2) aggregator
    agg_cfg = cfg.get("aggregator")
    if agg_cfg and agg_cfg.get("args") is not None:
        args = agg_cfg["args"]
        if "in_dim" not in args:
            args["in_dim"] = bb.output_dim

    agg = build_aggregator(agg_cfg)

    # 3) deploy head
    head_cfg = cfg.get("deploy_head")
    if head_cfg and head_cfg.get("args") is not None:
        args = head_cfg["args"]
        if "in_dim" not in args:
            args["in_dim"] = agg.output_dim
    head = build_deploy_head(head_cfg)

    enc = OrbitEncoder(backbone=bb, aggregator=agg, deploy_head=head)

    # 5) 可选：对某个组件注入 PEFT（LoRA）
    peft_cfg = cfg.get("peft")
    if peft_cfg and peft_cfg.get("enable", False):
        target = peft_cfg.get("target", "backbone").lower()
        if target == "backbone":
            enc.backbone = maybe_apply_peft(enc.backbone, peft_cfg)
        elif target == "aggregator":
            enc.aggregator = maybe_apply_peft(enc.aggregator, peft_cfg)
        elif target in ("head", "deploy_head"):
            enc.deploy_head = maybe_apply_peft(enc.deploy_head, peft_cfg)
        else:
            print(f"[PEFT] Unknown target '{target}', skip.")

    # 6) 可选：加载权重
    weights_cfg = cfg.get("weights", {}) or {}
    if weights_cfg:
        enc.load_weights(
            weights_cfg=weights_cfg,
            map_location="cpu",
        )
    return enc
