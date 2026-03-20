"""Torch Hub entrypoints for SupScene.

Usage:
    import torch
    model = torch.hub.load("<org>/<repo>", "dinov2_scpp_supscene_1536", pretrained=True)

For local testing in repo root:
    model = torch.hub.load(".", "dinov2_scpp_supscene_1536", source="local", pretrained=False)
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from supscene.encoder import create_encoder

import torch

dependencies = ["torch", "torchvision", "numpy", "peft", "safetensors"]


# Replace this with your real release URL after uploading the file.
DEFAULT_WEIGHT_URL = "https://github.com/Suxilan/SupScene/releases/latest/download/dinov2_scpp_supscene_1536.pth"
DEFAULT_WEIGHT_FILENAME = "dinov2_scpp_supscene_1536.pth"


# Frozen deploy-time model spec that matches released weights.
MODEL_CFG = {
    "backbone": {
        "name": "dinov2",
        "args": {
            "model_name": "dinov2_vitb14",
            "num_trainable_blocks": 12,
            "return_attn_maps": False,
            "return_cls_token": False,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_bias": "none",
            "lora_targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        },
    },
    "aggregator": {
        "name": "scpp",
        "args": {},
    },
    "deploy_head": {
        "name": "deploy_head",
        "args": {
            "use_projection": False,
        },
    },
    "use_bn": True,
    "whitening": True,
    "whitening_dim": 1536,
    "final_norm": True,
    "weights": None,
}


def _is_url(x: str) -> bool:
    if not x:
        return False
    p = urlparse(x)
    return p.scheme in ("http", "https") and bool(p.netloc)


def _resolve_weight_path(weights: str | None) -> str | None:
    if weights is None:
        return None
    if _is_url(weights):
        return torch.hub.load_state_dict_from_url(weights, map_location="cpu", progress=True, check_hash=False)
    p = Path(weights)
    if not p.exists():
        raise FileNotFoundError(f"Weight file not found: {weights}")
    return str(p)


def _build_model():
    model = create_encoder(dict(MODEL_CFG))
    return model


def _normalize_state_dict(obj: dict) -> dict:
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _load_state(model: torch.nn.Module, state_or_path):
    if state_or_path is None:
        return model

    if isinstance(state_or_path, dict):
        state = _normalize_state_dict(state_or_path)
    else:
        ckpt = torch.load(state_or_path, map_location="cpu")
        state = _normalize_state_dict(ckpt)

    if not isinstance(state, dict):
        raise RuntimeError("Unsupported weight format for torch hub loading")

    missing = model.load_state_dict(state, strict=False)
    print(f"[TorchHub] missing keys: {len(getattr(missing, 'missing_keys', []))}")
    print(f"[TorchHub] unexpected keys: {len(getattr(missing, 'unexpected_keys', []))}")
    return model


def dinov2_scpp_supscene_1536(
    pretrained: bool = False,
    weights: str | None = None,
    map_location: str = "cpu",
):
    """Build SupScene encoder and optionally load released pretrained weights.

    Args:
        pretrained: If True, load the released weight file.
        weights: Optional local file path or URL. If provided, overrides default release URL.
        map_location: Device string for model.to().
    """
    model = _build_model()

    if pretrained:
        if weights is None and DEFAULT_WEIGHT_URL:
            weights = DEFAULT_WEIGHT_URL
        elif weights is None:
            local_default = Path("weights") / DEFAULT_WEIGHT_FILENAME
            if local_default.exists():
                weights = str(local_default)
            else:
                raise RuntimeError(
                    "No default release URL configured yet. "
                    "Provide weights=<local path or url>, or set DEFAULT_WEIGHT_URL in hubconf.py"
                )

        state_or_path = _resolve_weight_path(weights)
        model = _load_state(model, state_or_path)

    model = model.to(map_location).eval()
    return model
