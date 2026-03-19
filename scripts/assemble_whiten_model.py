#!/usr/bin/env python3
"""Assemble SupScene model with initialized whitening layer.

Workflow:
1) Build model from config with whitening enabled.
2) Optionally load base weights.
3) Load extracted feature matrix (.npy/.pt/.h5).
4) Initialize whitening via model.init_whitening(X).
5) Save assembled model state_dict.

Example:
  python scripts/assemble_whiten_model.py \
      --config configs/peft-dinov2-scpp-lora.yaml \
      --weights experiments/peft-dinov2-scpp-lora/checkpoints/last.pth \
      --features cache/features_for_whitening.npy \
      --whitening-dim 1024 \
      --out weights/peft-dinov2-scpp-lora-whiten.pth
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.conf import load_config
from supscene import create_encoder


def _load_safetensors(path: Path) -> dict:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "Detected safetensors checkpoint, but safetensors is unavailable. "
            "Install with: pip install safetensors"
        ) from exc
    return load_file(str(path), device="cpu")


def load_checkpoint_state(weights_path: str) -> dict:
    p = Path(weights_path)

    if not p.exists() and p.suffix == ".pth":
        alt_dir = p.with_suffix("")
        if alt_dir.exists() and alt_dir.is_dir():
            print(f"[info] '{p}' not found, use accelerate checkpoint dir: '{alt_dir}'")
            p = alt_dir

    if not p.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    if p.is_dir():
        st_main = p / "model.safetensors"
        if st_main.exists():
            return _load_safetensors(st_main)

        st_shard = p / "model_1.safetensors"
        if st_shard.exists():
            return _load_safetensors(st_shard)

        raise RuntimeError(
            f"Unsupported checkpoint directory: {p}. "
            "Expected model.safetensors or model_1.safetensors"
        )

    if p.suffix == ".safetensors":
        return _load_safetensors(p)

    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(p), map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]

    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise RuntimeError(f"Unsupported checkpoint format: {p}")


def load_features(path: str) -> torch.Tensor:
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".npy":
        arr = np.load(p)
        return torch.from_numpy(arr).float()

    if suf == ".pt" or suf == ".pth":
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.float()
        if isinstance(obj, dict):
            for k in ("features", "feats", "embeddings", "x"):
                if k in obj:
                    v = obj[k]
                    if isinstance(v, np.ndarray):
                        return torch.from_numpy(v).float()
                    if isinstance(v, torch.Tensor):
                        return v.float()
        raise RuntimeError(f"Cannot find features tensor in: {p}")

    if suf == ".h5" or suf == ".hdf5":
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError("h5py is required for .h5 inputs. Install with: pip install h5py") from exc

        with h5py.File(p, "r") as f:
            for k in ("features", "feats", "embeddings", "x"):
                if k in f:
                    return torch.from_numpy(f[k][:]).float()
        raise RuntimeError(f"Cannot find feature dataset in: {p}")

    raise RuntimeError(f"Unsupported feature file: {p}")


def load_model(config_path: str, weights_path: str | None, whitening_dim: int | None):
    cfg = load_config(config_path, args=None)

    model_cfg = dataclasses.asdict(cfg.model)
    model_cfg["whitening"] = True
    if whitening_dim is not None:
        model_cfg["whitening_dim"] = int(whitening_dim)
    model_cfg["weights"] = None

    model = create_encoder(model_cfg)

    if weights_path:
        state = load_checkpoint_state(weights_path)
        missing = model.load_state_dict(state, strict=False)
        print(f"Loaded weights from: {weights_path}")
        print(f"  missing keys: {len(getattr(missing, 'missing_keys', []))}")
        print(f"  unexpected keys: {len(getattr(missing, 'unexpected_keys', []))}")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble SupScene model with whitening")
    p.add_argument("--config", required=True, help="SupScene config path")
    p.add_argument(
        "--weights",
        default=None,
        help="Optional base model checkpoint: .pth/.ckpt/.safetensors or accelerate checkpoint directory",
    )
    p.add_argument("--features", required=True, help="Feature file (.npy/.pt/.h5)")
    p.add_argument("--whitening-dim", type=int, default=None, help="Override whitening dimension")
    p.add_argument(
        "--out",
        default="weights/dinov2_scpp_supscene_1536.pth",
        help="Output .pth path (default: weights/dinov2_scpp_supscene_1536.pth)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(args.config, args.weights, args.whitening_dim)
    feats = load_features(args.features)

    with torch.no_grad():
        model.init_whitening(feats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)

    print(f"Features shape: {tuple(feats.shape)}")
    print(f"Saved assembled model to: {out}")


if __name__ == "__main__":
    main()
