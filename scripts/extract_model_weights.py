#!/usr/bin/env python3
"""Extract model weights from SupScene checkpoints.

Supports:
- Vanilla checkpoint files saved by trainer: last.pth / best.pth (contains model_state_dict)
- Accelerate checkpoint directory: checkpoints/last/ (contains model.safetensors)
- Plain state_dict .pth files

Examples:
  python scripts/extract_model_weights.py \
      --ckpt experiments/scpp/checkpoints/last.pth \
      --out weights/supscene_model.pth

  python scripts/extract_model_weights.py \
      --ckpt experiments/scpp/checkpoints/last \
      --out weights/supscene_model_from_accelerate.pth

  python scripts/extract_model_weights.py \
      --ckpt experiments/scpp/checkpoints/last.pth \
      --prefix backbone. \
      --out weights/supscene_backbone.pth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors is required to read accelerate checkpoint directories. "
            "Install with: pip install safetensors"
        ) from exc
    return load_file(str(path), device="cpu")


def load_checkpoint_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    p = Path(ckpt_path)

    if not p.exists() and p.suffix == ".pth":
        alt_dir = p.with_suffix("")
        if alt_dir.exists() and alt_dir.is_dir():
            print(f"[info] '{p}' not found, use accelerate checkpoint dir: '{alt_dir}'")
            p = alt_dir

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if p.is_dir():
        st_path = p / "model.safetensors"
        if st_path.exists():
            return _load_safetensors(st_path)
        shard_path = p / "model_1.safetensors"
        if shard_path.exists():
            return _load_safetensors(shard_path)
        raise RuntimeError(f"Unsupported checkpoint directory: {p}")

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

    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")


def extract_with_prefix(
    state: Dict[str, torch.Tensor],
    prefix: Optional[str],
    keep_prefix: bool = False,
) -> Dict[str, torch.Tensor]:
    if not prefix:
        return dict(state)

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            nk = k if keep_prefix else k[len(prefix):]
            out[nk] = v

    if not out:
        examples = list(state.keys())[:10]
        raise RuntimeError(f"No keys matched prefix '{prefix}'. Example keys: {examples}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract model/submodule weights from SupScene checkpoints")
    p.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path (.pth/.ckpt/.safetensors) or accelerate checkpoint directory",
    )
    p.add_argument("--out", required=True, help="Output .pth file path")
    p.add_argument("--prefix", default=None, help="Optional prefix to filter keys, e.g. 'backbone.'")
    p.add_argument("--keep-prefix", action="store_true", help="Keep prefix in saved keys")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    state = load_checkpoint_state(args.ckpt)
    print(f"Loaded params: {len(state)}")

    sub_state = extract_with_prefix(state, args.prefix, keep_prefix=args.keep_prefix)
    print(f"Extracted params: {len(sub_state)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sub_state, out)
    print(f"Saved weights to: {out}")


if __name__ == "__main__":
    main()
