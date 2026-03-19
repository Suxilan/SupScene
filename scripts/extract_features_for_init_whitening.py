#!/usr/bin/env python3
"""Extract global features for whitening initialization.

This script scans images recursively, runs SupScene encoder forward,
and saves extracted features to .npy (and optional path list .txt).

Examples:
  python scripts/extract_features_for_init_whitening.py \
      --config configs/peft-dinov2-scpp-lora.yaml \
      --weights experiments/peft-dinov2-scpp-lora/checkpoints/last.pth \
      --roots data/GL3D/train \
      --batch-size 64 \
      --img-size 322 \
      --out-npy cache/features_for_whitening.npy
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T2
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.conf import load_config
from supscene import create_encoder

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(roots: List[Path]) -> List[Path]:
    imgs: List[Path] = []
    for r in roots:
        if not r.exists():
            print(f"[warn] root not found, skip: {r}")
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                imgs.append(p)
    imgs.sort()
    return imgs


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path], tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        return self.tf(im), idx


def load_model(config_path: str, weights: str | None, device: torch.device):
    cfg = load_config(config_path, args=None)

    model_cfg = dataclasses.asdict(cfg.model)
    model_cfg["whitening"] = False
    model_cfg["final_norm"] = False
    model_cfg["weights"] = None
    model = create_encoder(model_cfg)

    if weights:
        state = load_checkpoint_state(weights)
        missing = model.load_state_dict(state, strict=False)
        print(f"Loaded weights from: {weights}")
        print(f"  missing keys: {len(getattr(missing, 'missing_keys', []))}")
        print(f"  unexpected keys: {len(getattr(missing, 'unexpected_keys', []))}")

    model = model.to(device).eval()
    return model


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

    # Common typo/legacy path: checkpoints/last.pth while actual artifact is checkpoints/last/
    if not p.exists() and p.suffix == ".pth":
        alt_dir = p.with_suffix("")
        if alt_dir.exists() and alt_dir.is_dir():
            print(f"[info] '{p}' not found, use accelerate checkpoint dir: '{alt_dir}'")
            p = alt_dir

    if not p.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    # Accelerate checkpoint directory
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

    # Single safetensors file
    if p.suffix == ".safetensors":
        return _load_safetensors(p)

    # PyTorch checkpoint formats
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract features for whitening init")
    p.add_argument("--config", required=True, help="SupScene config path")
    p.add_argument(
        "--weights",
        default=None,
        help="Optional model checkpoint: .pth/.ckpt/.safetensors or accelerate checkpoint directory (e.g. .../checkpoints/last)",
    )
    p.add_argument("--roots", nargs="+", required=True, help="Image root directories")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=322)
    p.add_argument("--out-npy", required=True, help="Output features .npy path")
    p.add_argument("--out-list", default=None, help="Optional output image path list .txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    roots = [Path(x) for x in args.roots]
    paths = find_images(roots)
    if not paths:
        raise RuntimeError("No images found under provided roots.")
    print(f"Found images: {len(paths)}")

    tf = T2.Compose([
        T2.ToImage(),
        T2.Resize(size=(args.img_size, args.img_size), interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = ImageDataset(paths, tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args.config, args.weights, device)

    feat_chunks: List[torch.Tensor] = []
    idx_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for images, idx in tqdm(dl, desc="Extracting"):
            images = images.to(device, non_blocking=True)
            feats = model(images).float().cpu()
            feat_chunks.append(feats)
            idx_chunks.append(idx.cpu())

    feats_all = torch.cat(feat_chunks, dim=0)
    idx_all = torch.cat(idx_chunks, dim=0)
    order = torch.argsort(idx_all)
    feats_all = feats_all[order].numpy()

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, feats_all)
    print(f"Saved features to: {out_npy} (shape={feats_all.shape})")

    if args.out_list:
        out_list = Path(args.out_list)
        out_list.parent.mkdir(parents=True, exist_ok=True)
        with out_list.open("w", encoding="utf-8") as f:
            for p in paths:
                f.write(str(p) + "\n")
        print(f"Saved image list to: {out_list}")


if __name__ == "__main__":
    main()
