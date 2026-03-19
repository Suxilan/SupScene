#!/usr/bin/env python3
"""Evaluate a plain pretrained SupScene state_dict weight.

This is for released weights like:
  weights/dinov2_scpp_supscene_1536.pth

Example:
  python scripts/eval_pretrained_weight.py \
    --config configs/peft-dinov2-scpp-lora.yaml \
    --weights weights/dinov2_scpp_supscene_1536.pth \
    --whitening-dim 1536 \
    --out experiments/eval_dinov2_scpp_supscene_1536.json
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import torch

from engine.conf import load_config
from supscene.encoder import create_encoder
from supscene.eval import SupSceneEvaluator, EvalConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate released SupScene pretrained weight")
    p.add_argument("--config", required=True, help="Config yaml path")
    p.add_argument("--weights", required=True, help="Pretrained state_dict .pth")
    p.add_argument("--whitening-dim", type=int, default=1536, help="Whitening dim of released model")
    p.add_argument("--batch-size", type=int, default=64, help="Eval batch size")
    p.add_argument("--out", default="experiments/eval_dinov2_scpp_supscene_1536.json", help="Output metrics JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, args=None)

    model_cfg = dataclasses.asdict(cfg.model)
    model_cfg["weights"] = None
    model_cfg["whitening"] = True
    model_cfg["whitening_dim"] = int(args.whitening_dim)

    model = create_encoder(model_cfg)
    state = torch.load(args.weights, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)

    print(f"Loaded weights: {args.weights}")
    print(f"  missing keys: {len(getattr(missing, 'missing_keys', []))}")
    print(f"  unexpected keys: {len(getattr(missing, 'unexpected_keys', []))}")

    ev_cfg = EvalConfig(
        root_dir=cfg.data.root_dir,
        split_txt=cfg.data.val_split_file,
        img_size=cfg.data.image_size,
        batch_size=args.batch_size,
        num_workers=cfg.data.num_workers,
        device=cfg.system.device,
        ks=tuple(cfg.metric.metric_ks),
        pos_th=cfg.metric.metric_pos_th,
        mode="similarity",
        use_amp=(str(cfg.system.mixed_precision).lower() in ("fp16", "bf16")),
        pin_memory=cfg.data.pin_memory,
        global_retrieval=True,
        use_accelerate=False,
    )

    out = SupSceneEvaluator(model, ev_cfg).run(dump_json=args.out)

    macro = {k: v for k, v in out["macro"].items() if k.startswith("recall") or k.startswith("map")}
    global_m = {k: v for k, v in out["global"].items() if k.startswith("recall") or k.startswith("map")}

    print("macro:", macro)
    print("global:", global_m)
    print(f"saved json: {args.out}")


if __name__ == "__main__":
    main()
