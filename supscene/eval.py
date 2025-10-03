import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- 依赖你已有的数据工具 ---
from .datasets.gl3d_subgraph_dataset import read_lines
from .datasets.gl3d_batch_dataset import GL3DBatchDataset

# metrics (injectable; lightweight fallback handled in evaluator)
try:
    from utils.metrics import (
        compute_retrieval_metrics,
        compute_global_retrieval_metrics,
    )
except ImportError:
    compute_retrieval_metrics = None
    compute_global_retrieval_metrics = None

# accelerate (optional)
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


# =============================================================
# Evaluator: SupSceneEvaluator (accelerate‑safe)
# =============================================================
@dataclass
class EvalConfig:
    root_dir: str
    split_txt: str
    img_size: int = 322
    batch_size: int = 256
    num_workers: int = 4
    device: str = "cuda"
    ks: Tuple[int, ...] = (1, 5, 10, 20)
    pos_th: float = 0.25
    mode: str = "similarity"  # "similarity" | "distance"
    save_embeds: bool = False
    embeds_dir: Optional[str] = None
    use_amp: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    global_retrieval: bool = False
    use_accelerate: bool = True
    accelerator: Optional[Any] = None  # external accelerator (optional)

class SupSceneEvaluator:
    """Full‑split evaluator for GL3D.

    Args:
        encoder: nn.Module that maps (B,3,H,W) → (B,D), L2 optional.
        cfg: EvalConfig.
        metrics_fn: per‑scene metrics function (emb, overlap, ks, mask=None, pos_th, mode, node_mask).
    """
    def __init__(
        self,
        encoder: nn.Module,
        cfg: EvalConfig,
        metrics_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.cfg = cfg
        self.metrics_fn = metrics_fn or compute_retrieval_metrics
        if self.metrics_fn is None:
            raise RuntimeError("compute_retrieval_metrics not found; inject via `metrics_fn=`.")
        
        # accelerator / device
        self.accelerator = None
        self.is_main_process = True
        if cfg.use_accelerate and ACCELERATE_AVAILABLE:
            if cfg.accelerator is not None:
                self.accelerator = cfg.accelerator
                self.encoder = encoder.eval()
            else:
                self.accelerator = Accelerator()
                self.encoder = self.accelerator.prepare(encoder).eval()
            self.is_main_process = self.accelerator.is_main_process
            self.device = self.accelerator.device
        else:
            self.encoder = encoder.to(cfg.device).eval()
            self.device = torch.device(cfg.device)

        # split
        sids = read_lines(cfg.split_txt)
        self.scene_dirs = [os.path.join(cfg.root_dir, "GL3D", sid) for sid in sids]
        self.scene_ids = [os.path.basename(p) for p in self.scene_dirs]

        # amp dtype
        self.amp_dtype = torch.float16 if cfg.use_amp and ("cuda" in str(self.device)) else None

        # embed save dir
        self.embeds_dir = None
        if cfg.save_embeds and self.is_main_process:
            self.embeds_dir = cfg.embeds_dir or os.path.join(cfg.root_dir, "embeds")
            os.makedirs(self.embeds_dir, exist_ok=True)

    @torch.no_grad()
    def _embed_all(self) -> Tuple[torch.Tensor, List[torch.Tensor], List[str], List[int]]:
        """Extract embeddings for all images once, preserving global order.

        Returns:
            all_emb: (T,D)
            scene_O: list of O tensors per scene
            scene_ids: list of scene ids
            scene_offsets: starting index for each scene in all_emb
        """
        ds = GL3DBatchDataset(self.scene_dirs, img_size=self.cfg.img_size, return_index=True)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
        )
        if self.accelerator is not None:
            dl = self.accelerator.prepare(dl)

        all_emb_chunks: List[torch.Tensor] = []
        all_idx_chunks: List[torch.Tensor] = []
        if self.is_main_process:
            print(f"[Eval] Extracting features for {len(ds)} images…")

        for batch in dl:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, idx = batch
            else:  # backward compatibility (dataset returning only images)
                x, idx = batch, None
            x = x.to(self.device, non_blocking=True)

            if self.amp_dtype is None:
                z = self.encoder(x)
            else:
                # device type string: "cuda" or "cpu"
                dev_type = str(self.device).split(":")[0]
                with torch.amp.autocast(dev_type, dtype=self.amp_dtype):
                    z = self.encoder(x)
            z = z.float()

            if self.accelerator is not None:
                z = self.accelerator.gather_for_metrics(z)
                if idx is not None:
                    idx = idx.to(z.device)
                    idx = self.accelerator.gather_for_metrics(idx)
            all_emb_chunks.append(z)
            if idx is not None:
                all_idx_chunks.append(idx)

        all_emb = torch.cat(all_emb_chunks, dim=0)  # (T,D)

        # order recovery when using accelerate
        if all_idx_chunks:
            all_idx = torch.cat(all_idx_chunks, dim=0).long()
            order = torch.argsort(all_idx)
            all_emb = all_emb[order]

        # scene meta
        scene_O: List[torch.Tensor] = []
        scene_ids: List[str] = []
        scene_offsets: List[int] = []
        cur = 0
        for sid, s, e, O in ds.get_scene_info():
            scene_ids.append(sid)
            scene_offsets.append(cur)
            scene_O.append(torch.from_numpy(O).float())
            cur += (e - s)
            # optional save
            if self.embeds_dir is not None:
                np.save(os.path.join(self.embeds_dir, f"{sid}.npy"), all_emb[s:e].cpu().numpy())
        return all_emb, scene_O, scene_ids, scene_offsets


    @torch.no_grad()
    def evaluate_scene(self, scene_dir: str) -> Dict[str, Any]:
        """Eval a single scene (standalone path)."""
        # Build dataset for one scene
        G = SceneGraph(scene_dir)
        paths = G.image_paths
        tf = A.Compose([
            A.Resize(self.cfg.img_size, self.cfg.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        import cv2
        X = []
        for p in paths:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            X.append(tf(image=im)["image"])  # (3,H,W)
        X = torch.stack(X, dim=0).to(self.device)
        dev_type = str(self.device).split(":")[0]
        with torch.amp.autocast(dev_type, enabled=self.amp_dtype is not None, dtype=self.amp_dtype or torch.float32):
            z = self.encoder(X).float()
        O = G.dense_overlap(np.arange(G.N), add_self=True)

        node_mask = torch.ones(z.size(0), dtype=torch.bool)
        metrics = self.metrics_fn(
            emb=z,
            overlap=torch.from_numpy(O),
            ks=self.cfg.ks,
            mask=None,
            pos_th=self.cfg.pos_th,
            mode=self.cfg.mode,
            node_mask=node_mask,
        )
        Y = (torch.from_numpy(O) >= self.cfg.pos_th)
        torch.diagonal(Y).fill_(False)
        num_valid = (Y.sum(dim=-1) > 0).sum().item()
        return {"scene_id": os.path.basename(scene_dir), 
                "N": z.size(0), 
                "num_valid_queries": num_valid, 
                "metrics": metrics}

    @torch.no_grad()
    def evaluate_global_efficient(
        self,
        all_emb: torch.Tensor,
        scene_O: List[torch.Tensor],
        scene_offsets: List[int]) -> Dict[str, Any]:
        """
        Global retrieval metrics across all images.
        """
        if compute_global_retrieval_metrics is None:
            raise RuntimeError("cannot find compute_global_retrieval_metrics, please check utils.global_retrieval module")
        
        gm = compute_global_retrieval_metrics(
            all_emb=all_emb,
            scene_overlaps=scene_O,
            scene_offsets=scene_offsets,
            ks=self.cfg.ks,
            pos_th=self.cfg.pos_th,
            mode=self.cfg.mode,
        )
        
        num_valid = 0
        for O in scene_O:
            Y = (O >= self.cfg.pos_th)
            Y.fill_diagonal_(False)
            num_valid += int((Y.sum(dim=-1) > 0).sum().item())
        return {
            "global_metrics": gm,
            "total_images": int(all_emb.size(0)),
            "num_scenes": len(scene_O),
            "num_valid_queries": num_valid,
        }

    @torch.no_grad()
    def _per_scene_eval(
        self, all_emb: torch.Tensor, 
        scene_O: List[torch.Tensor], 
        scene_ids: List[str], 
        scene_offsets: List[int]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, (sid, O) in enumerate(zip(scene_ids, scene_O)):
            s = scene_offsets[i]
            e = scene_offsets[i + 1] if i + 1 < len(scene_offsets) else all_emb.size(0)
            Z = all_emb[s:e]
            node_mask = torch.ones(Z.size(0), dtype=torch.bool)
            try:
                metrics = self.metrics_fn(
                    emb=Z,
                    overlap=O,
                    ks=self.cfg.ks,
                    mask=None,
                    pos_th=self.cfg.pos_th,
                    mode=self.cfg.mode,
                    node_mask=node_mask,
                )
                Y = (O >= self.cfg.pos_th)
                Y.fill_diagonal_(False)
                num_valid = int((Y.sum(dim=-1) > 0).sum().item())
                out.append({"scene_id": sid, "N": int(Z.size(0)), "num_valid_queries": num_valid, "metrics": metrics})
                if self.is_main_process:
                    recs = ", ".join([f"{k}:{metrics[k]:.4f}" for k in metrics if k.startswith("recall")])
                    print(f"[eval] ({len(out)}/{len(scene_ids)}) {sid}: N={Z.size(0)} valid={num_valid} → {recs}")
            except Exception as e:
                if self.is_main_process:
                    print(f"[eval] scene {sid} failed: {e}")
        return out

    @torch.no_grad()
    def run(self, dump_json: Optional[str] = None) -> Dict[str, Any]:
        """Run full evaluation over the split.

        Returns dict with macro/micro/global/per_scene + timings.
        """
        t0 = time.time()
        all_emb, scene_O, scene_ids, scene_offsets = self._embed_all()
        all_emb = all_emb.cpu()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        if not self.is_main_process:
            return {"skipped": True, "reason": "non‑main process"}

        # global
        global_res = self._global_eval(all_emb, scene_O, scene_offsets)
        print(
            f"[global] total_images={global_res['total_images']} scenes={global_res['num_scenes']} valid={global_res['num_valid_queries']}"
        )
        print(
            "Global metrics: "
            + ", ".join([f"{k}:{v:.4f}" for k, v in global_res["global_metrics"].items() if k.startswith("recall")])
        )

        # per‑scene
        per_scene = self._per_scene_eval(all_emb, scene_O, scene_ids, scene_offsets)
        if not per_scene:
            raise RuntimeError("No scene was successfully evaluated. Check paths or dependencies.")

        # aggregate
        keys = list(per_scene[0]["metrics"].keys())
        macro = {k: float(np.mean([s["metrics"][k] for s in per_scene])) for k in keys}
        weights = np.maximum(np.array([s["num_valid_queries"] for s in per_scene], dtype=np.float64), 1.0)
        micro = {k: float(np.average([s["metrics"][k] for s in per_scene], weights=weights)) for k in keys}

        out = {
            "macro": macro,
            "micro": micro,
            "global": global_res["global_metrics"],
            "total_images": global_res["total_images"],
            "num_scenes": global_res["num_scenes"],
            "num_valid_queries": global_res["num_valid_queries"],
            "per_scene": per_scene,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        if dump_json:
            os.makedirs(os.path.dirname(dump_json), exist_ok=True)
            with open(dump_json, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        return out

# =============================================================
# Minimal CLI (optional)
# =============================================================

# def _load_encoder_from_cfg(orbit_cfg_path: str) -> nn.Module:
#     """Example loader; adapt to your project (create_encoder / create_orbit)."""
#     import yaml
#     from .encoder import create_encoder  # or from .encoder import create_encoder
#     with open(orbit_cfg_path, "r", encoding="utf-8") as f:
#         ocfg = yaml.safe_load(f)
#     enc = create_encoder(ocfg)
#     return enc


# def main_cli():  # pragma: no cover
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--root_dir", type=str, required=True)
#     p.add_argument("--split_txt", type=str, required=True)
#     p.add_argument("--orbit_cfg", type=str, required=True)
#     p.add_argument("--img_size", type=int, default=322)
#     p.add_argument("--batch_size", type=int, default=256)
#     p.add_argument("--num_workers", type=int, default=4)
#     p.add_argument("--device", type=str, default="cuda")
#     p.add_argument("--pos_th", type=float, default=0.3)
#     p.add_argument("--ks", type=str, default="1,5,10,20")
#     p.add_argument("--save_embeds", action="store_true")
#     p.add_argument("--embeds_dir", type=str, default=None)
#     p.add_argument("--use_amp", action="store_true")
#     p.add_argument("--dump_json", type=str, default=None)
#     args = p.parse_args()

#     ks = tuple(int(x) for x in args.ks.split(","))
#     enc = _load_encoder_from_cfg(args.orbit_cfg)

#     cfg = EvalConfig(
#         root_dir=args.root_dir,
#         split_txt=args.split_txt,
#         img_size=args.img_size,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         device=args.device,
#         pos_th=args.pos_th,
#         ks=ks,
#         save_embeds=args.save_embeds,
#         embeds_dir=args.embeds_dir,
#         use_amp=args.use_amp,
#     )

#     eva = SupSceneEvaluator(enc, cfg)
#     out = eva.run(dump_json=args.dump_json)
#     print("\n===== EVAL SUMMARY =====")
#     print(f"Elapsed: {out['elapsed_sec']}s")
#     print("Macro:", json.dumps(out["macro"], indent=2))
#     print("Micro:", json.dumps(out["micro"], indent=2))


# if __name__ == "__main__":  # pragma: no cover
#     main_cli()
