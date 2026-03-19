"""
SupSceneTrainer - SupScene task trainer.
Inheriting BaseTrainer, TaskManager is integrated for multi-task training
"""
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from utils import BaseTrainer
from utils import compute_batch_retrieval_metrics
from .taskmanager import TaskManager
from .eval import EvalConfig, SupSceneEvaluator

# Optional ConFIG (conflict‑free) imports
try:
    from conflictfree.grad_operator import ConFIG_update  
    from conflictfree.utils import get_gradient_vector, apply_gradient_vector  
    CONFIG_AVAILABLE = True
except ImportError:
    print("[Warning] ConFIG not available, falling back to standard multi‑task training")
    CONFIG_AVAILABLE = False


class SupSceneTrainer(BaseTrainer):
    """SupScene trainer.

    Flow per step:
        images → model → z ∈ ℝ^{B×N×D} → task heads → losses/metrics

    Args:
        model (nn.Module): Encoder producing per‑node embeddings.
        optimizer: Torch optimizer.
        task_manager (TaskManager): Provides task heads & loss aggregation.
        root_dir (str): Dataset root for evaluation.
        val_split (str): Split file for evaluation.
        metric_pos_th (float): Positive threshold for retrieval metrics.
        metric_ks (tuple): Recall@K list.
        use_conflictfree (bool): Enable ConFIG if available.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        task_manager: TaskManager,
        root_dir: str,
        val_split: str,
        metric_pos_th: float = 0.3,
        metric_ks: tuple = (1, 5, 10),
        use_conflictfree: bool = False,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.val_split = val_split
        self.metric_pos_th: float = metric_pos_th
        self.metric_ks: tuple = metric_ks
        super().__init__(model, optimizer, **kwargs)
        
        self.task_manager = task_manager
        self.use_conflictfree = use_conflictfree and CONFIG_AVAILABLE
        
        if self.use_accelerate and self.accelerator is not None:
            self.task_manager.heads = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.task_manager.heads)
            self.task_manager = self.task_manager.prepare_with_accelerator(self.accelerator)
        else:
            self.task_manager.heads = self.task_manager.heads.to(self.device)
        
        self.logger.info(f"[SupSceneTrainer] initialized with:")
        self.logger.info(f"  ConFIG conflict free: {self.use_conflictfree}")
        self.logger.info(f"  metric pos threshold: {self.metric_pos_th}")
        self.logger.info(f"  metric ks: {self.metric_ks}")
    # -------------------------
    # 1) data wrappers (B,N,...) ↔ (B*N,...)
    # -------------------------
    def _data_wrapper(self, batch: Dict[str, Any]) -> Tuple[int, int, Optional[torch.Tensor], torch.Tensor]:
        """Flatten (B,N,3,H,W) to valid nodes only.

        Args:
            batch: Input dict with `images` and optional `node_mask`.

        Returns:
            Tuple: (B, N, mask, x)
                B (int), N (int), mask (BoolTensor[B*N] or None), x (FloatTensor[num_valid,3,H,W])
        """
        images = batch["images"]  # (B,N,3,H,W) or (B,3,H,W)
        if images.dim() == 4:
            images = images.unsqueeze(1)
        B, N = images.shape[:2]

        node_mask = batch.get("node_mask")
        if node_mask is not None:
            mask = node_mask.view(-1).bool()
            x = images.view(B * N, *images.shape[2:])[mask]
            if x.numel() == 0:
                raise RuntimeError("No valid nodes found in node_mask.")
        else:
            mask = None
            x = images.view(B * N, *images.shape[2:])
        return B, N, mask, x
    
    def _data_unwrapper(self, B: int, N: int, mask: Optional[torch.Tensor], z_valid: torch.Tensor) -> torch.Tensor:
        """Restore (B*N,...) back to (B,N,...).

        Args:
            B: Batch size.
            N: Nodes per sample.
            mask: Bool mask on flattened nodes or None.
            z_valid: Embeddings for valid nodes.

        Returns:
            Tensor: z of shape (B, N, ...)
        """
        if mask is None:
            return z_valid.view(B, N, *z_valid.shape[1:])
        
        output_shape = (B * N,) + z_valid.shape[1:]
        z_flat = torch.zeros(output_shape, device=z_valid.device, dtype=z_valid.dtype)
        z_flat[mask] = z_valid
        z = z_flat.view(B, N, *z_valid.shape[1:])
        return z
    
    # -------------------------
    # 2) forward: images/z → heads
    # -------------------------
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward one batch.

        Args:
            batch: Dict with fields `images`, optional `teacher`.

        Returns:
            Dict with keys:
                - "z": (B,N,D)
                - "task_outputs": {name: tensor}
        """
        B, N, mask, x = self._data_wrapper(batch)
        z_valid = self.model(x)  # (num_valid, D)
        z = self._data_unwrapper(B, N, mask, z_valid)  # (B,N,D)
        task_outs = self.task_manager.forward_heads(z)
        return {"z": z, "task_outputs": task_outs}

    # -------------------------
    # 3) loss: multi‑task weighted sum via TaskManager
    # -------------------------
    def compute_loss(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Assemble labels & delegate to TaskManager.

        Expected batch fields:
            overlap (B,N,N),
            pair_mask (B,N,N) optional,
            node_mask (B,N) optional,
            teacher (B,N,D) optional.
        """
        outs = model_outputs["task_outputs"]
        overlap = batch["overlap"]
        pair_mask = batch.get("pair_mask")
        node_mask = batch.get("node_mask")
        teacher_features = batch.get("teacher")

        if self.use_ema and teacher_features is None:
            B, N, mask, x = self._data_wrapper(batch)
            z_flat = self.get_ema_features(x)
            teacher_features = self._data_unwrapper(B, N, mask, z_flat)

        losses = self.task_manager.compute_loss(
            outputs=outs,
            overlap=overlap,
            pair_mask=pair_mask,
            node_mask=node_mask,
            teacher_features=teacher_features,
            accelerator=self.accelerator if self.use_accelerate else None,
        )
        return losses

    # -------------------------
    # 4) metrics: batch retrieval metrics (vectorized)
    # -------------------------
    def compute_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Compute retrieval metrics on the batch using cosine(z,z)."""
        z = model_outputs["z"]  # (B,N,D)
        overlap = batch["overlap"]
        node_mask = batch.get("node_mask")
        try:
            m = compute_batch_retrieval_metrics(
                emb=z,
                overlap_gt=overlap,
                node_mask=node_mask,
                ks=self.metric_ks,
                pos_th=self.metric_pos_th,
            )
            return m
        except Exception as e:
            self.logger.error(f"[SupSceneTrainer] metric computation failed: {e}")
            return {}
        
    # -------------------------
    # 5) training / validation steps
    # -------------------------
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One training step."""
        self.model.train()
        self.task_manager.heads.train()

        outputs = self.forward_step(batch)
        loss_dict = self.compute_loss(batch, outputs)

        if self.use_conflictfree and CONFIG_AVAILABLE:
            self._conflictfree_backward_step(loss_dict)
        else:
            self._standard_backward_step(loss_dict)

        if self.scheduler is not None:
            self.scheduler.step()

        if self.use_ema:
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            self.update_ema()
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        metrics = self.compute_metrics(batch, outputs)

        result = {f"loss/{k}": (v.item() if isinstance(v, torch.Tensor) else float(v)) 
                  for k, v in loss_dict.items()}
        result.update({f"metric/{k}": float(v) for k, v in metrics.items()})

        torch.cuda.empty_cache()
        del outputs, loss_dict, metrics, batch
        return result
    
    @torch.no_grad()
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One validation step."""
        self.model.eval()
        self.task_manager.heads.eval()

        outputs = self.forward_step(batch)
        loss_dict = self.compute_loss(batch, outputs)
        metrics = self.compute_metrics(batch, outputs)

        result = {f"loss/{k}": (v.item() if isinstance(v, torch.Tensor) else float(v)) 
                  for k, v in loss_dict.items()}
        result.update({f"metric/{k}": float(v) for k, v in metrics.items()})

        torch.cuda.empty_cache()
        del outputs, loss_dict, metrics, batch
        return result
    
    # -------------------------
    # 6) backprop steps: standard or ConFIG
    # -------------------------
    def _standard_backward_step(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Standard backward with optional grad accumulation/clip (keeps behavior)."""
        total = loss_dict["total_loss"]
        if self.use_accelerate:
            self.accelerator.backward(total)
            if self.accelerator.sync_gradients and self.grad_clip is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.accelerator.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
        else:
            if self.grad_accum_steps > 1:
                total = total / self.grad_accum_steps
            total.backward()
            if (self.global_step) % self.grad_accum_steps == 0 and self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def _conflictfree_backward_step(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """ConFIG conflict‑free backward across tasks.

        Collect per‑task grads → ConFIG update → apply unified grad.
        """
        # unwrap models for PEFT/accelerate compatibility
        if self.use_accelerate:
            model = self.accelerator.unwrap_model(self.model)
            heads = self.accelerator.unwrap_model(self.task_manager.heads)
        else:
            model = self.model
            heads = self.task_manager.heads

        class _Combined(nn.Module):
            def __init__(self, m, h):
                super().__init__()
                self.model = m
                self.heads = h

        combined = _Combined(model, heads)

        # gather scalar task losses (exclude total)
        task_losses = {}
        for k, v in loss_dict.items():
            if isinstance(k, str) and k.endswith("_loss") and k.lower() != "total_loss" and isinstance(v, torch.Tensor) and v.dim() == 0:
                task_losses[k] = v

        grads = []
        names = list(task_losses.keys())
        for i, name in enumerate(names):
            loss_i = task_losses[name]
            self.optimizer.zero_grad()
            retain = i < len(names) - 1
            if self.use_accelerate:
                self.accelerator.backward(loss_i, retain_graph=retain)
            else:
                loss_i.backward(retain_graph=retain)
            g_i = get_gradient_vector(combined, none_grad_mode="zero")
            grads.append(g_i)

        g = grads[0] if len(grads) == 1 else ConFIG_update(grads)
        if g is not None:
            if not self.use_accelerate and self.grad_accum_steps > 1:
                g = g / float(self.grad_accum_steps)
            self.optimizer.zero_grad()
            apply_gradient_vector(combined, g, none_grad_mode="zero")
            if self.grad_clip is not None:
                if self.use_accelerate and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.accelerator.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def set_conflictfree(self, use_conflictfree: bool):
        self.use_conflictfree = use_conflictfree and CONFIG_AVAILABLE
        self.logger.info(f"[SupSceneTrainer] ConFIG conflict free: {self.use_conflictfree}")
    
    def get_conflictfree_status(self) -> bool:
        return self.use_conflictfree
    
    # -------------------------
    # 7) checkpoint I/O (TaskManager aware)
    # -------------------------
    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None, suffix: str = "") -> None:
        """Save checkpoint; includes TaskManager state.

        Args:
            is_best: Save best model separately.
            epoch: Overwrite current epoch (e.g., on interruption).
            suffix: Optional tag for snapshot name.
        """
        if epoch is not None:
            self.current_epoch = epoch
            if suffix == "interrupted":
                self.current_epoch -= 1

        if self.use_accelerate:
            extra = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "train_metrics": dict(self.train_metrics),
                "val_metrics": dict(self.val_metrics),
            }
            # store wandb run id if available
            if self.log_with == "wandb" and hasattr(self.accelerator, "trackers"):
                for tr in self.accelerator.trackers:
                    if hasattr(tr, "run") and hasattr(tr.run, "id"):
                        extra["wandb_run_id"] = tr.run.id
                        break
            if self.use_ema and self.ema is not None:
                extra["ema_state_dict"] = self.ema.state_dict()

            ckpt_dir = Path("checkpoints") / ("last" if not suffix else f"last_{suffix}")
            outdir = self.output_dir / ckpt_dir
            self.accelerator.save_state(output_dir=str(outdir), safe_serialization=True)

            if self.accelerator.is_main_process:
                torch.save(extra, outdir / "extra_state.pth")

            if is_best:
                self.accelerator.save_model(self.model, save_directory=str(self.output_dir / "best"), safe_serialization=True)
            self.logger.info(f"Checkpoint saved at epoch {self.current_epoch}")
            return

        # Non‑accelerate path
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "train_metrics": dict(self.train_metrics),
            "val_metrics": dict(self.val_metrics),
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if self.use_ema and self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()
        if self.task_manager is not None:
            ckpt["task_heads"] = self.task_manager.state_dict()

        name = f"last_{suffix}.pth" if suffix else "last.pth"
        torch.save(ckpt, self.output_dir / name)

        if is_best:
            best = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model": self.model.state_dict(),
                "best_metric": self.best_metric,
                "train_metrics": dict(self.train_metrics),
                "val_metrics": dict(self.val_metrics),
            }
            torch.save(best, self.output_dir / "best.pth")
        self.logger.info(f"Checkpoint saved at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint; restores TaskManager when available.

        Returns:
            int: Next epoch to run.
        """
        p = Path(checkpoint_path)
        if self.use_accelerate:
            if p.is_dir() and (p / "model.safetensors").exists():
                self.accelerator.load_state(str(p))
                extra_p = p / "extra_state.pth"
                if extra_p.exists():
                    extra = torch.load(extra_p, map_location=self.device)
                    if self.use_ema and self.ema is not None and "ema_state_dict" in extra:
                        self.ema.load_state_dict(extra["ema_state_dict"])  # noqa: E501
                        self.logger.info("EMA state loaded from checkpoint")
                    from collections import defaultdict
                    self.current_epoch = extra.get("epoch", 1) + 1
                    self.global_step = extra.get("global_step", 1)
                    self.best_metric = extra.get("best_metric")
                    self.train_metrics = defaultdict(list, extra.get("train_metrics", {}))
                    self.val_metrics = defaultdict(list, extra.get("val_metrics", {}))
                else:
                    self.logger.warning("Extra state file not found; using defaults")
            else:
                raise ValueError(f"Unsupported accelerate checkpoint: {p}")
        else:
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            ckpt = torch.load(p, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scheduler is not None and "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if self.use_ema and self.ema is not None and "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
                self.logger.info("EMA state loaded from checkpoint")
            if self.task_manager is not None and "task_heads" in ckpt:
                self.task_manager.load_state_dict(ckpt["task_heads"])
                self.logger.info("TaskManager state loaded from checkpoint")
            self.current_epoch = ckpt.get("epoch", 1) + 1
            self.global_step = ckpt.get("global_step", 1)
            self.best_metric = ckpt.get("best_metric")
            from collections import defaultdict
            self.train_metrics = defaultdict(list, ckpt.get("train_metrics", {}))
            self.val_metrics = defaultdict(list, ckpt.get("val_metrics", {}))

        self.logger.info(f"Checkpoint loaded from {p}")
        self.logger.info(f"Resuming at epoch {self.current_epoch}, step {self.global_step}")
        return self.current_epoch
    
    # -------------------------
    # 8) evaluation
    # -------------------------
    def evaluate(self, epoch: int) -> None:
        """Run full evaluation on GL3D split."""
        cfg = EvalConfig(
            root_dir=self.root_dir,
            split_txt=self.val_split,
            img_size=322,
            batch_size=64,
            num_workers=4,
            device=self.device,
            pos_th=self.metric_pos_th,
            ks=self.metric_ks,
            save_embeds=False,
            global_retrieval=True,
            use_accelerate=self.use_accelerate,
            accelerator=self.accelerator,
        )
        self.model.eval()
        use_ema_for_eval = False  # switchable flag
        if self.use_ema and use_ema_for_eval:
            with self.get_ema_model_for_inference():
                evaluator = SupSceneEvaluator(self.model, cfg)
        else:
            evaluator = SupSceneEvaluator(self.model, cfg)

        self.logger.info(f"#scenes: {len(evaluator.scene_dirs)}")
        self.logger.info("Starting full evaluation…")
        out_json = self.output_dir / "eval_metrics" / f"epoch-{epoch}.json"
        result = evaluator.run(dump_json=str(out_json))

        self.logger.info("\n✅ Evaluation done!")
        self.logger.info(f"  - elapsed: {result.get('elapsed_sec', 'N/A')} s")
        self.logger.info(f"  - total images: {result.get('total_images', 'N/A')}")
        self.logger.info(f"  - #scenes: {result.get('num_scenes', 'N/A')}")
        self.logger.info(f"  - valid queries: {result.get('num_valid_queries', 'N/A')}")

        eval_metrics = {"epoch": epoch}
        if "macro" in result:
            for k, v in result["macro"].items():
                if ("map" in k.lower()) or ("recall" in k.lower()):
                    eval_metrics[f"macro_{k}"] = v
        if "micro" in result:
            for k, v in result["micro"].items():
                if ("map" in k.lower()) or ("recall" in k.lower()):
                    eval_metrics[f"micro_{k}"] = v
        if "global" in result:
            for k, v in result["global"].items():
                if ("map" in k.lower()) or ("recall" in k.lower()):
                    eval_metrics[f"global_{k}"] = v
        self._log_metrics(eval_metrics, self.global_step, "eval")
    
    # -------------------------
    # 9) optional visualization hooks
    # -------------------------
    def _extract_data(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features & assignment maps for visualization.

        Returns empty dict if aggregator type unsupported.
        """
        actual_model = self.model.module if hasattr(self.model, "module") else self.model
        agg = getattr(actual_model, "aggregator", None)
        if agg is None:
            self.logger.warning("Model has no `aggregator` attribute")
            return {}

        agg_name = type(agg).__name__
        supported = {
            "AdaptiveGeMPool",
            "GeMPool",
            "NetVLAD",
            "SCPP",
        }
        if agg_name not in supported:
            self.logger.warning(f"Unsupported aggregator for viz: {agg_name}")
            return {}
        
        try:
            actual_model.eval()
            with torch.no_grad():
                B, N, mask, x = self._data_wrapper(batch)
                feats = actual_model.backbone(x)

                g, A, *_ = actual_model.aggregator(feats, return_maps=True)
                A = self._data_unwrapper(B, N, mask, A)

                imgs = batch["images"]
                if imgs.dim() == 4:
                    imgs = imgs.unsqueeze(1)
                imgs = imgs[:, 0].detach().cpu()  # (B,3,H,W)

                # De‑normalize (ImageNet) for logging
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                imgs = torch.clamp(imgs * std + mean, 0, 1)

                imgs = (imgs * 255).byte().numpy()  # (B,3,H,W) uint8
                A_np = A[:, 0].detach().cpu().numpy()  # (B,K,H,W) take N=0
                return {
                    "attention_maps": A_np,
                    "original_images": imgs,
                    "epoch": self.current_epoch,
                    "K": A_np.shape[1],
                }
                
        except Exception as e:
            self.logger.error(f"Visualization data extraction failed: {e}")
            return {}
    
    def visualize_model_components(self, epoch: int, batch: Dict[str, Any], prefix: str = "") -> None:
        """Visualize assignment/activation maps via logger backends."""
        try:
            data = self._extract_data(batch)
            if not data:
                return
            from supscene.viz_utils.viz_assign_map import create_overlay_batch 
            heat = create_overlay_batch(
                data["original_images"], data["attention_maps"], alpha=0.5, colormap_name="magma"
            )
            if heat is not None:
                vis = {f"heatmap_{i}": heat[i] for i in range(heat.shape[0])}
                self._log_image(vis, self.global_step, prefix)
                self.logger.info(f"Visualization logged — epoch {epoch}")
        except Exception as e:
            self.logger.warning(f"Visualization error: {e}")

