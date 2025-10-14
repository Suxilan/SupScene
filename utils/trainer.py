import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import defaultdict
from abc import ABC, abstractmethod
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .ema import ExponentialMovingAverage

try:
    from accelerate import Accelerator
    from accelerate.state import DistributedType
    from accelerate.logging import get_logger
    from accelerate.utils import tqdm
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class BaseTrainer(ABC):
    """Abstract trainer with standard train/val loops, logging, ckpt & EMA.

    Key features:
        1) Standard loops with grad accumulation/clip.
        2) Optimizer & scheduler management.
        3) Optional Accelerate (single/multi‑process) + mixed precision.
        4) Checkpoint save/load (Accelerate or vanilla).
        5) Logging: Accelerate trackers / WandB / TensorBoard.
        6) Optional EMA of parameters.

    Notes:
        - Public method names/signatures preserved; subclasses implement
          `forward_step`, `compute_loss`, `evaluate`.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: Union[str, torch.device] = "cuda",
        output_dir: Union[str, Path] = "experiments",
        
        # training
        grad_clip: Optional[float] = None,
        grad_accum_steps: int = 1,
        seed: int = 42,
        deterministic: bool = False,
        device_specific: bool = False,
        
        # logging cadence
        log_interval: int = 100,
        val_interval: int = 1,
        save_interval: int = 5,
        eval_per_epochs: int = 10,
        
        # accelerate
        use_accelerate: bool = True,
        mixed_precision: str = "no",  # "no", "fp16", "bf16"
        gradient_accumulation_steps: Optional[int] = None,
        
        # logging backends
        log_with: Optional[str] = None,  # "wandb", "tensorboard", None
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        tensorboard_log_dir: Optional[str] = None,
        resume_wandb_id: Optional[str] = None,
        
        # EMA
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_use_num_updates: bool = True,
        
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # training cfg
        self.grad_clip = grad_clip
        self.grad_accum_steps = gradient_accumulation_steps or grad_accum_steps
        self.deterministic = deterministic
        self.device_specific = device_specific
        self.seed = seed
        
        # cadence
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.eval_per_epochs = eval_per_epochs
        self.log_with = log_with
        
        # state
        self.current_epoch = 1
        self.global_step = 1
        self.best_metric = None
        self.train_metrics: DefaultDict[Any, Any] = defaultdict(list)
        self.val_metrics: DefaultDict[Any, Any] = defaultdict(list)
               
        # accelerate
        self.use_accelerate = bool(use_accelerate and ACCELERATE_AVAILABLE)
        self.accelerator: Optional[Accelerator] = None
        self.is_main_process = True
        if self.use_accelerate:  # keep True for single or multi process
            accelerate_log_with = None
            if log_with == "wandb" and WANDB_AVAILABLE:
                accelerate_log_with = "wandb"
            elif log_with == "tensorboard" and TENSORBOARD_AVAILABLE:
                accelerate_log_with = "tensorboard"
            self.accelerator = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=self.grad_accum_steps,
                log_with=accelerate_log_with,
                project_dir=str(self.output_dir),
            )
            # SyncBN only when truly multi‑process
            if self.accelerator.distributed_type in {
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_XPU,
                DistributedType.MULTI_MLU,
                DistributedType.MULTI_HPU,
                DistributedType.MULTI_MUSA,
                DistributedType.MULTI_SDAA,
                DistributedType.MULTI_NPU,
            }:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            else:
                self.use_accelerate = False  # revert to False if single process
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            if self.scheduler is not None:
                self.scheduler = self.accelerator.prepare(self.scheduler)
            self.device = self.accelerator.device
            self.is_main_process = self.accelerator.is_main_process
        else:
            self.model = self.model.to(self.device)
        
        # EMA
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_use_num_updates = bool(ema_use_num_updates)
        self.ema: Optional[ExponentialMovingAverage] = None
        if self.use_ema:
            self.ema = ExponentialMovingAverage(
                parameters=self.model.parameters(),
                decay=self.ema_decay,
                use_num_updates=self.ema_use_num_updates,
            )
        
        # logger
        self.logger = self._setup_logger()

        # external trackers
        self.wandb_run = None
        self.tensorboard_writer = None
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_config = wandb_config
        self.resume_wandb_id = resume_wandb_id
        self._init_logging_backends(wandb_project, wandb_name, wandb_config, tensorboard_log_dir, resume_wandb_id)

        # persist config (main proc only)
        self._save_config(kwargs)
        if self.use_ema:
            self.logger.info(f"[EMA] enabled: decay={self.ema_decay}, use_num_updates={self.ema_use_num_updates}")
    
    def update_ema(self) -> None:
        """Update EMA after optimizer.step()."""
        if self.ema is not None:
            self.ema.update(self.model.parameters())

    def get_ema_model_for_inference(self):  # context manager
        """Context manager that swaps in EMA weights for inference."""
        if self.ema is not None:
            if self.accelerator is not None:
                unwrapped = self.accelerator.unwrap_model(self.model)
                return self.ema.average_parameters(unwrapped.parameters())
            return self.ema.average_parameters(self.model.parameters())
        return None

    def get_ema_features(self, *args, **kwargs):
        """Forward pass using EMA params via context manager."""
        if self.ema is not None:
            with self.get_ema_model_for_inference():
                return self.model(*args, **kwargs)
        return None     
    
    @staticmethod
    def seed_worker(worker_id: int) -> None:
        """Seed dataloader workers for reproducibility."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def create_generator(seed: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g
    
    def _setup_logger(self):
        if self.use_accelerate and self.accelerator is not None:
            logger = get_logger(self.__class__.__name__, log_level="INFO")
            # ensure stream handler on underlying logger
            import logging
            ul = logger.logger  # type: ignore[attr-defined]
            if not any(isinstance(h, logging.StreamHandler) for h in ul.handlers):
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
                ul.addHandler(h)
            ul.setLevel(logging.INFO)
            logger.info("[BaseTrainer] accelerate logger ready")
            return logger
        # standard python logger
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.info("[BaseTrainer] std logger ready")
        return logger
    
    def _init_logging_backends(
        self,
        wandb_project: Optional[str],
        wandb_name: Optional[str],
        wandb_config: Optional[Dict[str, Any]],
        tensorboard_log_dir: Optional[str],
        resume_run_id: Optional[str] = None,
    ) -> None:
        """Init external trackers (main process only)."""
        if self.use_accelerate and self.accelerator and not self.accelerator.is_main_process:
            return
        # WandB
        if self.log_with == "wandb" and WANDB_AVAILABLE:
            try:
                if self.use_accelerate and self.accelerator is not None:
                    init_kwargs = {
                        "wandb": {
                            "name": wandb_name,
                            "dir": str(self.output_dir)
                        }
                    }
                    if resume_run_id:
                        init_kwargs["wandb"].update({"id": resume_run_id, "resume": "must"})
                        print(f"[WandB] resume run: {resume_run_id}")
                    self.accelerator.init_trackers(
                        project_name=wandb_project or "orbit",
                        config=wandb_config or {},
                        init_kwargs=init_kwargs,
                    )
                else:
                    init_kwargs = {
                        "project": wandb_project,
                        "name": wandb_name,
                        "config": wandb_config or {},
                        "dir": str(self.output_dir)
                    }
                    if resume_run_id:
                        init_kwargs.update({"id": resume_run_id, "resume": "must"})
                        self.logger.info(f"[WandB] resume run: {resume_run_id}")
                    self.wandb_run = wandb.init(**init_kwargs)
            except Exception as e:
                self.logger.error(f"Warning: Failed to initialize WandB: {e}")
                self.log_with = None
        
        elif self.log_with == "tensorboard" and TENSORBOARD_AVAILABLE and not self.use_accelerate:
            try:
                tb_log_dir = tensorboard_log_dir or str(self.output_dir / "tensorboard")
                self.tensorboard_writer = SummaryWriter(tb_log_dir)
            except Exception as e:
                self.logger.error(f"Warning: Failed to initialize TensorBoard: {e}")
                self.log_with = None
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        if not metrics:
            return
        if self.use_accelerate and self.accelerator and not self.accelerator.is_main_process:
            return
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if self.use_accelerate and self.accelerator is not None and self.accelerator.is_main_process:
            try:
                self.accelerator.log(metrics, step=step)
            except Exception as e:
                self.logger.error(f"accelerate.log failed: {e}")
        elif self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                self.logger.error(f"wandb.log failed: {e}")
        elif self.tensorboard_writer is not None:
            try:
                for k, v in metrics.items():
                    self.tensorboard_writer.add_scalar(k, v, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                self.logger.error(f"tensorboard log failed: {e}")
    
    def _log_image(self, images: Dict[str, np.ndarray], step: int, prefix: str = "") -> None:
        if not images:
            return
        if self.use_accelerate and self.accelerator and not self.accelerator.is_main_process:
            return
        if prefix:
            images = {f"{prefix}/{k}": v for k, v in images.items()}
        # via accelerate (WandB/TB handled by trackers)
        if self.use_accelerate and self.accelerator is not None:
            try:
                payload = {}
                for k, img in images.items():
                    if not isinstance(img, np.ndarray):
                        continue
                    if img.ndim == 4:
                        img = img[0]
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))  # CHW→HWC
                    payload[k] = wandb.Image(img) if WANDB_AVAILABLE else img  # type: ignore
                if payload:
                    self.accelerator.log(payload, step=step)
            except Exception as e:
                self.logger.error(f"image log failed (accelerate): {e}")
        elif self.wandb_run is not None and WANDB_AVAILABLE:
            try:
                payload = {}
                for k, img in images.items():
                    if img.ndim == 4:
                        img = img[0]
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))
                    payload[k] = wandb.Image(img)
                if payload:
                    self.wandb_run.log(payload, step=step)
            except Exception as e:
                self.logger.error(f"image log failed (wandb): {e}")
        elif self.tensorboard_writer is not None:
            try:
                for k, img in images.items():
                    if img.ndim == 4:
                        img = img[0]
                    if img.ndim == 3:  # CHW expected
                        self.tensorboard_writer.add_image(k, img, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                self.logger.error(f"image log failed (tb): {e}")
    
    def visualize_model_components(self, epoch: int, batch: Optional[Dict[str, Any]] = None, prefix: str = "") -> None:
        """Hook for subclasses to log images/heatmaps.
        Signature matches common subclasses that pass `batch`.
        """
        return
    
    def _save_config(self, extra_config: Dict[str, Any]) -> None:
        cfg = {
            "model_name": self.model.__class__.__name__,
            "optimizer_name": self.optimizer.__class__.__name__,
            "scheduler_name": self.scheduler.__class__.__name__ if self.scheduler else None,
            "grad_clip": self.grad_clip,
            "grad_accum_steps": self.grad_accum_steps,
            "base_lr": self.optimizer.param_groups[0]['lr'],
            "use_accelerate": self.use_accelerate,
            "seed": getattr(self, "seed", None),
            "deterministic": getattr(self, "deterministic", None),
            **extra_config,
        }
        if not self.use_accelerate or (self.accelerator and self.accelerator.is_main_process):
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2, default=str)

    #=======================================================================
    # Abstracts to implement
    #=======================================================================
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses dict; must include 'total_loss'."""
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass producing model outputs dict."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, epoch: int):
        """Full evaluation entry (dataset‑specific)."""
        raise NotImplementedError

    def compute_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Optional: return metrics dict (defaults to empty)."""
        return {}
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors in nested structures to self.device (no‑op under Accelerate)."""
        if self.use_accelerate and self.accelerator is not None:
            return batch
        def _move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move(x) for x in obj)
            return obj
        return _move(batch)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One training step: forward → loss → backward → step.
        Subclasses often override this for task‑specific behavior.
        """
        self.model.train()
        outputs = self.forward_step(batch)
        loss_dict = self.compute_loss(batch, outputs)
        total = loss_dict["total_loss"]

        if self.use_accelerate and self.accelerator is not None:
            self.accelerator.backward(total)
            if self.accelerator.sync_gradients and self.grad_clip is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        else:
            if self.grad_accum_steps > 1:
                total = total / self.grad_accum_steps
            total.backward()
            if (self.global_step) % self.grad_accum_steps == 0 and self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        
        if self.use_ema:
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            self.update_ema()
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        
        metrics = self.compute_metrics(batch, outputs)
        result = {f"loss/{k}": (v.item() if isinstance(v, torch.Tensor) else v) 
                  for k, v in loss_dict.items()}
        result.update({f"metric/{k}": v for k, v in metrics.items()})

        torch.cuda.empty_cache()
        del outputs, loss_dict, metrics, batch
        return result
    
    @torch.no_grad()
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()
        outputs = self.forward_step(batch)
        loss_dict = self.compute_loss(batch, outputs)
        metrics = self.compute_metrics(batch, outputs)
        result = {f"loss/{k}": (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        result.update({f"metric/{k}": v for k, v in metrics.items()})
        torch.cuda.empty_cache()
        del outputs, loss_dict, metrics, batch
        return result
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch; averages metrics across steps (and processes)."""
        epoch_metrics: DefaultDict[str, list] = defaultdict(list)
        self.model.train()
        t0 = time.time()
        
        bar = tqdm(
            train_loader, 
            desc=f"Train Epoch {self.current_epoch}",
            disable=self.use_accelerate and not self.accelerator.is_main_process
        )
        
        for bidx, batch in enumerate(bar):
            try:
                batch = self._move_to_device(batch)
                step = self.train_step(batch)
                for k, v in step.items():
                    epoch_metrics[k].append(v)
            except KeyboardInterrupt:
                self.logger.error("⚠️ Training interrupted by user")
                raise
            
            self.global_step += 1
            lr = self.optimizer.param_groups[0]["lr"]
            bar.set_postfix({
                'Loss': f"{step.get('loss/total_loss', 0):.4f}",
                'LR': f"{lr:.2e}"
            })
            
            if self.global_step % self.log_interval == 0:
                log = {
                    "loss/total_loss": step.get("loss/total_loss", 0),
                    "lr": lr
                }
                for k, v in step.items():
                    if k.startswith("metric/"):
                        log[k] = v
                self._log_metrics(log, self.global_step, "train_step")
        
        avg = {"epoch": self.current_epoch}
        for k, v in epoch_metrics.items():
            if self.use_accelerate and self.accelerator is not None:
                vals = torch.tensor(v, device=self.device, dtype=torch.float32)
                vals = self.accelerator.gather(vals)
                avg[k] = vals.mean().item()
            else:
                avg[k] = float(sum(v) / max(1, len(v)))

        dt = time.time() - t0
        self.logger.info(f"Train Epoch {self.current_epoch} in {dt:.2f}s")
        self._log_metrics(avg, self.global_step, "train_avg")

        try:
            self.visualize_model_components(self.current_epoch, batch, "train_viz")
        except Exception as e:
            self.logger.warning(f"viz failed: {e}")
        return avg
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        epoch_metrics: DefaultDict[str, list] = defaultdict(list)
        self.model.eval()
        t0 = time.time()
        
        bar = tqdm(
            val_loader,
            desc=f"Val Epoch {self.current_epoch}",
            disable=self.use_accelerate and not self.accelerator.is_main_process
        )
        
        for batch in bar:
            batch = self._move_to_device(batch)
            step = self.val_step(batch)
            for k, v in step.items():
                epoch_metrics[k].append(v)
            bar.set_postfix({
                'Val Loss': f"{step.get('loss/total_loss', 0):.4f}"
            })
        
        avg = {"epoch": self.current_epoch}
        for k, v in epoch_metrics.items():
            if self.use_accelerate and self.accelerator is not None:
                vals = torch.tensor(v, device=self.device, dtype=torch.float32)
                vals = self.accelerator.gather(vals)
                avg[k] = vals.mean().item()
            else:
                avg[k] = float(sum(v) / max(1, len(v)))

        dt = time.time() - t0
        self.logger.info(f"Val Epoch {self.current_epoch} in {dt:.2f}s")
        self._log_metrics(avg, self.global_step, "val_avg")

        try:
            self.visualize_model_components(self.current_epoch, batch, "val_viz")
        except Exception as e:
            self.logger.warning(f"viz failed: {e}")
        return avg
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        monitor_metric: str = "loss/total_loss",
        monitor_mode: str = "min",
    ) -> None:
        """Main training loop.

        Args:
            train_loader: training set loader.
            val_loader: optional val loader.
            epochs: total epochs.
            monitor_metric: key to track best.
            monitor_mode: "min" or "max".
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Monitor metric: {monitor_metric} ({monitor_mode})")
        if self.use_accelerate:
            self.logger.info(f"Using accelerate with {self.accelerator.num_processes} processes")
        
        start = self.current_epoch
        for ep in range(start, epochs + 1):
            self.current_epoch = ep

            tr = self.train_epoch(train_loader)
            self.train_metrics[ep] = tr

            if val_loader is not None and ep % self.val_interval == 0:
                va = self.validate(val_loader)
                self.val_metrics[ep] = va
                cur = va.get(monitor_metric)
                if cur is not None:
                    is_best = self._is_best_metric(cur, monitor_mode)
                    if is_best:
                        self.best_metric = cur
                        self.save_checkpoint(is_best=True)
                        self.logger.info(f"New best! {monitor_metric}: {cur:.4f}")
            # periodic save (keep original always‑save behavior if desired)
            # if ep % self.save_interval == 0:
            self.save_checkpoint(is_best=False)
                
            if ep % self.eval_per_epochs == 0:
                self.evaluate(ep)
        
        if not self.use_accelerate or (self.accelerator and self.accelerator.is_main_process):
            self.logger.info("Training completed!")
        if self.use_accelerate and self.accelerator is not None:
            self.accelerator.end_training()
            self.logger.info("Accelerate training ended")
        self._cleanup_logging_backends()
    
    def _is_best_metric(self, current: float, mode: str) -> bool:
        if self.best_metric is None:
            return True
        if mode == "min":
            return current < self.best_metric
        if mode == "max":
            return current > self.best_metric
        raise ValueError(f"Unknown monitor mode: {mode}")

    def _cleanup_logging_backends(self) -> None:
        if not self.use_accelerate or (self.accelerator and self.accelerator.is_main_process):
            if self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception as e:
                    self.logger.error(f"wandb finish failed: {e}")
            if self.tensorboard_writer is not None:
                try:
                    self.tensorboard_writer.close()
                except Exception as e:
                    self.logger.error(f"tensorboard close failed: {e}")
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        if self.use_accelerate and self.accelerator is not None:
            extra = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "train_metrics": dict(self.train_metrics),
                "val_metrics": dict(self.val_metrics),
            }
            if self.log_with == "wandb" and hasattr(self.accelerator, "trackers"):
                for tr in self.accelerator.trackers:
                    if hasattr(tr, "run") and hasattr(tr.run, "id"):
                        extra["wandb_run_id"] = tr.run.id
                        break
            if self.use_ema and self.ema is not None:
                extra["ema_state_dict"] = self.ema.state_dict()
            
            out = self.output_dir / "checkpoints" / "last"
            self.accelerator.save_state(output_dir=str(out), safe_serialization=True)
            if self.accelerator.is_main_process:
                torch.save(extra, out / "extra_state.pth")
            
            if is_best:
                self.accelerator.save_model(
                    self.model,
                    save_directory=str(self.output_dir / "best"),
                    safe_serialization=True
                )
            self.logger.info(f"Checkpoint saved at epoch {self.current_epoch}")
            return
        
        # vanilla
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "train_metrics": dict(self.train_metrics),
            "val_metrics": dict(self.val_metrics),
        }
            
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.use_ema and self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()

        torch.save(ckpt, self.output_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best.pth")
        self.logger.info(f"Checkpoint saved at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        p = Path(checkpoint_path)
        if self.use_accelerate and self.accelerator is not None:
            if p.is_dir() and (p / "model.safetensors").exists():
                self.accelerator.load_state(str(p))
                extra_p = p / "extra_state.pth"
                if extra_p.exists():
                    extra = torch.load(extra_p, map_location=self.device)
                    if self.use_ema and self.ema is not None and "ema_state_dict" in extra:
                        self.ema.load_state_dict(extra["ema_state_dict"])  # noqa: E501
                        self.logger.info("EMA state loaded from checkpoint")
                    self.current_epoch = extra.get("epoch", 1) + 1
                    self.global_step = extra.get("global_step", 1)
                    self.best_metric = extra.get("best_metric")
                    self.train_metrics = defaultdict(list, extra.get("train_metrics", {}))
                    self.val_metrics = defaultdict(list, extra.get("val_metrics", {}))
                else:
                    self.logger.warning("extra_state.pth not found; using defaults")
            else:
                raise ValueError(f"Unsupported checkpoint format: {p}")
            self.logger.info(f"Checkpoint loaded from {p}")
            self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
            return
        # vanilla
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.use_ema and self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
            self.logger.info("EMA state loaded from checkpoint")
        self.current_epoch = ckpt.get("epoch", 1) + 1
        self.global_step = ckpt.get("global_step", 1)
        self.best_metric = ckpt.get("best_metric")
        self.train_metrics = defaultdict(list, ckpt.get("train_metrics", {}))
        self.val_metrics = defaultdict(list, ckpt.get("val_metrics", {}))
        self.logger.info(f"Checkpoint loaded from {p}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

