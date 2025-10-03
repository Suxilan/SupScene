from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import argparse

# config & builder
from engine.conf import SupSceneConfig, load_config
from engine.builder import build_all_components
from utils import printf

def _safe_get(dct_or_obj, key: str, default: Optional[str] = None):
    if isinstance(dct_or_obj, dict):
        return dct_or_obj.get(key, default)
    return getattr(dct_or_obj, key, default)


def _import_trainer_class():
    """Try multiple paths to import SupSceneTrainer for robustness."""
    try:
        from supscene import SupSceneTrainer 
        return SupSceneTrainer
    except Exception:
        try:
            from supscene.sups_trainer import SupSceneTrainer  # alt
            return SupSceneTrainer
        except Exception:
            raise ImportError("Cannot import SupSceneTrainer from supscene module.")

def create_trainer(components: Dict[str, Any], config: SupSceneConfig, resume_wandb_id: Optional[str] = None):
    """Create SupSceneTrainer with logging/EMA/accelerate settings.

    Args:
        components: outputs from build_all_components
        config: SupSceneConfig
        resume_wandb_id: optional WandB run id to resume
    Returns:
        trainer instance
    """
    T = _import_trainer_class()

    data_cfg = config.data
    log_cfg = config.log
    model_cfg = config.model
    system_cfg = config.system
    optim_cfg = config.optim
    metric_cfg = config.metric

    # WandB config (flat summary fields)
    wb_cfg = dict(log_cfg.wandb_config or {})
    wb_cfg.update({
        "backbone": _safe_get(model_cfg.backbone, "name", "unknown"),
        "aggregator": _safe_get(model_cfg.aggregator, "name", "unknown"),
        "batch_size": data_cfg.batch_size,
        "learning_rate": optim_cfg.lr,
        "epochs": optim_cfg.epochs,
    })

    trainer = T(
        model=components["model"],
        optimizer=components["optimizer"],
        task_manager=components["task_manager"],
        scheduler=components["scheduler"],
        device=components["device"],
        output_dir=log_cfg.output_dir,
        root_dir=data_cfg.root_dir,
        split_txt=data_cfg.val_split_file,
        metric_pos_th=metric_cfg.metric_pos_th,
        metric_ks=tuple(metric_cfg.metric_ks),

        # train
        grad_clip=optim_cfg.grad_clip,
        grad_accum_steps=optim_cfg.accumulate_grad_batches,
        use_conflictfree=optim_cfg.use_conflictfree,
        seed=system_cfg.seed,
        deterministic=system_cfg.deterministic,
        device_specific=system_cfg.device_specific,

        # logging
        log_interval=log_cfg.log_interval,
        val_interval=log_cfg.eval_interval,
        save_interval=log_cfg.save_interval,
        eval_per_epochs=log_cfg.eval_per_epochs,

        # accelerate
        use_accelerate=system_cfg.use_accelerate,
        mixed_precision=system_cfg.mixed_precision,

        # backends
        log_with=log_cfg.log_with,
        wandb_project=log_cfg.wandb_project,
        wandb_name=log_cfg.wandb_name,
        wandb_config=wb_cfg,
        resume_wandb_id=resume_wandb_id,

        # EMA
        use_ema=_safe_get(model_cfg, "use_ema", False),
        ema_decay=_safe_get(model_cfg, "ema_decay", 0.999),
        ema_use_num_updates=_safe_get(model_cfg, "ema_use_num_updates", True),
    )
    return trainer

def _maybe_extract_wandb_id_from_checkpoint(resume_path: str) -> Optional[str]:
    """Read extra_state.pth (accelerate) or flat ckpt for wandb_run_id if present."""
    try:
        p = Path(resume_path)
        if p.is_dir():
            extra = p / "extra_state.pth"
            if extra.exists():
                st = torch.load(extra, map_location="cpu")
                return st.get("wandb_run_id")
        else:
            st = torch.load(p, map_location="cpu")
            return st.get("wandb_run_id")
    except Exception as e:
        printf(f"  - Failed to load checkpoint: {e}")
    return None

def train_model(config: SupSceneConfig, args: argparse.Namespace):
    """Full train loop orchestration."""
    resume_path = args.resume if args.resume else None
    rewandb = bool(args.rewandb)

    printf("🚀  Starting SupScene training...")
    printf(f"📁 Output directory: {config.log.output_dir}")
      
    components = build_all_components(config)
    
    # resolve wandb resume id
    resume_wandb_id: Optional[str] = None
    if resume_path and os.path.exists(resume_path) and rewandb:
        printf(f"📂 Detected resume path: {resume_path}")
        printf("🔄 Enabling WandB resume mode")
        resume_wandb_id = _maybe_extract_wandb_id_from_checkpoint(resume_path)
        if resume_wandb_id:
            printf(f"  - Found WandB run_id: {resume_wandb_id}")
        else:
            printf("  - No WandB run_id found, will create a new run")
    elif resume_path and os.path.exists(resume_path):
        printf(f"📂 Detected resume path: {resume_path}")
        printf("🆕 WandB resume disabled, will create a new run")
    
    # trainer
    printf("🎯 Creating trainer...")
    trainer = create_trainer(components, config, resume_wandb_id)

    # dataloaders: prepare once with accelerator
    train_loader = components["train_dataloader"]
    val_loader = components["val_dataloader"]
    if getattr(trainer, "use_accelerate", False):
        train_loader = trainer.accelerator.prepare(train_loader)
        if val_loader is not None:
            val_loader = trainer.accelerator.prepare(val_loader)

    # resume
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        printf(f"📂 Resuming training from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
        start_epoch = getattr(trainer, "current_epoch", 1)
        printf(f"  - Resuming from epoch {start_epoch}")

    # optional pre-eval
    if getattr(config.log, "eval_on_start", False) and start_epoch == 1:
        printf("🔍 Running pre-training evaluation...")
        trainer.evaluate(epoch=0)

    # fit
    printf(f"🏃 Starting training for {config.optim.epochs} epochs...")
    t0 = time.time()
    try:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.optim.epochs,
            monitor_metric=config.log.monitor_metric,
            monitor_mode=config.log.monitor_mode,
        )
        dt = time.time() - t0
        printf(f"✅ Training finished! Total time: {dt/3600:.2f} hours")
        if getattr(trainer, "best_metric", None) is not None:
            printf(f"🏆 Best metric: {trainer.best_metric:.6f}")
    except KeyboardInterrupt:
        if getattr(trainer, "use_accelerate", False):
            trainer.accelerator.print("\n⚠️  Training interrupted by user")
            if trainer.accelerator.is_main_process:
                trainer.save_checkpoint(epoch=trainer.current_epoch, is_best=False, suffix="interrupted")
                trainer.accelerator.print("💾 Interrupted checkpoint saved")
        else:
            printf("\n⚠️  Training interrupted by user")
            trainer.save_checkpoint(epoch=trainer.current_epoch, is_best=False, suffix="interrupted")
            printf("💾 Interrupted checkpoint saved")
    except Exception as e:
        printf(f"❌ Error during training: {e}")
        raise

def evaluate_model(config: SupSceneConfig, args: argparse.Namespace):
    """Run full evaluation from a checkpoint path."""
    printf("🔍 Starting model evaluation...")
    if not args.resume:
        raise ValueError("Evaluation mode requires --resume argument")
    ckpt = args.resume
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")

    components = build_all_components(config)
    trainer = create_trainer(components, config)
    trainer.load_checkpoint(ckpt)

    printf("📊 Running full evaluation...")
    trainer.evaluate(epoch="final")
    printf("✅ Evaluation complete!")

# ---------- CLI ----------
def _build_argparser():
    p = argparse.ArgumentParser(description="OrbitSFM training launcher")
    p.add_argument("--config", type=str, help="Path to configuration file", required=True)
    p.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    p.add_argument("--rewandb", action="store_true", help="Resume WandB run from checkpoint", default=False)
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only", default=False)
    return p


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    # load config
    config: SupSceneConfig = load_config(args.config, args)

    # I/O
    os.makedirs(config.log.output_dir, exist_ok=True)
    cfg_out = os.path.join(config.log.output_dir, "config.yaml")
    config.to_yaml(cfg_out)
    printf(f"💾 Configuration saved to: {cfg_out}")

    # print summary
    printf("\n" + "=" * 60)
    printf("📋 Training configuration summary")
    printf("=" * 60)
    printf(f"🏗️  Model: {_safe_get(config.model.backbone, 'name', 'unknown')} + {_safe_get(config.model.aggregator, 'name', 'unknown')}")
    printf(f"📊 Data: batch_size={config.data.batch_size}, n_sub={config.data.n_sub}")
    printf(f"🎯 Task: contrast={config.task.contrast_enabled}, huber={config.task.huber_enabled}")
    printf(f"⚙️  Optimization: {config.optim.optimizer}, lr={config.optim.lr}, epochs={config.optim.epochs}")
    printf(f"📱 Device: {config.system.device}, mixed_precision={config.system.mixed_precision}")
    printf(f"📝 Logging: {config.log.log_with or 'console'}")
    printf("=" * 60 + "\n")

    # mode
    if args.eval_only:
        if not args.resume:
            raise ValueError("Evaluation mode requires --resume argument")
        evaluate_model(config, args)
    else:
        train_model(config, args)


if __name__ == "__main__":
    main()