from __future__ import annotations

import os, sys, dataclasses
from typing import Dict, Any

# repo path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate.utils import set_seed

from torch.optim import AdamW, Adam, SGD

# cfg & utils
from engine.conf import SupSceneConfig
from supscene.datasets import GL3DSubgraphDataset, SubgraphSampler, make_pad_collate
from supscene import create_encoder
from supscene import TaskManager, TaskConfig
from utils.lr_schedulers import create_scheduler
from utils import BaseTrainer, printf

def build_dataset(config: SupSceneConfig, split: str = "train") -> GL3DSubgraphDataset:
    """Create GL3D subgraph dataset for a split.

    Args:
        config: global config dataclass
        split: "train" | "val"
    Returns:
        GL3DSubgraphDataset
    """
    data_cfg = config.data

    if split == "train":
        split_file = data_cfg.split_file
    elif split == "val":
        split_file = data_cfg.val_split_file
    
    else:
        raise ValueError(f"Unknown split: {split}")

    sampler = SubgraphSampler(
        mode=data_cfg.sampler_mode,
        iou_th=data_cfg.iou_thresh,
        topk_per_hop=data_cfg.topk_per_hop,
    )

    ds = GL3DSubgraphDataset(
        root_dir=data_cfg.root_dir,
        split_txt=split_file,
        n_sub=data_cfg.n_sub,
        sampler=sampler,
        load_images=data_cfg.load_images,
        img_size=int(data_cfg.image_size),  # keep int even when not loading images
        scenes_per_epoch=data_cfg.scenes_per_epoch,
        samples_per_scene=data_cfg.samples_per_scene,
        teacher_name=data_cfg.teacher_name,
        min_images_per_scene=data_cfg.min_images_per_scene,
    )
    return ds

def build_dataloader(dataset: GL3DSubgraphDataset, config: SupSceneConfig, split: str = "train") -> DataLoader:
    """Wrap dataset with DataLoader.

    Shapes:
        per batch images: [B, N, 3, H, W] or [B, 3, H, W] (dataset controls)
    """
    data_cfg = config.data
    collate_fn = make_pad_collate(diag_weight=data_cfg.diag_weight)

    if split == "train":
        shuffle = data_cfg.shuffle
        batch_size = data_cfg.batch_size
    else:
        shuffle = False
        batch_size = data_cfg.batch_size

    seed = config.system.seed
    gen = BaseTrainer.create_generator(seed + (1 if split == "train" else 0))

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
        generator=gen,
        worker_init_fn=BaseTrainer.seed_worker if data_cfg.num_workers > 0 else None,
    )
    return dl

def count_params(model: nn.Module):
    """Print trainable params and return (total, trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    printf("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            printf(f"  {name}: {param.numel()} params")
    return total, trainable

def build_model(config: SupSceneConfig) -> nn.Module:
    """Build encoder from config.model using `create_encoder`.

    Note: `config.model` should be a dataclass containing backbone/aggregator/head.
    """
    model_cfg = config.model
    model = create_encoder(dataclasses.asdict(model_cfg))
    return model

def build_task_manager(config: SupSceneConfig, in_dim: int) -> TaskManager:
    """Assemble TaskManager from config.task/head/loss with given input dim."""
    task_cfg = config.task

    task_cfgs = [
        TaskConfig(name="contrast", enabled=task_cfg.contrast_enabled, head_name=None, loss_name="contrast_loss", loss_weight=task_cfg.contrast_weight),
        # TaskConfig(name="huber", enabled=task_cfg.huber_enabled, head_name="overlap_head", loss_name="huber_loss", loss_weight=task_cfg.huber_weight),
        # TaskConfig(name="lowrank", enabled=task_cfg.lowrank_enabled, head_name="cluster_head", loss_name="lowrank_loss", loss_weight=task_cfg.lowrank_weight),
        # TaskConfig(name="distill", enabled=task_cfg.distill_enabled, head_name="distill_head", loss_name="distill_loss", loss_weight=task_cfg.distill_weight),
    ]

    tm = TaskManager(
        in_dim=in_dim,
        task_cfgs=task_cfgs,
        head_cfgs=dataclasses.asdict(config.head),
        loss_cfgs=dataclasses.asdict(config.loss),
    )
    return tm

def build_optimizer(model: nn.Module, task_manager: TaskManager, config: SupSceneConfig) -> torch.optim.Optimizer:
    """Create optimizer over encoder + heads.

    Note: keep behavior (optimize all params, even if some are frozen now).
    """
    optim_cfg = config.optim
    params = list(model.parameters()) + list(task_manager.parameters())
    
    if optim_cfg.optimizer.lower() == "adamw":
        opt = AdamW(
            params,
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
        )
    elif optim_cfg.optimizer.lower() == "adam":
        opt = Adam(
            params,
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
        )
    elif optim_cfg.optimizer.lower() == "sgd":
        opt = SGD(
            params,
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_cfg.optimizer}")
    return opt

def build_scheduler(optimizer: torch.optim.Optimizer, dataloader: DataLoader, config: SupSceneConfig):
    """Create LR scheduler by name (warmup, cosine, etc.)."""
    optim_cfg = config.optim
    sch = create_scheduler(
        optimizer,
        scheme=optim_cfg.scheduler,
        dataloader=dataloader,
        epochs=optim_cfg.epochs,
        warmup_pct=optim_cfg.warmup_pct,
        warmup_start_factor=optim_cfg.warmup_start_factor,
        args={"eta_min": optim_cfg.eta_min},
    )
    return sch

def setup_device(config: SupSceneConfig) -> str:
    """Pick device string and enable safe SDP kernels when available."""
    sys_cfg = config.system
    if sys_cfg.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = sys_cfg.device

    if dev == "cuda" and not torch.cuda.is_available():
        printf("⚠️  CUDA not available, fallback to CPU")
        dev = "cpu"

    # enable SDP knobs if present (PyTorch ≥ 2.x)
    if torch.cuda.is_available():
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    return dev

def seed_everything(seed: int, device_specific: bool = False, deterministic: bool = False): 
    if deterministic:
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        set_seed(seed, device_specific=device_specific, deterministic=deterministic)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        set_seed(seed, device_specific=device_specific, deterministic=deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def build_all_components(config: SupSceneConfig) -> Dict[str, Any]:
    """Build datasets, model, tasks, optimizer, scheduler.

    Returns a dict:
        {"model", "task_manager", "optimizer", "scheduler",
         "train_dataloader", "val_dataloader", "device"}
    """
    printf("🔧 Building components…")
    seed_everything(config.system.seed, config.system.device_specific, config.system.deterministic)

    device = setup_device(config)
    printf(f"📱 Device: {device}")

    # data
    printf("📊 Building datasets…")
    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")

    train_dataloader = build_dataloader(train_dataset, config, "train")
    val_dataloader = build_dataloader(val_dataset, config, "val")

    printf(f"  - Train: {len(train_dataset)} samples, {len(train_dataloader)} batches")
    printf(f"  - Val  : {len(val_dataset)} samples, {len(val_dataloader)} batches")

    # model
    printf("🏗️  Building encoder…")
    model = build_model(config)
    total, trainable = count_params(model)
    printf(f"Params — total: {total/1e6:.3f}M, trainable: {trainable/1e6:.3f}M")
    in_dim = model.output_dim

    # tasks
    printf("🎯 Building task manager…")
    task_manager = build_task_manager(config, in_dim)

    # optim & sched
    printf("⚙️  Building optimizer/scheduler…")
    optimizer = build_optimizer(model, task_manager, config)
    scheduler = build_scheduler(optimizer, train_dataloader, config)

    printf("✅ Components ready!")
    return {
        "model": model,
        "task_manager": task_manager,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "device": device,
    }


if __name__ == "__main__":
    printf("🧪 Smoke test…")
    cfg = SupSceneConfig.from_yaml("configs/default.yaml")

    # fast test overrides
    cfg.data.root_dir = "data"
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.optim.epochs = 5

    try:
        comps = build_all_components(cfg)
        printf("✅ Build OK!")

        # quick forward sanity (no actual model call here; encoder API may vary)
        model = comps["model"].eval()
        td = comps["train_dataloader"]
        batch = next(iter(td))
        printf("🔄 Forward pass is model‑specific; integrate in your trainer.")
        printf("✅ Smoke test done!")
    except Exception as e:
        printf(f"❌ Build failed: {e}")