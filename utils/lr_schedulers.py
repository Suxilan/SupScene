from __future__ import annotations

from typing import  Optional, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    LinearLR,
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    SequentialLR,
    ConstantLR,
)

def create_scheduler(
    optimizer: Optimizer,
    scheme: str = "cosine",
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    epochs: Optional[int] = None,
    total_steps: Optional[int] = None,
    warmup_pct: float = 0.0,
    warmup_start_factor: float = 0.01,
    args: Optional[Dict[str, Any]] = None,
) -> _LRScheduler:
    """
    create and return a learning rate scheduler

    Args:
        optimizer: the optimizer to attach the scheduler to
        scheme: the main scheduling scheme to use, one of:
            - "cosine": cosine annealing
            - "step": step decay
            - "multistep": multi-step decay
            - "exponential": exponential decay
            - "constant": constant LR (usually for stable phase)
        dataloader: the dataloader used for training (required if epochs is provided and total_steps is None)
        epochs: number of epochs to train (required if dataloader is provided and total_steps is None)
        total_steps: total number of training steps (overrides dataloader and epochs if provided)
        warmup_pct: fraction of total steps to use for linear warmup (0.0 means no warmup)
        warmup_start_factor: starting factor for LinearLR warmup (multiplied by base LR)
        args: additional arguments for the specific scheduling scheme

    Returns:
        _LRScheduler instance (possibly SequentialLR when warmup_pct > 0)
    """
    args = args or {}

    # total_steps
    if total_steps is not None:
        steps = int(total_steps)
    else:
        if dataloader is None or epochs is None:
            raise ValueError("create_scheduler: need to provide total_steps or (dataloader, epochs)")
        steps = int(len(dataloader) * epochs)
    if steps <= 0:
        raise ValueError("create_scheduler: the total_steps must be > 0")

    scheme = scheme.lower()

    # if warmup_pct > 0, create LinearLR warmup
    warmup_steps = 0
    if warmup_pct and warmup_pct > 0.0:
        warmup_steps = max(1, int(steps * warmup_pct))
    remaining_steps = max(1, steps - warmup_steps) if warmup_steps > 0 else steps

    if scheme == "cosine":
        eta_min = float(args.get("eta_min", 0.0))
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining_steps), eta_min=eta_min)

    elif scheme == "step":
        step_size = args.get("step_size")
        if step_size is None:
            raise ValueError("step scheme need args['step_size']")
        gamma = float(args.get("gamma", 0.1))
        main_scheduler = StepLR(optimizer, step_size=int(step_size), gamma=gamma)

    elif scheme == "multistep":
        milestones = args.get("milestones")
        if not milestones:
            raise ValueError("multistep scheme need args['milestones'] (List[int])")
        gamma = float(args.get("gamma", 0.1))
        main_scheduler = MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)

    elif scheme == "exponential":
        gamma = float(args.get("gamma", 0.95))
        main_scheduler = ExponentialLR(optimizer, gamma=gamma)

    elif scheme == "constant":
        factor = float(args.get("factor", 1.0))
        main_scheduler = ConstantLR(optimizer, factor=factor, total_iters=max(1, remaining_steps))

    else:
        raise ValueError(f"Unknown scheduling scheme: {scheme}")

    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=float(warmup_start_factor),
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_steps],
        )

    return main_scheduler


# def example_usage():
#     model = nn.Linear(10, 1)
#     optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
#     warmup_steps = 1000
#     total_steps = 10000
    
#     scheduler = create_sequential_scheduler(
#         optimizer=optimizer,
#         warmup_steps=warmup_steps,
#         total_steps=total_steps,
#         warmup_start_factor=0.01,  
#         cosine_eta_min=1e-6
#     )
    
#     print("Step\tLR")
#     print("-" * 20)
    
#     for step in range(0, total_steps, 500):
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"{step}\t{current_lr:.2e}")
        
#         optimizer.zero_grad()
#         optimizer.step()
#         scheduler.step()
    
#     print(f"{total_steps}\t{optimizer.param_groups[0]['lr']:.2e}")

# def create_advanced_scheduler(
#     optimizer: torch.optim.Optimizer,
#     warmup_steps: int = 1000,
#     stable_steps: int = 2000,
#     decay_steps: int = 7000,
#     warmup_start_factor: float = 0.01,
#     cosine_eta_min: float = 1e-6
# ):

#     from torch.optim.lr_scheduler import ConstantLR
    
#     linear_scheduler = LinearLR(
#         optimizer,
#         start_factor=warmup_start_factor,
#         end_factor=1.0,
#         total_iters=warmup_steps
#     )
    
#     constant_scheduler = ConstantLR(
#         optimizer,
#         factor=1.0,
#         total_iters=stable_steps
#     )
    
#     cosine_scheduler = CosineAnnealingLR(
#         optimizer,
#         T_max=decay_steps,
#         eta_min=cosine_eta_min
#     )
    
#     sequential_scheduler = SequentialLR(
#         optimizer,
#         schedulers=[linear_scheduler, constant_scheduler, cosine_scheduler],
#         milestones=[warmup_steps, warmup_steps + stable_steps]
#     )
    
#     return sequential_scheduler


# if __name__ == "__main__":
#     print("=== Basic LinearLR + CosineAnnealingLR example ===")
#     example_usage()
    
#     model = nn.Linear(10, 1)
#     optimizer = AdamW(model.parameters(), lr=1e-3)
    
#     advanced_scheduler = create_advanced_scheduler(
#         optimizer=optimizer,
#         warmup_steps=1000,
#         stable_steps=2000,
#         decay_steps=7000
#     )
    
#     print("Step\tPhase\t\tLR")
#     print("-" * 35)