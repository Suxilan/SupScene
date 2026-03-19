"""
SupScene training configuration manager.
Supports YAML configuration files and command-line argument overrides.
"""
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from utils import printf


@dataclass
class DataConfig:
    root_dir: str = "data"
    split_file: str = "data/GL3D/train"
    val_split_file: str = "data/GL3D/test"
    n_sub: int = 128
    sampler_mode: str = "anchor_expand"  # anchor_expand, random, density
    iou_thresh: float = 0.2
    topk_per_hop: int = 32
    load_images: bool = True
    image_size: int = 322
    scenes_per_epoch: Optional[int] = None  # None=all scenes
    samples_per_scene: Optional[int] = None  # None=adaptive based on scene size
    teacher_name: Optional[str] = None
    diag_weight: float = 0.1
    
    # DataLoader
    batch_size: int = 2
    num_workers: int = 8
    pin_memory: bool = True
    shuffle: bool = True
    min_images_per_scene: int = 0


@dataclass 
class ModelConfig:
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "dinov2", 
        "args": {
            "model_name": "dinov2_vitb14", 
            "num_trainable_blocks": 1, 
            "return_cls_token": True,
            "return_attn_maps": False
        }})
    
    aggregator: Dict[str, Any] = field(default_factory=lambda: {
        "name": "scpp", 
        "args": {}
    })
    
    deploy_head: Dict[str, Any] = field(default_factory=lambda: {
        "name": "deploy_head", 
        "args": {}
    })
    
    weights: Dict[str, Any] = field(default_factory=lambda:None)
    
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_use_num_updates: bool = True
    
@dataclass
class HeadConfig:
    contrast_head: Dict[str, Any] = field(default_factory=lambda: {
        "type": "ContrastiveHead",
        "params": {
            "style": "moco",
            "out_dim": 256,
            "hidden_dim": 2048,
            "nlayers": 2,
            "last_bn": False
        }
    })

    # distill_head: Dict[str, Any] = field(default_factory=lambda: {
    #     "type": "DistillHead",
    #     "params": {
    #         "style": "moco",
    #         "out_dim": 256,
    #         "hidden_dim": 2048,
    #         "nlayers": 3,
    #         "last_bn": False
    #     }
    # })

    # cluster_head: Dict[str, Any] = field(default_factory=lambda: {
    #     "type": "SimpleClusterHead",
    #     "params": {
    #         "K": 64,
    #         "hidden": 512,
    #         "dropout": 0.1,
    #         "tau": 0.07,
    #         "learnable_tau": False,
    #         "num_layers": 2,
    #         "bias": True,
    #     }
    # })

    # overlap_head: Dict[str, Any] = field(default_factory=lambda: {
    #     "type": "OverlapPredictorHead",
    #     "params": {
    #         "use_bias": True,
    #         "apply_sigmoid": True
    #     }
    # })

@dataclass
class LossConfig:
    contrast_loss: Dict[str, Any] = field(default_factory=lambda: {
        "type": "MultiSimilarityLoss",
        "params": {
            "pos_th": 0.25,
            "exclude_self": True,
            "eps": 1e-8,
            "alpha": 2.0,
            "beta": 50.0,
            "base": 0.5,
            "rank_weight": 10.0,
            "ov_margin": 0.05,
            "sim_margin": 0.05,
        }
    })
    distill_loss: Dict[str, Any] = field(default_factory=lambda: {
        "type": "DistillLoss",
        "params": {
            "distill_type": "relation",   # "relation" | "cosine" | "mse" | "kl"
            "tau": 4.0
        }
    })
    # huber_loss: Dict[str, Any] = field(default_factory=lambda: {
    #     "type": "HuberPPLoss",
    #     "params": {
    #         "huber_delta": 0.1
    #     }
    # })
    # lowrank_loss: Dict[str, Any] = field(default_factory=lambda: {
    #     "type": "LowRankRegularizer",
    #     "params": {
    #         "lowrank_weights": [1.0, 0.1, 2.0]
    #     }
    # })

@dataclass
class TaskConfig:
    contrast_enabled: bool = True
    huber_enabled: bool = False
    lowrank_enabled: bool = False
    distill_enabled: bool = False

    contrast_weight: float = 1.0
    huber_weight: float = 0.5
    lowrank_weight: float = 0.2
    distill_weight: float = 5.0

@dataclass
class OptimConfig:
    optimizer: str = "adamw"  # adamw, adam, sgd
    lr: float = 1e-4
    weight_decay: float = 1e-2

    scheduler: str = "cosine"  # cosine, linear, step, plateau
    warmup_pct: float = 0.1
    warmup_start_factor: float = 0.01
    eta_min: float = 1e-7   # args
    
    epochs: int = 50
    grad_clip: float = 1.0
    accumulate_grad_batches: int = 1
    
    use_conflictfree: bool = False

@dataclass
class LogConfig:
    output_dir: str = "experiments/scpp"
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 1
    eval_per_epochs: int = 1
    
    log_with: str = "wandb"  # wandb, tensorboard, None
    wandb_project: str = "SupScene"
    wandb_name: Optional[str] = "SCPP"
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    eval_on_start: bool = True
    monitor_metric: str = "metric/recall@25"
    monitor_mode: str = "max"  # min, max

@dataclass
class MetricConfig:
    metric_pos_th: float = 0.25
    metric_ks: List[int] = field(default_factory=lambda: [1, 25, 50, 100])

@dataclass
class SystemConfig:
    device: str = "cuda"  # auto, cuda, cpu
    use_accelerate: bool = True
    mixed_precision: str = "no"  # fp16, bf16, no
    seed: int = 42
    deterministic: bool = True
    device_specific: bool = False

@dataclass
class SupSceneConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    log: LogConfig = field(default_factory=LogConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SupSceneConfig':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # recursive function to create dataclass from dict
        def create_config(config_cls, config_data):
            if config_data is None:
                return config_cls()
            
            # filter out invalid fields
            valid_fields = {f.name for f in config_cls.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}
            
            for field_name, field_info in config_cls.__dataclass_fields__.items():
                if field_name in filtered_data:
                    field_type = field_info.type
                    if hasattr(field_type, '__dataclass_fields__'):
                        filtered_data[field_name] = create_config(field_type, filtered_data[field_name])
            
            return config_cls(**filtered_data)
        
        return create_config(cls, config_dict)
    
    def to_yaml(self, output_path: str):
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        config_dict = dataclass_to_dict(self)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def update_from_args(self, args: argparse.Namespace):
        flat_mapping = {
            'data_root': ('data', 'root_dir'),
            'batch_size': ('data', 'batch_size'),
            'num_workers': ('data', 'num_workers'),
            'n_sub': ('data', 'n_sub'),
            
            'backbone': ('model', 'backbone_name'),
            'feature_dim': ('model', 'feature_dim'),
            'ema_decay': ('model', 'ema_decay'),
            
            'lr': ('optim', 'lr'),
            'epochs': ('optim', 'epochs'),
            'weight_decay': ('optim', 'weight_decay'),
            
            'output_dir': ('log', 'output_dir'),
            'log_interval': ('log', 'log_interval'),
            'wandb_project': ('log', 'wandb_project'),
        
            'device': ('system', 'device'),
            'seed': ('system', 'seed'),
        }
        
        for arg_name, (section, field) in flat_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                section_obj = getattr(self, section)
                setattr(section_obj, field, getattr(args, arg_name))


def create_default_config() -> SupSceneConfig:
    """create a default configuration"""
    return SupSceneConfig()


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="SupScene Training")
    
    # yaml config file
    parser.add_argument('--config', type=str, help='path to config YAML file')

    # data parameters
    parser.add_argument('--data_root', type=str, help='data root directory')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers')
    parser.add_argument('--n_sub', type=int, help='number of submaps per scene')
    
    # model parameters
    parser.add_argument('--backbone', type=str, help='backbone model name')
    parser.add_argument('--feature_dim', type=int, help='feature dimension')
    parser.add_argument('--ema_decay', type=float, help='EMA decay ratio')

    # optimization parameters
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='epoch numbers')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    
    # logging parameters
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--log_interval', type=int, help='interval for logging')
    parser.add_argument('--wandb_project', type=str, help='WandB project name')
    
    # system parameters
    parser.add_argument('--device', type=str, help='device type: auto, cuda, cpu')
    parser.add_argument('--seed', type=int, help='random seed')

    # others
    parser.add_argument('--resume', type=str, help='path to resume training checkpoint')
    parser.add_argument('--rewandb', action='store_true', help='resume wandb run from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='evaluation mode only')

    return parser.parse_args()


def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> SupSceneConfig:
    resolved_path: Optional[str] = None
    if config_path:
        if os.path.exists(config_path):
            resolved_path = config_path
        else:
            cfg_in_configs = os.path.join("configs", config_path)
            if os.path.exists(cfg_in_configs):
                resolved_path = cfg_in_configs

    if resolved_path is not None:
        config = SupSceneConfig.from_yaml(resolved_path)
        printf(f"✅ successfully loading config from: {resolved_path}")
    else:
        config = create_default_config()
        if config_path:
            printf(f"⚠️ config not found: {config_path}, fallback to default config")
        else:
            printf("✅ use default config")

    if args:
        config.update_from_args(args)
        printf("✅ successfully update config from command line arguments")
    
    return config


if __name__ == "__main__":
    config = create_default_config()
    
    os.makedirs("configs", exist_ok=True)
    config.to_yaml("configs/default.yaml")
    printf("✅ default config saved to: configs/default.yaml")

    loaded_config = SupSceneConfig.from_yaml("configs/default.yaml")
    printf(f"✅ successfully loaded config from YAML")
    
    import sys
    sys.argv = ['config.py', '--lr', '2e-4', '--epochs', '50']
    args = parse_args()
    loaded_config.update_from_args(args)
    printf(f"✅ successfully update config from command line arguments: lr={loaded_config.optim.lr}, epochs={loaded_config.optim.epochs}")
