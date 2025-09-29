# orbit/eval.py
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 依赖你已有的数据工具 ---
from .datasets.gl3d_subgraph_dataset import read_lines
from .datasets.gl3d_batch_dataset import GL3DBatchDataset

# --- 依赖你的检索指标实现 ---
try:
    from utils.metrics import (
        compute_retrieval_metrics,
        compute_global_retrieval_metrics,
    )
except ImportError:
    compute_retrieval_metrics = None
    compute_global_retrieval_metrics = None

# --- Accelerate支持 ---
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


# -----------------------------
# 配置与小工具
# -----------------------------
@dataclass
class EvalConfig:
    root_dir: str                 # GL3D 根目录（含 GL3D/{scene_id}）
    split_txt: str                # split 文件（如 val.txt）
    img_size: int = 322
    batch_size: int = 256
    num_workers: int = 4
    device: str = "cuda"
    ks: Tuple[int, ...] = (1, 5, 10, 20)
    pos_th: float = 0.25
    mode: str = "similarity"      # "similarity" or "distance"
    save_embeds: bool = False     # 是否保存场景 embeddings 到硬盘
    embeds_dir: Optional[str] = None  # 保存路径
    use_amp: bool = False         # 半精度推理
    pin_memory: bool = True
    persistent_workers: bool = True
    global_retrieval: bool = False  # 是否启用全局检索模式
    use_accelerate: bool = True   # 是否使用accelerate（自动检测DDP）
    accelerator: Optional[Any] = None  # 外部传入的accelerator实例


# -----------------------------
# 核心：Evaluator
# -----------------------------
class OrbitEvaluator:
    """
    用于在 GL3D split 上做全量前向与检索评估的评估器。

    - 支持逐场景评估与数据集聚合（macro/micro）
    - 支持全局检索模式：高效计算全局检索指标
    - 支持DDP/Accelerate：分布式特征提取，主进程评估
    - 可选择把每个场景的 embeddings 落盘（复用）
    - 训练循环中可直接持有实例并反复调用 run(...)
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
            raise RuntimeError("未找到 compute_retrieval_metrics，请通过参数 metrics_fn 注入")

        # 初始化Accelerator
        self.accelerator = None
        self.is_main_process = True
        
        if cfg.use_accelerate and ACCELERATE_AVAILABLE:
            if cfg.accelerator is not None:
                # 使用外部传入的accelerator（训练中调用）
                self.accelerator = cfg.accelerator
                self.encoder = encoder.eval()  # 假设已经被prepare过
            else:
                # 创建新的accelerator（独立评估）
                self.accelerator = Accelerator()
                self.encoder = self.accelerator.prepare(encoder).eval()
            
            self.is_main_process = self.accelerator.is_main_process
            self.device = self.accelerator.device
        else:
            # 单卡模式
            self.encoder = encoder.to(cfg.device)
            self.device = cfg.device

        # 解析场景列表
        split_list = read_lines(cfg.split_txt)
        self.scene_dirs = [os.path.join(cfg.root_dir, "GL3D", sid) for sid in split_list]
        self.scene_ids = [os.path.basename(p) for p in self.scene_dirs]

        # 混合精度
        self.amp_dtype = torch.float16 if cfg.use_amp and "cuda" in str(self.device) else None

        # 准备保存目录（仅主进程）
        if cfg.save_embeds and self.is_main_process:
            self.embeds_dir = cfg.embeds_dir or os.path.join(cfg.root_dir, "embeds")
            os.makedirs(self.embeds_dir, exist_ok=True)
        else:
            self.embeds_dir = None

    @torch.no_grad()
    def _embed_all_scenes_efficient(self) -> Tuple[torch.Tensor, List[torch.Tensor], List[str], List[int]]:
        """
        高效批量提取所有场景的特征，避免重复创建DataLoader
        返回：
          - all_emb: [total_N, D] 所有图片的embeddings
          - scene_overlaps: List[torch.Tensor] 每个场景的重叠矩阵
          - scene_ids: 场景ID列表
          - scene_offsets: 每个场景在全局中的起始位置
        """
        # 创建批量数据集
        batch_dataset = GL3DBatchDataset(self.scene_dirs, img_size=self.cfg.img_size)
        
        # 创建DataLoader
        dataloader = DataLoader(
            batch_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=False,
            num_workers=self.cfg.num_workers, 
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
        )
        
        # 如果使用accelerate，准备dataloader
        if self.accelerator is not None:
            dataloader = self.accelerator.prepare(dataloader)
        
        # 批量前向推理
        all_embeddings = []
        
        if self.is_main_process:
            print(f"Extracting features for {len(batch_dataset)} images...")
        
        for batch in tqdm(dataloader, desc="Extracting features", disable=not self.is_main_process):
            batch = batch.to(self.device, non_blocking=True)
            
            if self.amp_dtype is None:
                embeddings = self.encoder(batch)
            else:
                with torch.amp.autocast(str(self.device).split(':')[0], dtype=self.amp_dtype):
                    embeddings = self.encoder(batch)
            
            embeddings = embeddings.float()
            
            # 收集DDP进程的embeddings（在batch内收集避免padding）
            if self.accelerator is not None:
                embeddings = self.accelerator.gather_for_metrics(embeddings)
            
            all_embeddings.append(embeddings)
        
        # 拼接所有embeddings
        all_emb = torch.cat(all_embeddings, dim=0)  # [total_N, D]
        
        # 获取场景信息（仅主进程处理）
        scene_overlaps = []
        scene_ids = []
        scene_offsets = []
        
        if self.is_main_process:
            scene_info = batch_dataset.get_scene_info()
            current_offset = 0
            
            for scene_id, start_idx, end_idx, overlap_matrix in scene_info:
                scene_ids.append(scene_id)
                scene_offsets.append(current_offset)
                scene_overlaps.append(torch.from_numpy(overlap_matrix).float())
                current_offset += (end_idx - start_idx)
                
                # 保存embeddings（如果需要）
                if self.embeds_dir:
                    scene_emb = all_emb[start_idx:end_idx].cpu().numpy()
                    npy_path = os.path.join(self.embeds_dir, f"{scene_id}.npy")
                    np.save(npy_path, scene_emb)
        
        return all_emb, scene_overlaps, scene_ids, scene_offsets


    @torch.no_grad()
    def evaluate_scene(self, scene_dir: str) -> Dict[str, Any]:
        """
        评估单个场景，返回：
          {
            "scene_id": str,
            "N": int,
            "metrics": { "recall@k": .., "precision@k": .., "map@k": .., "ndcg@k": .. },
            "num_valid_queries": int
          }
        """
        emb, O, scene_id = self._embed_scene(scene_dir)

        # node_mask（全部有效）
        node_mask = torch.ones(emb.size(0), dtype=torch.bool)

        # 评估（你的 metrics 接口：emb, overlap, ks, mask=None, pos_th, mode, node_mask）
        # 这里 mask 传 None，因为单场景内部全部有效；对角在你的 metrics 内部会屏蔽
        metrics = self.metrics_fn(
            emb=emb,
            overlap=torch.from_numpy(O),
            ks=self.cfg.ks,
            mask=None,
            pos_th=self.cfg.pos_th,
            mode=self.cfg.mode,
            node_mask=node_mask,
        )

        # 有效 query 数量：有正样本的行
        Y = (torch.from_numpy(O) >= self.cfg.pos_th)
        torch.diagonal(Y).fill_(False)
        num_valid = (Y.sum(dim=-1) > 0).sum().item()

        return {
            "scene_id": scene_id,
            "N": emb.size(0),
            "num_valid_queries": num_valid,
            "metrics": metrics,
        }

    @torch.no_grad()
    def evaluate_global_efficient(
        self,
        all_emb: torch.Tensor,
        scene_overlaps: List[torch.Tensor],
        scene_offsets: List[int]) -> Dict[str, Any]:
        """
        高效全局检索评估：避免构建完整重叠矩阵
        """
        if compute_global_retrieval_metrics is None:
            raise RuntimeError("未找到 compute_global_retrieval_metrics，请检查 utils.global_retrieval 模块")
        
        # 使用高效的全局检索指标计算
        global_metrics = compute_global_retrieval_metrics(
            all_emb=all_emb,
            scene_overlaps=scene_overlaps,
            scene_offsets=scene_offsets,
            ks=self.cfg.ks,
            pos_th=self.cfg.pos_th,
            mode=self.cfg.mode
        )
        
        # 统计有效query数量
        num_valid = 0
        for i, overlap in enumerate(scene_overlaps):
            Y = (overlap >= self.cfg.pos_th)
            Y.fill_diagonal_(False)
            num_valid += (Y.sum(dim=-1) > 0).sum().item()
        
        return {
            "global_metrics": global_metrics,
            "total_images": all_emb.size(0),
            "num_scenes": len(scene_overlaps),
            "num_valid_queries": num_valid,
        }

    @torch.no_grad()
    def evaluate_per_scene_efficient(
        self,
        all_emb: torch.Tensor,
        scene_overlaps: List[torch.Tensor],
        scene_ids: List[str],
        scene_offsets: List[int]) -> List[Dict[str, Any]]:
        """
        从全局embeddings中分场景评估
        """
        per_scene = []
        
        for i, (scene_id, overlap) in enumerate(zip(scene_ids, scene_overlaps)):
            start = scene_offsets[i]
            end = scene_offsets[i + 1] if i + 1 < len(scene_offsets) else all_emb.size(0)
            
            # 提取该场景的embeddings
            scene_emb = all_emb[start:end]
            
            # node_mask（全部有效）
            node_mask = torch.ones(scene_emb.size(0), dtype=torch.bool)
            
            try:
                # 计算该场景的检索指标
                metrics = self.metrics_fn(
                    emb=scene_emb,
                    overlap=overlap,
                    ks=self.cfg.ks,
                    mask=None,
                    pos_th=self.cfg.pos_th,
                    mode=self.cfg.mode,
                    node_mask=node_mask,
                )
                
                # 有效 query 数量
                Y = (overlap >= self.cfg.pos_th)
                Y.fill_diagonal_(False)
                num_valid = (Y.sum(dim=-1) > 0).sum().item()
                
                result = {
                    "scene_id": scene_id,
                    "N": scene_emb.size(0),
                    "num_valid_queries": num_valid,
                    "metrics": metrics,
                }
                per_scene.append(result)
                
                if self.is_main_process:
                    print(f"[eval] ({len(per_scene)}/{len(scene_ids)}) {scene_id}: N={result['N']}, valid={result['num_valid_queries']} -> "
                          + ", ".join([f"{k}:{v:.4f}" for k,v in result["metrics"].items() if k.startswith("recall")]))
                      
            except Exception as e:
                if self.is_main_process:
                    print(f"[eval] scene {scene_id} failed: {e}")
        
        return per_scene

    @torch.no_grad()
    def run(self, dump_json: Optional[str] = None) -> Dict[str, Any]:
        """
        对 split 内所有场景评估，并返回聚合结果：
          {
            "macro": {metrics...},      # 各场景 metrics 的简单平均
            "micro": {metrics...},      # 按"有效 query 数量"加权平均
            "global": {metrics...},     # 全局检索指标（高效计算）
            "per_scene": [ ... ],
            "elapsed_sec": float
          }
        """
        t0 = time.time()
        
        # 高效批量提取所有场景的特征
        all_emb, scene_overlaps, scene_ids, scene_offsets = self._embed_all_scenes_efficient()
        all_emb = all_emb.cpu()
        
        # 同步所有进程（如果使用DDP）
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        # 仅主进程进行评估
        if not self.is_main_process:
            return {"skipped": True, "reason": "non-main process"}
        
        # 全局检索评估（高效版本）
        global_result = self.evaluate_global_efficient(all_emb, scene_overlaps, scene_offsets)
        print(f"[global eval] total_images={global_result['total_images']}, "
              f"scenes={global_result['num_scenes']}, valid_queries={global_result['num_valid_queries']}")
        print("Global metrics: " + ", ".join([f"{k}:{v:.4f}" for k,v in global_result["global_metrics"].items() if k.startswith("recall")]))
        
        # 分场景评估（高效版本）
        per_scene = self.evaluate_per_scene_efficient(all_emb, scene_overlaps, scene_ids, scene_offsets)
        
        if not per_scene:
            raise RuntimeError("没有任何场景评估成功，请检查数据路径或依赖。")

        # 聚合：macro（简单平均）
        keys = list(per_scene[0]["metrics"].keys())
        macro = {k: float(np.mean([s["metrics"][k] for s in per_scene])) for k in keys}

        # 聚合：micro（按有效 query 数量加权）
        weights = np.array([s["num_valid_queries"] for s in per_scene], dtype=np.float64)
        weights = np.maximum(weights, 1)  # 防 0
        micro = {
            k: float(np.average([s["metrics"][k] for s in per_scene], weights=weights))
            for k in keys
        }

        out = {
            "macro": macro,
            "micro": micro,
            "global": global_result["global_metrics"],
            "total_images": global_result["total_images"],
            "num_scenes": global_result["num_scenes"],
            "num_valid_queries": global_result["num_valid_queries"],
            "per_scene": per_scene,
            "elapsed_sec": round(time.time() - t0, 2),
        }

        if dump_json:
            os.makedirs(os.path.dirname(dump_json), exist_ok=True)
            with open(dump_json, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        return out


# -----------------------------
# 简易 CLI
# -----------------------------
def _load_encoder_from_orbit_cfg(orbit_cfg_path: str) -> nn.Module:
    """
    如果你有 create_orbit(cfg) 接口，可在 CLI 模式下用它构建 encoder；
    这里示例用动态 import。
    """
    import yaml
    from .orbit import create_orbit
    with open(orbit_cfg_path, "r", encoding="utf-8") as f:
        ocfg = yaml.safe_load(f)
    enc = create_orbit(ocfg)
    return enc


def main_cli():
    """
    命令行示例：
    python -m orbit.eval \
      --root_dir /path/to/data \
      --split_txt /path/to/dataset_split/gl3d/val.txt \
      --orbit_cfg /path/to/orbit_cfg.yaml \
      --save_embeds \
      --embeds_dir /tmp/embeds_val \
      --device cuda \
      --dump_json /tmp/val_metrics.json
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--split_txt", type=str, required=True)
    parser.add_argument("--orbit_cfg", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=322)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pos_th", type=float, default=0.3)
    parser.add_argument("--ks", type=str, default="1,5,10,20")
    parser.add_argument("--save_embeds", action="store_true")
    parser.add_argument("--embeds_dir", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--dump_json", type=str, default=None)
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))

    encoder = _load_encoder_from_orbit_cfg(args.orbit_cfg)

    cfg = EvalConfig(
        root_dir=args.root_dir,
        split_txt=args.split_txt,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        pos_th=args.pos_th,
        ks=ks,
        save_embeds=args.save_embeds,
        embeds_dir=args.embeds_dir,
        use_amp=args.use_amp,
    )

    evaluator = OrbitEvaluator(encoder, cfg)
    out = evaluator.run(dump_json=args.dump_json)
    print("\n===== EVAL SUMMARY =====")
    print(f"Elapsed: {out['elapsed_sec']}s")
    print("Macro:", json.dumps(out["macro"], indent=2))
    print("Micro:", json.dumps(out["micro"], indent=2))


if __name__ == "__main__":
    main_cli()
