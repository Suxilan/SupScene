"""
OrbitTrainer - Orbit任务专用训练器
继承BaseTrainer，集成TaskManager进行多任务训练
"""
from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from utils import BaseTrainer
from utils import compute_batch_retrieval_metrics
from .taskmanager import TaskManager

# ConFIG imports
try:
    from conflictfree.grad_operator import ConFIG_update
    from conflictfree.utils import get_gradient_vector, apply_gradient_vector
    CONFIG_AVAILABLE = True
except ImportError:
    print("[Warning] ConFIG not available, using standard multi-task training")
    CONFIG_AVAILABLE = False
from .eval import EvalConfig, OrbitEvaluator


class OrbitTrainer(BaseTrainer):
    """
    Orbit 任务训练器：
    - forward_step: 负责 images/z -> z(B,N,D) + heads 前向
    - compute_loss: 负责调用 TaskManager 的 losses（多任务加权）
    - compute_metrics: 训练中按批快速算检索指标（可关）
    - 支持ConFIG冲突消解的多任务优化
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 task_manager: TaskManager,
                 root_dir: str,
                 split_txt: str,
                 metric_pos_th: float = 0.3,
                 metric_ks: tuple = (1, 5, 10),
                 use_conflictfree: bool = False,
                 **kwargs):
        # 训练指标配置
        self.root_dir = root_dir
        self.split_txt = split_txt
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
        
        self.logger.info(f"[OrbitTrainer] 初始化完成")
        self.logger.info(f"  ConFIG冲突消解: {self.use_conflictfree}")
        self.logger.info(f"  指标阈值: {self.metric_pos_th}")
        self.logger.info(f"  指标ks: {self.metric_ks}")
   # -------------------------
    # 1) 前向：images/z -> z -> heads
    # -------------------------
    def _data_wrapper(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        输入：
          batch: batch
        输出：
          z: [B,N,D]
        """
        images = batch["images"]  # [B,N,3,H,W] 或 [B,3,H,W]
        if images.dim() == 4:
            # 无 N 维，视为 N=1
            images = images.unsqueeze(1)
        B, N = images.shape[:2]

        node_mask = batch.get("node_mask", None)
        if node_mask is not None:
            mask = node_mask.view(-1).bool()         # [B*N]
            x = images.view(B * N, *images.shape[2:])[mask]  # [num_valid, 3, H, W]
            if x.numel() == 0:
                raise RuntimeError("No valid nodes found in node_mask.")
        else:
            mask = None
            x = images.view(B * N, *images.shape[2:])        # [B*N, 3, H, W]

        return B, N, mask, x
    
    def _data_unwrapper(self, B: int, N: int, mask: torch.Tensor, z_valid: torch.Tensor) -> torch.Tensor:
        """
        输入：
          B: 批次大小
          N: 节点数
          mask: [B*N] 的有效节点掩码，None表示全部有效
          z_valid: [num_valid,...] 的特征
        输出：
          z: [B,N,...] 的特征
        """
        if mask is None:
            # 全部有效，直接reshape
            return z_valid.view(B, N, *z_valid.shape[1:])
        
        # 创建全零张量，保持除第一维外的所有维度
        output_shape = (B * N,) + z_valid.shape[1:]
        z_flat = torch.zeros(output_shape, device=z_valid.device, dtype=z_valid.dtype)
        z_flat[mask] = z_valid
        z = z_flat.view(B, N, *z_valid.shape[1:])
        return z
    
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入：
          batch["images"]: [B,N,3,H,W] 或 [B,3,H,W]（可选）
          batch["teacher"]: [B,N,D]（可选，预缓存）
        输出：
          {
            "z": [B,N,D],                        # 部署向量
            "task_outputs": {name: tensor,...},  # 各 head 输出（q/P/…）
          }
        """
        # 1) 取 z
        B, N, mask, x = self._data_wrapper(batch)
        z_valid = self.model(x)
        z = self._data_unwrapper(B, N, mask, z_valid)

        # 2) 通过 TaskManager 的 heads
        task_outs = self.task_manager.forward_heads(z)  # {name: tensor([B,N,*])}

        return {"z": z, "task_outputs": task_outs}

    # -------------------------
    # 2) 损失：多任务加权汇总
    # -------------------------
    def compute_loss(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        读取 batch 的结构化标签，调用 TaskManager.compute_loss 做汇总。
        需要的 batch 字段：
          - overlap: [B,N,N]
          - pair_mask(可选): [B,N,N]，若无则由 node_mask 构造
          - node_mask(可选): [B,N]
          - teacher(可选): [B,N,D]
        """
        outs = model_outputs["task_outputs"]
        overlap = batch["overlap"]
        pair_mask = batch.get("pair_mask", None)
        node_mask = batch.get("node_mask", None)
        teacher_features = batch.get("teacher", None)
        
        # 准备损失计算所需的数据
        if self.use_ema and teacher_features is None:
            B, N, mask, x = self._data_wrapper(batch)
            z_flat = self.get_ema_features(x)
            teacher_features = self._data_unwrapper(B, N, mask, z_flat)
        
        # 调用TaskManager计算损失
        losses = self.task_manager.compute_loss(
            outputs=outs,
            overlap=overlap,
            pair_mask=pair_mask,
            node_mask=node_mask,
            teacher_features=teacher_features,
        )
        
        return losses

    # -------------------------
    # 3) 训练时的批次级检索指标（可选）
    # -------------------------
    def compute_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        使用批次级指标实现（向量化）：
          compute_batch_retrieval_metrics(similarity, overlap_gt, node_mask, ks, threshold)
        这里的 similarity 用 z 计算：cosine(z,z)。
        """
        z = model_outputs["z"]  # [B,N,D]
        
        overlap = batch["overlap"]
        node_mask = batch.get("node_mask", None)
        if node_mask is not None:
            node_mask = node_mask   
        
        try:
            metrics = compute_batch_retrieval_metrics(
                emb=z,
                overlap_gt=overlap,
                node_mask=node_mask,
                ks=self.metric_ks,
                pos_th=self.metric_pos_th
            )
            return metrics
        except Exception as e:
            self.logger.error(f"[OrbitTrainer] 指标计算失败: {e}")
            return {}
        
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.task_manager.heads.train()
        
        # 前向传播
        model_outputs = self.forward_step(batch)
        
        # 计算损失
        loss_dict = self.compute_loss(batch, model_outputs)
        
        # 反向传播 - 支持ConFIG冲突消解
        if self.use_conflictfree and CONFIG_AVAILABLE:
            self._conflictfree_backward_step(loss_dict)
        else:
            self._standard_backward_step(loss_dict)
        
        # # 优化器步骤
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        
        # 调度器步骤
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 更新EMA参数
        if self.use_ema:
            if self.accelerator is not None: self.accelerator.wait_for_everyone()
            
            self.update_ema()
            
            if self.accelerator is not None: self.accelerator.wait_for_everyone() 
        
        # 计算指标
        metrics = self.compute_metrics(batch, model_outputs)
        
        # 合并损失和指标
        result = {f"loss/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in loss_dict.items()}
        result.update({f"metric/{k}": v for k, v in metrics.items()})
        
        torch.cuda.empty_cache()
        del model_outputs, loss_dict, metrics, batch
        
        return result
    
    @torch.no_grad()
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步验证"""
        self.model.eval()
        self.task_manager.heads.eval()
        
        # 前向传播
        model_outputs = self.forward_step(batch)
        
        # 计算损失
        loss_dict = self.compute_loss(batch, model_outputs)
        
        # 计算指标
        metrics = self.compute_metrics(batch, model_outputs)
        
        # 合并损失和指标
        result = {f"loss/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in loss_dict.items()}
        result.update({f"metric/{k}": v for k, v in metrics.items()})
        
        torch.cuda.empty_cache()
        del model_outputs, loss_dict, metrics, batch
        
        return result
    
    def _standard_backward_step(self, loss_dict: Dict[str, torch.Tensor]):
        """标准反向传播步骤"""
        total_loss = loss_dict['total_loss']
        
        if self.use_accelerate:
            # 使用accelerator处理梯度累积和反向传播
            self.accelerator.backward(total_loss)
            
            if self.accelerator.sync_gradients:
                # 梯度裁剪
                if self.grad_clip is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.accelerator.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
        else:
            # 标准训练流程
            if self.grad_accum_steps > 1:
                total_loss = total_loss / self.grad_accum_steps
            
            total_loss.backward()
            
            # 梯度累积
            if (self.global_step) % self.grad_accum_steps == 0:
                # 梯度裁剪
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
                    
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def _conflictfree_backward_step(self, loss_dict: Dict[str, torch.Tensor]):
        """ConFIG冲突消解反向传播步骤"""
        
        # 获取unwrapped模型对象（ConFIG需要模型对象而非参数列表）
        if self.use_accelerate:
            model = self.accelerator.unwrap_model(self.model)
            task_heads = self.accelerator.unwrap_model(self.task_manager.heads)
        else:
            model = self.model
            task_heads = self.task_manager.heads
            
        # 创建包含所有参数的模块容器（用于ConFIG梯度操作）
        class CombinedModule(nn.Module):
            def __init__(self, model, heads):
                super().__init__()
                self.model = model
                self.heads = heads
        
        combined_module = CombinedModule(model, task_heads)
               
        # 从loss_dict中提取各任务损失
        # 仅保留以 _loss 结尾的标量损失，且排除 total_loss
        task_losses = {}
        for key, loss in loss_dict.items():
            if not (isinstance(key, str) and key.endswith('_loss')):
                continue
            if key.lower() == 'total_loss':
                continue
            if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                task_losses[key] = loss
        
        # 计算每个任务的梯度
        task_gradients = []
        task_names = list(task_losses.keys())
        for i, task_name in enumerate(task_names):
            task_loss = task_losses[task_name]
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 反向传播计算梯度
            if self.use_accelerate:
                retain_graph = (i < len(task_names) - 1)
                self.accelerator.backward(task_loss, retain_graph=retain_graph)
            else:
                # 多次 backward 同一前向图：非最后一次需要保留计算图
                retain_graph = (i < len(task_names) - 1)
                task_loss.backward(retain_graph=retain_graph)
            
            # 获取当前任务的梯度向量（从combined_module获取）
            task_grad = get_gradient_vector(combined_module, none_grad_mode='zero')
            task_gradients.append(task_grad)
        
        # 使用ConFIG计算冲突消解梯度
        if len(task_gradients) > 1:
            conflict_free_grad = ConFIG_update(task_gradients)
        else:
            conflict_free_grad = task_gradients[0] if task_gradients else None
        
        # 应用冲突消解梯度（最简单累计：按标准路径缩放后每步更新）
        if conflict_free_grad is not None:
            # 简单梯度累计：非accelerate路径下按步缩放，保持与_standard_backward_step一致
            if not self.use_accelerate and self.grad_accum_steps > 1:
                conflict_free_grad = conflict_free_grad / float(self.grad_accum_steps)

            # 确保仅应用ConFIG后的梯度：避免叠加最后一次任务的原始梯度
            self.optimizer.zero_grad()
            apply_gradient_vector(combined_module, conflict_free_grad, none_grad_mode='zero')
            
            # 梯度裁剪
            if self.grad_clip is not None:
                if self.use_accelerate:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.accelerator.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.task_manager.heads.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def set_conflictfree(self, use_conflictfree: bool):
        """设置是否使用ConFIG冲突消解"""
        self.use_conflictfree = use_conflictfree and CONFIG_AVAILABLE
        self.logger.info(f"[OrbitTrainer] ConFIG冲突消解: {self.use_conflictfree}")
    
    def get_conflictfree_status(self) -> bool:
        """获取ConFIG冲突消解状态"""
        return self.use_conflictfree
    
    # -------------------------
    # 5) 检查点保存/加载支持TaskManager
    # -------------------------
    def save_checkpoint(self, is_best: bool = False, epoch: int = None, suffix: str = ""):
        """保存检查点，包含TaskManager状态"""
        if epoch is not None:
            self.current_epoch = epoch
            if suffix == "interrupted":
                self.current_epoch -= 1 
                
        if self.use_accelerate:           
            # 准备额外状态信息
            extra_state = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'train_metrics': dict(self.train_metrics),
                'val_metrics': dict(self.val_metrics)
            }
            
            # 保存wandb run id用于恢复
            if self.log_with == "wandb" and hasattr(self.accelerator, 'trackers'):
                for tracker in self.accelerator.trackers:
                    if hasattr(tracker, 'run') and hasattr(tracker.run, 'id'):
                        extra_state['wandb_run_id'] = tracker.run.id
                        break
            
            # 保存EMA状态
            if self.use_ema and self.ema is not None:
                extra_state['ema_state_dict'] = self.ema.state_dict()
            
            # TaskManager的heads已经通过accelerate prepare，会被save_state自动保存
            
            # 使用accelerate保存完整状态
            checkpoint_dir = Path("checkpoints") / ("last" if not suffix else f"last_{suffix}")
            self.accelerator.save_state(
                output_dir=str(self.output_dir / checkpoint_dir),
                safe_serialization=True
            )
            
            # 保存额外状态信息
            if self.use_accelerate and self.accelerator.is_main_process:
                torch.save(extra_state, self.output_dir / checkpoint_dir / "extra_state.pth")
            
            # 保存最佳模型
            if is_best:
                # 使用accelerator.save_model保存最佳模型
                self.accelerator.save_model(
                    self.model,
                    save_directory=str(self.output_dir / "best"),
                    safe_serialization=True
                )
            
            self.logger.info(f"Checkpoint saved at epoch {self.current_epoch}")
        else:
            # 非accelerate模式的原有逻辑
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'best_metric': self.best_metric,
                'train_metrics': dict(self.train_metrics),
                'val_metrics': dict(self.val_metrics)
            }
            
            if self.scheduler is not None:
                checkpoint['scheduler'] = self.scheduler.state_dict()
            
            # 保存EMA状态
            if self.use_ema and self.ema is not None:
                checkpoint['ema'] = self.ema.state_dict()
                
            # 保存TaskManager状态
            if self.task_manager is not None:
                checkpoint['task_heads'] = self.task_manager.state_dict()
            
            # 保存最新检查点
            if suffix:
                torch.save(checkpoint, self.output_dir / f"last_{suffix}.pth")
            else:
                torch.save(checkpoint, self.output_dir / "last.pth")
            
            # 保存最佳检查点
            if is_best:
                best_checkpoint = {
                    'epoch': self.current_epoch,
                    'global_step': self.global_step,
                    'model': model_state_dict,
                    'best_metric': self.best_metric,
                    'train_metrics': dict(self.train_metrics),
                    'val_metrics': dict(self.val_metrics)
                }
                torch.save(best_checkpoint, self.output_dir / "best.pth")
            
            self.logger.info(f"Checkpoint saved at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点，包含TaskManager状态"""
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        
        if self.use_accelerate:
            # 检查是否为accelerate格式的检查点目录
            if checkpoint_path.is_dir() and (checkpoint_path / "model.safetensors").exists():
                # 新格式：accelerate保存的检查点目录
                self.accelerator.load_state(str(checkpoint_path))
                
                # 加载额外状态信息
                extra_state_path = checkpoint_path / "extra_state.pth"
                if extra_state_path.exists():
                    extra_state = torch.load(extra_state_path, map_location=self.device)
                    
                    # 加载EMA状态
                    if self.use_ema and self.ema is not None and 'ema_state_dict' in extra_state:
                        self.ema.load_state_dict(extra_state['ema_state_dict'])
                        self.logger.info("EMA state loaded from checkpoint")
                    
                    # TaskManager的heads已经通过accelerate prepare，会被load_state自动加载
                    
                    # 加载训练状态
                    from collections import defaultdict
                    self.current_epoch = extra_state.get('epoch', 1) + 1
                    self.global_step = extra_state.get('global_step', 1)
                    self.best_metric = extra_state.get('best_metric')
                    self.train_metrics = defaultdict(list, extra_state.get('train_metrics', {}))
                    self.val_metrics = defaultdict(list, extra_state.get('val_metrics', {}))
                else:
                    self.logger.warning("Extra state file not found, using default values")
            else:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        else:
            # 非accelerate模式的原有逻辑
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型
            self.model.load_state_dict(checkpoint['model'])
            
            # 加载优化器
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 加载调度器
            if self.scheduler is not None and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # 加载EMA状态
            if self.use_ema and self.ema is not None and 'ema' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema'])
                self.logger.info("EMA state loaded from checkpoint")
            
            # 加载TaskManager状态
            if self.task_manager is not None and 'task_heads' in checkpoint:
                self.task_manager.load_state_dict(checkpoint['task_heads'])
                self.logger.info("TaskManager state loaded from checkpoint")
            
            # 加载训练状态
            self.current_epoch = checkpoint.get('epoch', 1) + 1
            self.global_step = checkpoint.get('global_step', 1)
            self.best_metric = checkpoint.get('best_metric')
            
            from collections import defaultdict
            self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
            self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
        return self.current_epoch
    
    def evaluate(self, epoch: int):
        """评估"""
        cfg = EvalConfig(
            root_dir=self.root_dir,
            split_txt=self.split_txt,
            img_size=322,
            batch_size=64,
            num_workers=4,
            device=self.device,
            pos_th=self.metric_pos_th,
            ks=self.metric_ks,
            save_embeds=False,
            global_retrieval=True,
        )
        self.model.eval()
        eval_ema = False
        if self.use_ema and eval_ema:
            with self.get_ema_model_for_inference():
                evaluator = OrbitEvaluator(self.model, cfg)
        else:
            evaluator = OrbitEvaluator(self.model, cfg)
        
        self.logger.info(f"总场景数量: {len(evaluator.scene_dirs)}")
        
        # 运行完整评估
        self.logger.info("开始完整评估...")
        result = evaluator.run(dump_json=f"{self.output_dir}/eval_metrics/epoch-{epoch}.json")
        
        self.logger.info("\n✅ 评估完成!")
        self.logger.info(f"  - 耗时: {result.get('elapsed_sec', 'N/A')}秒")
        self.logger.info(f"  - 总图像数: {result.get('total_images', 'N/A')}")
        self.logger.info(f"  - 场景数量: {result.get('num_scenes', 'N/A')}")
        self.logger.info(f"  - 有效查询数: {result.get('num_valid_queries', 'N/A')}")

        # 记录评估指标
        eval_metrics = {"epoch": epoch}
        
        # 添加macro指标
        if "macro" in result:
            for k, v in result["macro"].items():
                if "map" in k.lower() or "recall" in k.lower():
                    eval_metrics[f"macro_{k}"] = v
        
        # 添加micro指标
        if "micro" in result:
            for k, v in result["micro"].items():
                if "map" in k.lower() or "recall" in k.lower():
                    eval_metrics[f"micro_{k}"] = v
        
        # 添加global指标
        if "global" in result:
            for k, v in result["global"].items():
                if "map" in k.lower() or "recall" in k.lower():
                    eval_metrics[f"global_{k}"] = v
        
        self._log_metrics(eval_metrics, self.global_step, "eval")
    
    def _extract_data(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """提取模型的推理数据，不包含绘图逻辑"""
        # 获取实际模型（处理DDP包装）
        actual_model = self.model
        if hasattr(self.model, 'module'):
            actual_model = self.model.module
        
        # 检查是否使用GAP聚合器
        aggregator = getattr(actual_model, 'aggregator', None)
        if aggregator is None:
            self.logger.warning("模型没有aggregator属性")
            return {}
        
        # 检查是否为GAP类型的聚合器
        aggregator_type = type(aggregator).__name__
        if aggregator_type != 'GaussianAnchoredPool' and \
           aggregator_type != 'MomentTokenPool' and \
           aggregator_type != 'AdaptiveGeMPool' and \
           aggregator_type != 'GeMPool'and \
           aggregator_type != 'NetVLAD'and \
           aggregator_type != 'BoQ'and \
           aggregator_type != 'SALAD'and \
           aggregator_type != 'DinoAttentionAggregator' and \
           aggregator_type != 'AttnAGG3d':
            self.logger.warning(f"当前聚合器类型为 {aggregator_type}")
            return {}
        
        try:
            actual_model.eval()
            with torch.no_grad():
                # 获取整个batch进行可视化
                B, N, mask, x = self._data_wrapper(batch)
                # 前向传播获取特征
                features = actual_model.backbone(x)
                # if isinstance(features, tuple):
                #     features = features[0]  # 取特征图，忽略CLS token
                
                # 获取GAP的注意力图和参数
                T, A, _, _, _ = actual_model.aggregator(features, return_maps=True)
                A = self._data_unwrapper(B, N, mask, A)
                # 转换为numpy并准备数据，取N维度的第一个
                A_np = A[:, 0].detach().cpu().numpy()  # [B, K, H, W]
                
                # 获取原始图像 [B, N, 3, H_img, W_img]，取N维度的第一个
                images = batch["images"]
                if images.dim() == 4:
                    images = images.unsqueeze(1)
                orig_imgs = images[:, 0].detach().cpu()  # [B, 3, H_img, W_img]
                
                # 使用ImageNet标准化参数还原
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                # 反标准化
                orig_imgs = orig_imgs * std + mean
                orig_imgs = torch.clamp(orig_imgs, 0, 1)
                
                # 转换为[0, 255] numpy数组 [B, 3, H, W]
                orig_imgs = (orig_imgs * 255).numpy().astype(np.uint8)
                
                return {
                    'attention_maps': A_np,      # [B, K, H, W] 注意力图
                    'original_images': orig_imgs,  # [B, 3, H_img, W_img] 原始图像 [0, 255]
                    'epoch': self.current_epoch,
                    'K': A_np.shape[1]
                }
                
        except Exception as e:
            self.logger.error(f"GAP数据提取失败: {e}")
            return {}
    
    def visualize_model_components(self, epoch: int, batch, prefix: str = ""):
        """重写父类方法，实现GAP可视化"""
        try:
            # 提取GAP数据（纯推理，无绘图）
            gap_data = self._extract_data(batch)
            
            if gap_data:
                from supscene.viz_utils.viz_gaussian_heatmap import create_overlay_batch
                heatmap_tensor = create_overlay_batch(gap_data['original_images'], 
                                                      gap_data['attention_maps'], 
                                                      alpha=0.5,
                                                      colormap_name='magma') # plasma, viridis, inferno, magma
                
                if heatmap_tensor is not None:
                    # 记录图像到日志
                    # 提取batch作为键值
                    vis_images = {}
                    batch_size = heatmap_tensor.shape[0]
                    for i in range(batch_size):
                        vis_images[f'gap_heatmap_{i}'] = heatmap_tensor[i]
                    self._log_image(vis_images, self.global_step, prefix)
                    self.logger.info(f"GAP可视化已记录 - Epoch {epoch}")
            
        except Exception as e:
            self.logger.warning(f"可视化过程出错: {e}")
