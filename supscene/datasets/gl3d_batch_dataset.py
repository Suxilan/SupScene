"""
GL3D批量场景数据集 - 用于高效评估的批量场景图像加载
"""
import os
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .gl3d_subgraph_dataset import SceneGraph


class GL3DBatchDataset(Dataset):
    """
    批量GL3D场景数据集 - 一次性加载多个场景的所有图像
    避免每个场景都创建新的Dataset/DataLoader的开销
    """
    
    def __init__(self, scene_dirs: List[str], img_size: int = 322):
        """
        Args:
            scene_dirs: 场景目录列表
            img_size: 图像尺寸，默认322x322
        """
        self.scene_dirs = scene_dirs
        self.img_size = img_size
        
        # 构建全局图像路径列表和场景映射
        self.image_paths = []
        self.scene_info = []  # [(scene_id, start_idx, end_idx, overlap_matrix), ...]
        
        current_idx = 0
        for scene_dir in scene_dirs:
            try:
                G = SceneGraph(scene_dir)
                scene_id = os.path.basename(scene_dir)
                paths = G.image_paths
                
                # 记录场景信息
                start_idx = current_idx
                end_idx = current_idx + len(paths)
                
                # 获取重叠矩阵
                idx = np.arange(G.N, dtype=np.int64)
                overlap_matrix = G.dense_overlap(idx, add_self=True)  # [N,N] np.float32
                
                self.scene_info.append((scene_id, start_idx, end_idx, overlap_matrix))
                self.image_paths.extend(paths)
                current_idx = end_idx
                
            except Exception as e:
                print(f"Warning: Failed to load scene {scene_dir}: {e}")
                continue
        
        # 使用albumentations进行预处理
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        print(f"GL3DBatchDataset: {len(self.scene_info)} scenes, {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx: 全局图像索引
            
        Returns:
            torch.Tensor: 预处理后的图像张量 [3, H, W]
        """
        image_path = self.image_paths[idx]
        
        try:
            # 使用cv2读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # cv2默认读取为BGR格式，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用albumentations变换
            transformed = self.transform(image=image)
            return transformed['image']
            
        except Exception as e:
            # 如果图像加载失败，返回零张量
            print(f"Warning: Failed to load image {image_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size)
    
    def get_scene_info(self) -> List[Tuple[str, int, int, np.ndarray]]:
        """
        返回场景信息列表
        Returns:
            List[(scene_id, start_idx, end_idx, overlap_matrix), ...]
        """
        return self.scene_info
    
    def get_scene_slice(self, scene_idx: int) -> Tuple[str, slice, np.ndarray]:
        """
        获取指定场景的切片信息
        Args:
            scene_idx: 场景索引
        Returns:
            (scene_id, slice_obj, overlap_matrix)
        """
        scene_id, start_idx, end_idx, overlap_matrix = self.scene_info[scene_idx]
        return scene_id, slice(start_idx, end_idx), overlap_matrix
