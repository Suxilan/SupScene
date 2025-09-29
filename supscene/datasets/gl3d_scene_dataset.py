"""
GL3D场景级数据集 - 用于评估时的单场景图像加载
"""
import os
from typing import List
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GL3DSceneDataset(Dataset):
    """
    单个GL3D场景的图像数据集
    用于评估时批量加载场景内所有图像
    """
    
    def __init__(self, image_paths: List[str], img_size: int = 322):
        """
        Args:
            image_paths: 图像路径列表
            img_size: 图像尺寸，默认322x322
        """
        self.image_paths = image_paths
        self.img_size = img_size
        
        # 使用albumentations进行预处理
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx: 图像索引
            
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
