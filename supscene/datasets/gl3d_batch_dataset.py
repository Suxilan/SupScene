import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .scenegraph import SceneGraph


class GL3DBatchDataset(Dataset):
    """Batch GL3D dataset — flattens multiple scenes into one index space.

    Args:
        scene_dirs: List of scene directories (…/GL3D/scene_id)
        img_size: Square resize target.
        transform: Albumentations pipeline. If None, a default IMAGENET norm pipeline is used.
        return_index: If True, `__getitem__` returns (image, idx); else returns image only.

    Notes:
        - `scene_info`: list of tuples (scene_id, start_idx, end_idx, O) with O∈[0,1]^{N×N}.
    """
    
    def __init__(
        self,
        scene_dirs: List[str],
        img_size: int = 322,
        transform: Optional[A.Compose] = None,
        return_index: bool = True,
    ):
        self.scene_dirs = scene_dirs
        self.img_size = int(img_size)
        self.return_index = bool(return_index)

        # Build flat index & scene slices
        self.image_paths: List[str] = []
        self.scene_info: List[Tuple[str, int, int, np.ndarray]] = []
        
        cur = 0
        for sdir in scene_dirs:
            try:
                G = SceneGraph(sdir)
                sid = os.path.basename(sdir)
                paths = G.image_paths
                n = len(paths)
                start, end = cur, cur + n
                idx = np.arange(G.N, dtype=np.int64)
                O = G.dense_overlap(idx, add_self=True)  # (N,N)
                self.scene_info.append((sid, start, end, O))
                self.image_paths.extend(paths)
                cur = end
            except Exception as e:
                print(f"[GL3DBatchDataset] warn: skip scene {sdir}: {e}")
                continue

        # transforms
        self.transform = transform or A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        print(f"GL3DBatchDataset: scenes={len(self.scene_info)} images={len(self.image_paths)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError("cv2.imread returned None")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            x = self.transform(image=im)["image"]  # (3,H,W)
        except Exception as e:
            print(f"[GL3DBatchDataset] warn: bad image {path}: {e}")
            x = torch.zeros(3, self.img_size, self.img_size)
        return (x, idx) if self.return_index else x

    # ------- scene helpers -------
    def get_scene_info(self) -> List[Tuple[str, int, int, np.ndarray]]:
        """Return [(scene_id, start, end, O), …]."""
        return self.scene_info

    def get_scene_slice(self, scene_idx: int) -> Tuple[str, slice, np.ndarray]:
        sid, s, e, O = self.scene_info[scene_idx]
        return sid, slice(s, e), O
