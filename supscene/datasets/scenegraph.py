import os
from typing import List, Optional
import numpy as np
from dataclasses import dataclass


# =====================
# Utilities
# =====================
def read_lines(fp: str) -> List[str]:
    """Read non-empty, stripped lines from a text file.

    Args:
        fp (str): File path.

    Returns:
        List[str]: List of lines.
    """
    with open(fp, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]
    
@dataclass
class SceneMeta:
    """Lightweight container for scene metadata (optional, not required by loaders)."""

    scene_id: str
    image_paths: List[str]  # relative paths, following images.txt order
    overlaps_row: np.ndarray  # COO row indices
    overlaps_col: np.ndarray  # COO col indices
    overlaps_val: np.ndarray  # COO values in [0, 1]
    teacher_embs: Optional[np.ndarray] = None  # [N, D_t] (optional)


class SceneGraph:
    """Load per-scene graph: image list + sparse overlap matrix (COO).

    Notes:
        - Keeps original I/O structure; logic unchanged.
        - Accepts either images.txt or overlaps.npz["filenames"].
    """

    def __init__(self, scene_dir: str, teacher_name: str = "dinov2_g14_cls"):
        self.scene_dir = scene_dir
        self.images_dir = os.path.join(scene_dir, "images")
        self.images_txt = os.path.join(scene_dir, "images.txt")
        self.overlaps_npz = os.path.join(scene_dir, "overlaps.npz")
        self.teacher_dir = os.path.join(scene_dir, "teacher_embs")
        self.teacher_name = teacher_name
        self.teacher_npy = os.path.join(self.teacher_dir, f"{teacher_name}.npy")

        # 1) read image order
        if os.path.exists(self.images_txt):
            image_names = read_lines(self.images_txt)
        else:
            # if images.txt missing, try filenames field in overlaps.npz
            with np.load(self.overlaps_npz, allow_pickle=True) as coo:
                if "filenames" not in coo:
                    raise FileNotFoundError(f"images.txt missing and overlaps.npz lacks 'filenames': {self.overlaps_npz}")
                image_names = list(map(str, coo["filenames"]))

        self.N = len(image_names)
        self.image_paths = [os.path.join(self.images_dir, nm) for nm in image_names]

        # 2) read sparse overlaps (keys: row, col, weight or val)
        with np.load(self.overlaps_npz, allow_pickle=True) as coo:
            row = coo["row"].astype(np.int64)
            col = coo["col"].astype(np.int64)
            if "weight" in coo:
                val = coo["weight"].astype(np.float32)
            elif "val" in coo:
                val = coo["val"].astype(np.float32)
            else:
                raise KeyError(f"overlaps.npz missing 'weight'/'val' key: {self.overlaps_npz}")

        # symmetrize (safe if already symmetric) and remove self-loops
        row_sym = np.concatenate([row, col])
        col_sym = np.concatenate([col, row])
        val_sym = np.concatenate([val, val])
        keep = row_sym != col_sym
        self.row = row_sym[keep]
        self.col = col_sym[keep]
        self.val = np.clip(val_sym[keep], 0.0, 1.0)

        # 3) Teacher features (optional, memory-mapped)
        self.teacher: Optional[np.ndarray] = None
        if os.path.exists(self.teacher_npy):
            arr = np.load(self.teacher_npy, mmap_mode="r")
            if arr.dtype != np.float32:
                raise TypeError(
                    f"{self.teacher_npy} dtype={arr.dtype}, save as float32 to enable mmap"
                )
            self.teacher = arr
            assert (
                self.teacher.shape[0] == self.N
            ), f"Teacher feat count {self.teacher.shape[0]} != image count {self.N}"

    def dense_overlap(self, idx: np.ndarray, add_self: bool = True) -> np.ndarray:
        """Extract dense subgraph overlaps.

        Args:
            idx (np.ndarray): Node indices, shape (n,).
            add_self (bool): If True, fill diagonal with 1.

        Returns:
            np.ndarray: Dense overlap matrix O of shape (n, n), dtype float32.
        """
        n = idx.shape[0]
        map_idx = -np.ones(self.N, dtype=np.int64)
        map_idx[idx] = np.arange(n, dtype=np.int64)

        mu = map_idx[self.row]
        mv = map_idx[self.col]
        sel = (mu >= 0) & (mv >= 0)
        mu, mv = mu[sel], mv[sel]
        w = self.val[sel]

        O = np.zeros((n, n), dtype=np.float32)
        if mu.size > 0:
            # Max to guard against duplicated edges
            O[mu, mv] = np.maximum(O[mu, mv], w)
        if add_self:
            np.fill_diagonal(O, 1.0)
        return O

    def neighbor_lists(self, iou_th: float = 0.2) -> List[List[int]]:
        """Adjacency as neighbor lists under IoU threshold.

        Args:
            iou_th (float): IoU threshold.

        Returns:
            List[List[int]]: For each node u, list of neighbors v.
        """
        adj: List[List[int]] = [[] for _ in range(self.N)]
        sel = self.val >= iou_th
        for u, v in zip(self.row[sel], self.col[sel]):
            adj[u].append(v)
        return adj
