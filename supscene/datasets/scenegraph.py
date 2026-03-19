import os
import json
from typing import List, Optional, Tuple
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
        - Supports gt.npz/overlap.npz using rows/cols/vals or legacy row/col/val names.
        - Reads the image order from images.txt, image_index.json, or npz metadata.
    """

    def __init__(self, scene_dir: str, teacher_name: str = "dinov2_g14_cls"):
        self.scene_dir = scene_dir
        self.images_dir = os.path.join(scene_dir, "images")
        self.images_txt = os.path.join(scene_dir, "images.txt")
        self.image_index_file = os.path.join(scene_dir, "image_index.json")
        self.gt_npz = self._find_gt()
        # teacher fields (preserve original public API)
        self.teacher_dir = os.path.join(scene_dir, "teacher_embs")
        self.teacher_name = teacher_name
        self.teacher_npy = os.path.join(self.teacher_dir, f"{teacher_name}.npy")

        # Load metadata and sparse overlap matrix
        with np.load(self.gt_npz, allow_pickle=True) as data:
            # 1. Image names / order
            self.image_names = self._load_image_names(data)
            self.image_paths = [os.path.join(self.images_dir, nm) for nm in self.image_names]
            self.N = len(self.image_names)

            # 2. Load sparse matrix (COO)
            row = self._read_npz_field(data, ["row", "rows"], np.int64)
            col = self._read_npz_field(data, ["col", "cols"], np.int64)
            val = self._read_npz_field(data, ["weight", "val", "vals"], np.float32)

        # 3. Symmetrize and clean
        rows = np.concatenate([row, col])
        cols = np.concatenate([col, row])
        vals = np.concatenate([val, val])
        mask = rows != cols
        self.row = rows[mask].astype(np.int64)
        self.col = cols[mask].astype(np.int64)
        self.val = np.clip(vals[mask].astype(np.float32), 0.0, 1.0)

        # 4) Teacher features (optional, memory-mapped) — preserve original behavior
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

    def _find_gt(self) -> str:
        # prefer gt.npz, but accept overlaps.npz or overlap.npz for backward compatibility
        candidates = ["gt.npz", "overlaps.npz", "overlap.npz"]
        for fn in candidates:
            path = os.path.join(self.scene_dir, fn)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"missing gt/overlaps npz in {self.scene_dir}; tried: {candidates}")

    def _read_npz_field(self, data, candidates: List[str], dtype) -> np.ndarray:
        for key in candidates:
            if key in data:
                return data[key].astype(dtype)
        # fallback heuristic: any 1D numeric array
        for k in data.files:
            arr = data[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                try:
                    return arr.astype(dtype)
                except Exception:
                    continue
        raise KeyError(f"Missing fields {candidates} in {self.gt_npz}")

    def _load_image_names(self, npz_data) -> List[str]:
        # Strategy 1: images.txt
        if os.path.exists(self.images_txt):
            return read_lines(self.images_txt)

        # Strategy 2: image_index.json
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            if isinstance(index, list):
                return [str(x) for x in index]
            if isinstance(index, dict):
                if "id_to_name" in index:
                    return [str(x) for x in index["id_to_name"]]
                if "name_to_id" in index:
                    items = sorted(index["name_to_id"].items(), key=lambda x: int(x[1]))
                    return [str(k) for k, _ in items]

        # Strategy 3: embedded in NPZ
        for key in ["images", "filenames"]:
            if key in npz_data:
                return [str(x) for x in npz_data[key]]

        # Strategy 4: directory listing fallback
        if os.path.exists(self.images_dir):
            return sorted([x for x in os.listdir(self.images_dir) if x.lower().endswith(('.jpg', '.png'))])

        raise FileNotFoundError(f"Could not determine image order in {self.scene_dir}")

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

    def fetch_nodes(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch all nodes and their overlap weights with node idx.

        Returns:
            all_nodes: array of node indices excluding `idx` (shape [N-1])
            weights: array of overlap weights for each node in `all_nodes` (shape [N-1])
        """
        mask = self.row == idx
        ov_map = {int(n): float(w) for n, w in zip(self.col[mask], self.val[mask])}

        all_nodes = np.delete(np.arange(self.N, dtype=np.int64), idx)
        weights = np.array([ov_map.get(int(n), 0.0) for n in all_nodes], dtype=np.float32)

        return all_nodes, weights
