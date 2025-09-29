import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

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


class SubgraphSampler:
    """Subgraph sampler with uniform, anchor expansion, or balanced modes.

    Modes:
        - "uniform": uniform sampling without structure.
        - "anchor_expand": BFS-like expansion from a random anchor under IoU.
        - "balanced": greedy refinement to target a positive-pair ratio.
    """
    def __init__(
        self, 
        mode: str = "uniform", 
        iou_th: float = 0.0, 
        topk_per_hop: int = 32, 
        max_hops: int = 3,
        balanced_small_graph: bool = True, 
        small_graph_threshold: int = 16,
        target_positive_ratio: float = 0.5
    ):
        assert mode in ("uniform", "anchor_expand", "balanced"), f"Unknown mode: {mode}"
        self.mode = mode
        self.iou_th = iou_th
        self.topk_per_hop = topk_per_hop
        self.max_hops = max_hops
        self.balanced_small_graph = balanced_small_graph
        self.small_graph_threshold = small_graph_threshold
        self.target_positive_ratio = target_positive_ratio

    def sample(self, G: SceneGraph, n_sub: int) -> np.ndarray:
        """Sample a subgraph of size n_sub from scene graph G.

        Args:
            G (SceneGraph): Scene graph.
            n_sub (int): Number of nodes to sample (n).

        Returns:
            np.ndarray: Node indices of shape (n,), dtype int64.
        """
        N = G.N
        if n_sub >= N:
            return np.arange(N, dtype=np.int64)

        if (
            self.balanced_small_graph
            and n_sub <= self.small_graph_threshold
            and self.mode in ["uniform", "balanced"]
        ):
            return self._balanced_sample(G, n_sub)

        if self.mode == "uniform":
            return np.random.choice(N, size=n_sub, replace=False)
        elif self.mode == "balanced":
            return self._balanced_sample(G, n_sub)

        # anchor expansion based on adjacency
        adj = G.neighbor_lists(self.iou_th)
        anchor = np.random.randint(0, N)
        visited = set([int(anchor)])
        frontier = [int(anchor)]
        hops = 0
        while len(visited) < n_sub and len(frontier) > 0 and hops < self.max_hops:
            new_frontier = []
            for u in frontier:
                nbrs = adj[u]
                if len(nbrs) == 0:
                    continue
                if len(nbrs) > self.topk_per_hop:
                    nbrs = random.sample(nbrs, self.topk_per_hop)
                for v in nbrs:
                    if v not in visited:
                        visited.add(int(v))
                        new_frontier.append(int(v))
                        if len(visited) >= n_sub:
                            break
                if len(visited) >= n_sub:
                    break
            frontier = new_frontier
            hops += 1

        if len(visited) < n_sub:
            rest = [x for x in range(N) if x not in visited]
            more = np.random.choice(rest, size=(n_sub - len(visited)), replace=False)
            visited.update(more.tolist())

        idx = np.fromiter(visited, dtype=np.int64)
        np.random.shuffle(idx)
        return idx[:n_sub]

    def _balanced_sample(self, G: SceneGraph, n_sub: int) -> np.ndarray:
        """Greedy ratio balancing for positive/negative pairs under IoU.

        Args:
            G (SceneGraph): Scene graph.
            n_sub (int): Subgraph size.

        Returns:
            np.ndarray: Node indices (n,).
        """
        N = G.N
        if n_sub >= N:
            return np.arange(N, dtype=np.int64)

        # Precompute overlap dict for quick membership
        O_dict: Dict[tuple, float] = {}
        for u, v, val in zip(G.row, G.col, G.val):
            if val >= self.iou_th:
                O_dict[(u, v)] = val
                O_dict[(v, u)] = val

        def has_o(i: int, j: int) -> bool:
            return (i, j) in O_dict

        def pos_ratio(nodes: List[int]) -> float:
            if len(nodes) < 2:
                return 0.0
            p, tot = 0, 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    tot += 1
                    if has_o(nodes[i], nodes[j]):
                        p += 1
            return p / tot if tot > 0 else 0.0

        nodes = np.random.choice(N, size=n_sub, replace=False).tolist()
        r = pos_ratio(nodes)

        T = min(50, N)  # max iterations
        tol = 0.05
        for _ in range(T):
            if abs(r - self.target_positive_ratio) <= tol:
                break
            need_more_pos = r < self.target_positive_ratio
            unused = [i for i in range(N) if i not in nodes]
            if not unused:
                break

            best_imp = 0.0
            best_swap = None
            lim = min(20, len(unused))
            for ridx in range(len(nodes)):
                for cand in unused[:lim]:
                    trial = nodes.copy()
                    trial[ridx] = cand
                    r_new = pos_ratio(trial)
                    imp = (r_new - r) if need_more_pos else (r - r_new)
                    if imp > best_imp:
                        best_imp, best_swap = imp, (ridx, cand)

            if best_swap and best_imp > 1e-2:
                ridx, cand = best_swap
                nodes[ridx] = cand
                r = pos_ratio(nodes)
            else:
                break

        idx = np.array(nodes, dtype=np.int64)
        np.random.shuffle(idx)
        return idx


class GL3DSubgraphDataset(Dataset):
    """Return a subgraph per item (possibly from different scenes).

    Features:
        - Optional image loading (for fine-tuning encoders).
        - Optional teacher distillation (uses teacher_embs if present).
        - Per-epoch reshuffle/rebuild of sampling plan.
    """

    def __init__(
        self,
        root_dir: str,
        split_txt: str,
        n_sub: int = 256,
        sampler: Optional[SubgraphSampler] = None,
        load_images: bool = False,
        img_size: int = 322,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,  
        teacher_name: str = "dinov2_g14_cls",
        adaptive_sampling: bool = True,
        min_images_per_scene: int = 50,
    ):
        """Dataset constructor.

        Args:
            root_dir (str): Dataset root.
            split_txt (str): File with scene IDs, one per line.
            n_sub (int): Subgraph node count per sample.
            sampler (Optional[SubgraphSampler]): Sampler. Defaults to uniform.
            load_images (bool): If True, load and transform images.
            img_size (int): Square resize size (H=W=img_size).
            scenes_per_epoch (Optional[int]): Number of scenes per epoch.
            samples_per_scene (Optional[int]): Fixed samples per scene; if None and
                adaptive_sampling=True, per-scene counts are computed adaptively.
            teacher_name (str): Teacher embedding name to pick.
            adaptive_sampling (bool): Enable adaptive per-scene sampling counts.
            min_images_per_scene (int): Filter out scenes with fewer images.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split_list = read_lines(split_txt)
        self.scene_dirs = [os.path.join(root_dir, "GL3D", sid) for sid in self.split_list]
        self.teacher_name = teacher_name
        self.adaptive_sampling = adaptive_sampling
        self.min_images_per_scene = int(min_images_per_scene)

        scenes_all: List[SceneGraph] = [SceneGraph(d, teacher_name) for d in self.scene_dirs]
        if self.min_images_per_scene > 0:
            self.scenes: List[SceneGraph] = [s for s in scenes_all if s.N >= self.min_images_per_scene]
        else:
            self.scenes = scenes_all

        if len(self.scenes) == 0:
            raise ValueError(
                "No valid scenes found: min_images_per_scene={}.\
                Total scenes={}. Lower the threshold or check dataset.".format(
                    self.min_images_per_scene, len(scenes_all)
                )
            )

        self.n_sub = n_sub
        self.sampler = sampler or SubgraphSampler(mode="uniform")
        self.load_images = load_images
        self.img_size = img_size

        self.num_scenes = len(self.scenes)
        self.scenes_per_epoch = scenes_per_epoch if scenes_per_epoch is not None else self.num_scenes
        
        if samples_per_scene is None and adaptive_sampling:
            self.samples_per_scene_list = [self._adaptive_samples_per_scene(scene.N) for scene in self.scenes]
            self.samples_per_scene = 1  # unused in adaptive mode
        else:
            self.samples_per_scene = int(samples_per_scene or 1)
            self.samples_per_scene_list = None

        # Optional image transforms (cv2 + albumentations)
        self.tf = None
        self._cv2_ok = False
        if self.load_images:
            try:
                import cv2  
                import albumentations as A  
                from albumentations.pytorch import ToTensorV2  

                self._cv2_ok = True
            except Exception as e:
                raise ImportError(
                    "Install opencv-python and albumentations to use load_images=True"
                ) from e

            self.tf = A.Compose(
                [
                    A.Resize(height=self.img_size, width=self.img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

        # Build sampling plan for the epoch
        self._build_epoch_indices()
    
    def _adaptive_samples_per_scene(self, N_images: int, min_samples: int = 1, max_samples: int = 8) -> int:
        """Heuristic: larger scenes yield more subgraph samples.

        Args:
            N_images (int): Number of images in the scene (N).
            min_samples (int): Lower bound.
            max_samples (int): Upper bound.

        Returns:
            int: Number of samples for this scene.
        """
        N = int(N_images)
        if N <= self.n_sub:
            return int(min_samples)
        cov = N / float(self.n_sub)
        s = int(np.ceil(np.sqrt(cov) * 1.5))  # sqrt dampens growth
        return int(np.clip(s, min_samples, max_samples))

    def _build_epoch_indices(self) -> None:
        """Construct the list of scene indices for this epoch."""
        if self.samples_per_scene_list is not None:
            pool = random.sample(range(self.num_scenes), k=min(self.scenes_per_epoch, self.num_scenes))
            scene_indices: List[int] = []
            for i in pool:
                k = self.samples_per_scene_list[i]
                scene_indices.extend([i] * k)
            random.shuffle(scene_indices)
            self.epoch_scene_indices = scene_indices
            self.total_samples = len(scene_indices)
        else:
            self.total_samples = int(self.scenes_per_epoch) * int(self.samples_per_scene)
            reps = math.ceil(self.total_samples / max(1, self.num_scenes))
            scene_indices = (list(range(self.num_scenes)) * reps)[: self.total_samples]
            random.shuffle(scene_indices)
            self.epoch_scene_indices = scene_indices

    def reshuffle_epoch(
        self,
        epoch: Optional[int] = None,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,
    ) -> None:
        """Rebuild the sampling plan at epoch boundaries.

        Args:
            epoch (Optional[int]): Ignored here; hook for external schedulers.
            scenes_per_epoch (Optional[int]): Override scenes per epoch.
            samples_per_scene (Optional[int]): Override samples per scene; disables adaptive mode.
        """
        if scenes_per_epoch is not None:
            self.scenes_per_epoch = int(scenes_per_epoch)
        if samples_per_scene is not None:
            self.samples_per_scene = int(samples_per_scene)
            self.samples_per_scene_list = None  # disable adaptive when manually set
        self._build_epoch_indices()
    
    def get_teacher_names(self) -> List[str]:
        """List available teacher embedding names in the first scene directory."""
        if not self.scenes:
            return []
        teacher_dir = self.scenes[0].teacher_dir
        if not os.path.exists(teacher_dir):
            return []
        files = [f for f in os.listdir(teacher_dir) if f.endswith(".npy")]
        return [f[:-4] for f in files]
    
    def switch_teacher(self, teacher_name: str) -> None:
        """Switch teacher embedding type across scenes.

        Args:
            teacher_name (str): New teacher basename (without .npy).
        """
        self.teacher_name = teacher_name
        for scene in self.scenes:
            scene.teacher_name = teacher_name
            scene.teacher_npy = os.path.join(scene.teacher_dir, f"{teacher_name}.npy")
            if os.path.exists(scene.teacher_npy):
                arr = np.load(scene.teacher_npy, mmap_mode="r")
                if arr.dtype != np.float32:
                    raise TypeError(
                        f"{scene.teacher_npy} dtype={arr.dtype}, save as float32 to enable mmap"
                    )
                scene.teacher = arr
            else:
                scene.teacher = None

    def __len__(self):
        return self.total_samples

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        """Load and transform an RGB image as tensor.

        Args:
            path (str): Image file path.

        Returns:
            torch.Tensor: (3, H, W) float tensor.
        """
        assert self._cv2_ok, "cv2/albumentations not available"
        import cv2

        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert self.tf is not None, "Transforms not initialized; set load_images=True in dataset."
        return self.tf(image=im)["image"]

    def __getitem__(self, idx: int) -> Dict:
        """Assemble one sample consisting of a subgraph and optional images/teacher.

        Args:
            idx (int): Sample index.

        Returns:
            Dict: {
                'scene_id': str,
                'node_idx': LongTensor (n,),
                'overlap': FloatTensor (n, n),
                'image_paths': List[str],
                'images': FloatTensor (n, 3, H, W) or None,
                'teacher': FloatTensor (n, D_t) or None,
                'num_nodes': int,
            }
        """
        scene_idx = self.epoch_scene_indices[idx]
        G = self.scenes[scene_idx]
        node_idx = self.sampler.sample(G, self.n_sub)  # (n,)
        n = int(node_idx.shape[0])

        # Dense overlaps in the subgraph
        O_np = G.dense_overlap(node_idx, add_self=True)  # (n, n)

        # Optional teacher features
        T_np = None
        if G.teacher is not None:
            T_np = G.teacher[node_idx]  # (n, D_t)

        paths = [G.image_paths[i] for i in node_idx.tolist()]

        imgs = None
        if self.load_images:
            imgs = torch.stack([self._load_image_tensor(p) for p in paths], dim=0)  # (n, 3, H, W)

        return {
            "scene_id": os.path.basename(G.scene_dir),
            "node_idx": torch.from_numpy(node_idx).long(),
            "overlap": torch.from_numpy(O_np),
            "image_paths": paths,
            "images": imgs,
            "teacher": None if T_np is None else torch.from_numpy(T_np),
            "num_nodes": n,
        }


def make_pad_collate(diag_weight: Optional[float] = None):
    """Factory for a collate_fn with optional diagonal down-weighting.

    Args:
        diag_weight (Optional[float]): If None, no pair weights are returned;
            if a float x, returns 'pair_weight' with diagonal set to x (others 1).

    Returns:
        Callable: A collate function mapping a list of samples to a padded batch dict.
    """

    def pad_collate(batch: List[Dict]) -> Dict:
        B = len(batch)
        n_max = max(x["num_nodes"] for x in batch)

        images = None
        if batch[0]["images"] is not None:
            C, H, W = batch[0]["images"].shape[1:]
            images = torch.zeros(B, n_max, C, H, W)

        O = torch.zeros(B, n_max, n_max)
        T = None
        if batch[0]["teacher"] is not None:
            D_t = batch[0]["teacher"].shape[-1]
            T = torch.zeros(B, n_max, D_t)

        M_n = torch.zeros(B, n_max, dtype=torch.bool)
        M_p = torch.zeros(B, n_max, n_max, dtype=torch.bool)
        P_w = None
        if diag_weight is not None:
            P_w = torch.ones(B, n_max, n_max, dtype=torch.float32)

        scene_ids: List[str] = []
        image_paths: List[List[str]] = []

        for b, item in enumerate(batch):
            n = item["num_nodes"]
            scene_ids.append(item["scene_id"])
            image_paths.append(item["image_paths"])

            O[b, :n, :n] = item["overlap"]
            M_n[b, :n] = True
            M_p[b, :n, :n] = True

            if images is not None:
                images[b, :n] = item["images"]
            if T is not None and item["teacher"] is not None:
                T[b, :n] = item["teacher"]
            if P_w is not None:
                P_w[b, :n, :n].fill_(1.0)
                diag_idx = torch.arange(n)
                P_w[b, diag_idx, diag_idx] = float(diag_weight)

        out = {
            "scene_ids": scene_ids,
            "image_paths": image_paths,
            "overlap": O,  # (B, n_max, n_max)
            "node_mask": M_n,  # (B, n_max)
            "pair_mask": M_p,  # (B, n_max, n_max)
            "images": images,  # (B, n_max, 3, H, W) or None
            "teacher": T,  # (B, n_max, D_t) or None
        }
        if P_w is not None:
            out["pair_weight"] = P_w  # (B, n_max, n_max)
        return out

    return pad_collate

