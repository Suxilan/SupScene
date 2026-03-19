import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm

def cosine_sim(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    compute cosine similarity matrix
    
    Args:
        emb: [N, D] feature matrix
        eps: numerical stability parameter

    Returns:
        [N, N] similarity matrix
    """
    emb = F.normalize(emb, p=2, dim=1, eps=eps)
    return torch.mm(emb, emb.t())


def l2_dist(emb: torch.Tensor) -> torch.Tensor:
    """
    compute L2 distance matrix
    
    Args:
        emb: [N, D] feature matrix

    Returns:
        [N, N] distance matrix
    """
    # use torch.cdist for better efficiency and optimization (suitable for both GPU and CPU)
    return torch.cdist(emb, emb, p=2)


@torch.no_grad()
def recall_at_k(S: torch.Tensor, Y: torch.Tensor, k: int) -> torch.Tensor:
    """
    compute Recall@k (vectorized implementation)
    
    Args:
        S: [N, N] similarity matrix (processed diagonal and mask)
        Y: [N, N] binary label matrix (0/1)
        k: top-k
        
    Returns:
        recall@k
    """
    N = S.size(-1)
    k = min(k, N)

    # get topk indices
    topk_idx = torch.topk(S, k=k, dim=-1, largest=True).indices  # [N, k]
    arange = torch.arange(N, device=S.device)[:, None]  # [N, 1]
    topk_y = Y[arange, topk_idx].float()  # [N, k]
    
    pos_cnt = Y.sum(dim=-1)  # [N]
    valid = pos_cnt > 0

    # compute recall
    tp_k = topk_y.sum(dim=-1)  # [N]
    recall = (tp_k[valid] / pos_cnt[valid]).mean() if valid.any() else S.new_tensor(0.0)
    
    return recall


@torch.no_grad()
def precision_at_k(S: torch.Tensor, Y: torch.Tensor, k: int) -> torch.Tensor:
    """
    compute Precision@k (vectorized implementation)
    
    Args:
        S: [N, N] similarity matrix (processed diagonal and mask)
        Y: [N, N] binary label matrix (0/1)
        k: top-k
        
    Returns:
        precision@k
    """
    N = S.size(-1)
    k = min(k, N)

    # get topk indices
    topk_idx = torch.topk(S, k=k, dim=-1, largest=True).indices  # [N, k]
    arange = torch.arange(N, device=S.device)[:, None]  # [N, 1]
    topk_y = Y[arange, topk_idx].float()  # [N, k]

    # only consider queries with positive samples
    pos_cnt = Y.sum(dim=-1)  # [N]
    valid = pos_cnt > 0

    # compute precision
    tp_k = topk_y.sum(dim=-1)  # [N]
    precision = (tp_k[valid] / float(k)).mean() if valid.any() else S.new_tensor(0.0)
    
    return precision


@torch.no_grad()
def map_at_k(S: torch.Tensor, Y: torch.Tensor, k: int) -> torch.Tensor:
    """
    compute mAP@k (vectorized implementation)
    
    Args:
        S: [N, N] similarity matrix (processed diagonal and mask)
        Y: [N, N] binary label matrix (0/1)
        k: top-k
        
    Returns:
        mAP@k
    """
    N = S.size(-1)
    k = min(k, N)

    # get topk indices
    order = torch.argsort(S, dim=-1, descending=True)[..., :k]  # [N, k]
    arange = torch.arange(N, device=S.device)[:, None]  # [N, 1]
    rel = Y[arange, order].float()  # [N, k]

    cumsum = torch.cumsum(rel, dim=-1)  # [N, k]
    ranks = torch.arange(1, k+1, device=S.device, dtype=rel.dtype)[None, :]  # [1, k]
    prec_at_i = cumsum / ranks  # [N, k]

    # compute AP
    ap = (prec_at_i * rel).sum(dim=-1) / Y.sum(dim=-1).clamp_min(1.0)  # [N]
    valid = Y.sum(dim=-1) > 0
    
    return ap[valid].mean() if valid.any() else S.new_tensor(0.0)


@torch.no_grad()
def ndcg_at_k(S: torch.Tensor, O: torch.Tensor, k: int) -> torch.Tensor:
    """
    compute nDCG@k (vectorized implementation, using continuous overlap)
    
    Args:
        S: [N, N] similarity matrix (processed diagonal and mask)
        O: [N, N] continuous overlap matrix [0, 1]
        k: top-k
        
    Returns:
        nDCG@k
    """
    N = S.size(-1)
    k = min(k, N)

    # get topk indices
    order = torch.argsort(S, dim=-1, descending=True)[..., :k]  # [N, k]
    arange = torch.arange(N, device=S.device)[:, None]  # [N, 1]
    rel_topk = O[arange, order]  # [N, k]

    # compute DCG (using linear gain, more stable for continuous values)
    denom = torch.log2(torch.arange(2, k+2, device=S.device).float())[None, :]  # [1, k]
    dcg = (rel_topk / denom).sum(dim=-1)  # [N]
    
    # compute IDCG
    ideal_topk, _ = torch.sort(O, dim=-1, descending=True)
    ideal_topk = ideal_topk[..., :k]  # [N, k]
    idcg = (ideal_topk / denom).sum(dim=-1)  # [N]
    
    # compute nDCG
    valid = idcg > 0
    ndcg = (dcg[valid] / idcg[valid]).mean() if valid.any() else S.new_tensor(0.0)
    
    return ndcg


def _mask_diag_and_invalid(S: torch.Tensor, O: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    pass the diagonal and invalid pairs in similarity matrix S and overlap matrix O
    
    Args:
        S: [N, N] similarity matrix
        O: [N, N] continuous overlap matrix
        mask: [N, N] valid pair mask
        
    Returns:
        masked S and O
    """
    N = S.size(-1)
    eye = torch.eye(N, device=S.device, dtype=torch.bool)
    S = S.masked_fill(eye, -1e9)
    if mask is not None:
        S = S.masked_fill(~mask, -1e9)
    O = O.masked_fill(~mask, 0.0)
    return S, O


def compute_retrieval_metrics(
    emb: torch.Tensor,
    overlap: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
    mask: Optional[torch.Tensor] = None,
    pos_th: float = 0.3,
    mode: str = "similarity",
    node_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    compute all retrieval metrics (unified interface)
    
    Args:
        emb: [N, D] feature matrix
        overlap: [N, N] continuous overlap matrix [0, 1]
        ks: list of k values to compute
        mask: [N, N] valid pair mask
        pos_th: binarization threshold
        mode: "similarity" or "distance"
        node_mask: [N] node valid mask

    Returns:
        metrics dictionary
    """
    # compute similarity/distance matrix
    if mode == "similarity":
        S = cosine_sim(emb)
    else:
        S = -l2_dist(emb) 
    
    N = S.size(-1)
    pair_mask = ~torch.eye(N, device=S.device, dtype=torch.bool) 
    if node_mask is not None:
        pair_mask = pair_mask & (node_mask[:, None] & node_mask[None, :])
    if mask is not None:
        pair_mask = pair_mask & mask

    S, O = _mask_diag_and_invalid(S, overlap, pair_mask)

    Y = (O >= pos_th).float()
    
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(S, Y, k).item()
        metrics[f"precision@{k}"] = precision_at_k(S, Y, k).item()
        metrics[f"map@{k}"] = map_at_k(S, Y, k).item()
        metrics[f"ndcg@{k}"] = ndcg_at_k(S, O, k).item()
    
    return metrics

@torch.no_grad()
def compute_batch_retrieval_metrics(
    emb: torch.Tensor,
    overlap_gt: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    ks: Tuple[int, ...] = (1, 5, 10),
    mode: str = "similarity",
    pos_th: float = 0.1
) -> Dict[str, float]:
    """
    compute batch retrieval metrics (for quick evaluation during training)
    
    Args:
        similarity: [B, N, N] similarity matrix
        overlap_gt: [B, N, N] ground truth overlap matrix
        node_mask: [B, N] node valid mask
        ks: list of k values to compute
        pos_th: binarization threshold

    Returns:
        metrics dictionary
    """
    B = emb.shape[0]
    batch_metrics = []
    
    for b in range(B):
        emb_b = emb[b]  # [N, D]
        overlap_b = overlap_gt[b]  # [N, N]
        node_mask_b = node_mask[b] if node_mask is not None else None
        metrics_b = compute_retrieval_metrics(
            emb=emb_b,
            overlap=overlap_b,
            node_mask=node_mask_b,
            ks=ks,
            mode=mode,
            pos_th=pos_th
        )
        
        batch_metrics.append(metrics_b)
    
    if batch_metrics:
        avg_metrics = {}
        for key in batch_metrics[0]:
            values = [m[key] for m in batch_metrics if key in m]
            avg_metrics[key] = np.mean(values) if values else 0.0
        return avg_metrics
    else:
        return {f"{metric}@{k}": 0.0 for metric in ["recall", "precision", "map", "ndcg"] for k in ks}
    
@torch.no_grad()
def compute_global_retrieval_metrics(
    all_emb: torch.Tensor,
    scene_overlaps: List[torch.Tensor],
    scene_offsets: List[int],
    ks: Tuple[int, ...] = (1, 5, 10, 20),
    pos_th: float = 0.25,
    mode: str = "similarity"
) -> Dict[str, float]:
    """
    compute global retrieval metrics, avoid constructing the full overlap matrix
    
    Args:
        all_emb: [total_N, D] all image embeddings
        scene_overlaps: List[torch.Tensor] overlap matrix for each scene
        scene_offsets: List[int] starting position of each scene in the global context
        ks: list of k values to compute
        pos_th: binarization threshold
        mode: "similarity" or "distance"

    Returns:
        metrics dictionary
    """
    device = all_emb.device
    total_N = all_emb.size(0)
    
    if mode == "similarity":
        S = cosine_sim(all_emb)
    else:
        S = -l2_dist(all_emb)

    S.fill_diagonal_(-float('inf') if mode == "similarity" else float('inf'))
    
    global_pos_mask = torch.zeros(total_N, total_N, dtype=torch.bool, device=device)
    
    for i, overlap in enumerate(scene_overlaps):
        start = scene_offsets[i]
        end = scene_offsets[i + 1] if i + 1 < len(scene_offsets) else total_N
        
        overlap = overlap.to(device)
        pos_mask = (overlap >= pos_th)
        pos_mask.fill_diagonal_(False) 
        
        global_pos_mask[start:end, start:end] = pos_mask
    
    # compute metrics
    metrics = {}
    
    for k in ks:
        # get topk indices
        _, topk_indices = torch.topk(S, k, dim=-1, largest=True)
        
        recall_scores = []
        precision_scores = []
        ap_scores = []
        ndcg_scores = []
        
        for query_idx in tqdm(range(total_N), desc=f"Computing metrics for {k}", total=total_N):
            pos_indices = global_pos_mask[query_idx].nonzero(as_tuple=True)[0]
            
            if len(pos_indices) == 0:
                continue  
            
            pred_indices = topk_indices[query_idx]

            hits = torch.isin(pred_indices, pos_indices).sum().item()
            recall = hits / len(pos_indices)
            recall_scores.append(recall)

            precision = hits / k
            precision_scores.append(precision)

            ap = 0.0
            hits_so_far = 0
            for i, pred_idx in enumerate(pred_indices):
                if pred_idx in pos_indices:
                    hits_so_far += 1
                    ap += hits_so_far / (i + 1)
            # Align with GeoMetricLab `compute_metrics`: AP@k is normalized by
            # the total number of positives for the query, not by min(num_pos, k).
            ap = ap / len(pos_indices) if len(pos_indices) > 0 else 0.0
            ap_scores.append(ap)

            ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos_indices), k)))
            
            dcg = 0.0
            for i, pred_idx in enumerate(pred_indices):
                if pred_idx in pos_indices:
                    dcg += 1.0 / np.log2(i + 2)
            
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        if recall_scores:
            metrics[f"recall@{k}"] = np.mean(recall_scores)
            metrics[f"precision@{k}"] = np.mean(precision_scores)
            metrics[f"map@{k}"] = np.mean(ap_scores)
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
        else:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
    
    return metrics


# def compute_scene_aware_global_metrics(
#     all_emb: torch.Tensor,
#     scene_overlaps: List[torch.Tensor],
#     scene_offsets: List[int],
#     ks: Tuple[int, ...] = (1, 5, 10, 20),
#     pos_th: float = 0.25,
#     mode: str = "similarity",
#     inter_scene_penalty: float = 0.1
# ) -> Dict[str, float]:
#     """
#     scene-aware global retrieval metrics computation
#     Apply penalties to cross-scene retrieval results, focusing more on intra-scene retrieval performance.

#     Args:
#         inter_scene_penalty: Penalty coefficient for cross-scene retrieval.
#     """
#     device = all_emb.device
#     total_N = all_emb.size(0)

#     # compute global similarity matrix
#     if mode == "similarity":
#         all_emb_norm = torch.nn.functional.normalize(all_emb, p=2, dim=-1)
#         sim_matrix = torch.mm(all_emb_norm, all_emb_norm.t())
#     else:
#         sim_matrix = -torch.cdist(all_emb, all_emb, p=2)
    
#     sim_matrix.fill_diagonal_(-float('inf') if mode == "similarity" else float('inf'))

#     # apply penalties to cross-scene retrieval
#     for i in range(len(scene_offsets)):
#         start_i = scene_offsets[i]
#         end_i = scene_offsets[i + 1] if i + 1 < len(scene_offsets) else total_N
        
#         for j in range(len(scene_offsets)):
#             if i == j:
#                 continue  
            
#             start_j = scene_offsets[j]
#             end_j = scene_offsets[j + 1] if j + 1 < len(scene_offsets) else total_N

#             if mode == "similarity":
#                 sim_matrix[start_i:end_i, start_j:end_j] *= inter_scene_penalty
#             else:
#                 sim_matrix[start_i:end_i, start_j:end_j] /= inter_scene_penalty
    
#     return compute_global_retrieval_metrics(
#         all_emb, scene_overlaps, scene_offsets, ks, pos_th, mode
#     )

