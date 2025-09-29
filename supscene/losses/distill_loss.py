import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillLoss(nn.Module):
    """Distillation loss module with several modes.

    Supported modes:
      - "relation": MSE between pairwise cosine similarity matrices
      - "cosine":  per-node cosine loss (1 - cos)
      - "mse":     per-node feature MSE
      - "kl":      KL divergence on softened logits (scaled by T^2)

    Notes:
      - Public API (class name, method names, and argument names) is preserved.
      - Internal variables use short, math-like symbols: s (student), t (teacher),
        p (pair mask), n (node mask), B/N/D for shapes.
    """

    def __init__(self, distill_type: str = "relation", tau: float = 4.0):
        """Initialize.

        Args:
          distill_type: one of {"relation","cosine","mse","kl"}
          tau: temperature for KL mode

        Returns:
          None
        """
        super().__init__()
        self.distill_type = distill_type
        self.tau = float(tau)

    def forward(
        self,
        s_feats: torch.Tensor,   # [B, N, D_s]
        t_feats: torch.Tensor,   # [B, N, D_t]
        pair_mask: torch.Tensor,  # [B, N, N]
        node_mask: Optional[torch.Tensor]  # [B, N] or None
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
          s_feats: student features, shape [B,N,D_s]
          t_feats: teacher features, shape [B,N,D_t]
          pair_mask: boolean mask for valid pairs, shape [B,N,N]
          node_mask: optional per-node mask, shape [B,N]

        Returns:
          scalar tensor: loss
        """
        eps = 1e-8

        # detach teacher features to prevent gradient flow into teacher
        t = t_feats.detach()

        # short symbolic names for mathematical clarity
        s = s_feats
        p = pair_mask
        n = node_mask

        # apply per-node mask (if provided) by zeroing masked node features
        if n is not None:
            m = n.unsqueeze(-1).to(s.dtype)  # [B,N,1]
            s = s * m
            t = t * m

        # Relation: align pairwise cosine similarity matrices
        if self.distill_type == "relation":
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)
            S_s = s @ s.transpose(1, 2)  # student similarity [B,N,N]
            S_t = t @ t.transpose(1, 2)  # teacher similarity [B,N,N]
            mask = p.bool()
            diff = (S_s - S_t) * mask.float()
            return diff.pow(2).sum() / (mask.sum().float() + eps)

        # Cosine: per-node cosine similarity loss (1 - cos)
        if self.distill_type == "cosine":
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)
            cos = (s * t).sum(dim=-1)  # [B,N]
            if n is not None:
                cos = cos * n.float()
                return 1.0 - cos.sum() / (n.sum().float() + eps)
            return 1.0 - cos.mean()

        # MSE: per-node feature mean squared error
        if self.distill_type == "mse":
            mse = (s - t).pow(2).sum(dim=-1)  # [B,N]
            if n is not None:
                mse = mse * n.float()
                return mse.sum() / (n.sum().float() + eps)
            return mse.mean()

        # KL: softened-target KL divergence, scaled by T^2
        if self.distill_type == "kl":
            T = self.tau
            sl = s / T
            tl = t / T
            logp = F.log_softmax(sl, dim=-1)
            q = F.softmax(tl, dim=-1)
            kl = F.kl_div(logp, q, reduction='none').sum(dim=-1)  # [B,N]
            if n is not None:
                kl = kl * n.float()
                kl = kl.sum() / (n.sum().float() + eps)
            else:
                kl = kl.mean()
            return kl * (T * T)

        raise ValueError(f"Unknown distill_type: {self.distill_type}")


def distill_loss(
    s_feats: torch.Tensor,
    t_feats: torch.Tensor,
    pair_mask: torch.Tensor,
    node_mask: Optional[torch.Tensor],
    *,
    distill_type: str = "relation",
    tau: float = 4.0
) -> torch.Tensor:
    """Functional wrapper for DistillLoss.

    Args:
      s_feats, t_feats, pair_mask, node_mask: same as DistillLoss.forward
      distill_type: mode passed to DistillLoss
      tau: temperature

    Returns:
      scalar tensor: loss
    """
    return DistillLoss(distill_type=distill_type, tau=tau)(
        s_feats, t_feats, pair_mask, node_mask
    )