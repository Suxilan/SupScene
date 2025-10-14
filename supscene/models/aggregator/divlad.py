"""
    DiVLAD: Dino inspired VLAD with Attention-Gated Mixture

    Github Repo: https://github.com/MyRepo/SupScene
    Paper: https://arxiv.org/abs/666666666
    Reference:

"""

import torch, torch.nn as nn, torch.nn.functional as F
import math

class DiVLAD(nn.Module):
    """Dino inspired VLAD with Attention-Gated Mixture

    Math (per image):
        a_{k|n} = softmax_k(assign(x))
        w_{h,n} = sigmoid(zscore(attn_{h,n}))^γ
        g_{k,h} ∝ softplus(G_{k,h}) · s_h;  softmax over h
        m_{k,n} = Σ_h g_{k,h} · w_{h,n}
        v_k = Σ_n (a_{k|n} · m_{k,n}) · (x_n − c_k)
        g = L2( concat_k L2(v_k) ) → optional BN

    Args:
        in_dim (int): Input channel C.
        K (int): Number of clusters.
        num_heads (int): Attention heads H.
        gamma (float): Exponent for head quality.
    """
    def __init__(
        self,
        in_dim: int,
        K: int = 32,
        alpha=100.0, 
        num_heads: int = 12,
        gamma: float = 0.5,
        normalize_input=False, 
    ):
        super().__init__()
        self.C = in_dim
        self.K = K
        self.alpha = float(alpha)
        self.H = num_heads
        self.gamma = gamma
        self.eps = 1e-6
        self.normalize_input = bool(normalize_input)
        # Soft assignment (shared symbol with NetVLAD)
        self.assign = nn.Conv2d(in_dim, K, kernel_size=1, bias=False)

        # Cluster centers
        self.centers = nn.Parameter(torch.randn(K, in_dim))

        # Head‑gate logits (K×H), softplus→≥0 before softmax over H
        self.G_raw = nn.Parameter(torch.zeros(K, num_heads))

        # Optional BN on flattened descriptor
        self.bn = nn.BatchNorm1d(K * in_dim)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.centers)
        # nn.init.xavier_uniform_(self.assign.weight)
        # with torch.no_grad():
        #     self.assign.weight.mul_(self.alpha)

    @staticmethod
    def _head_quality(attn: torch.Tensor, gamma: float = 0.5, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Per‑head token quality and image‑level confidence.

        Args:
            attn (Tensor): (B,H,Hh,Ww)
            gamma (float): Exponent.
            eps (float): Stability.

        Returns:
            Tuple[Tensor,Tensor]: (w_hn ∈ (0,1), s_h ∈ (eps,1]) with shapes (B,H,N) and (B,H).
        """
        B, H, Hh, Ww = attn.shape
        N = Hh * Ww
        a = attn.view(B, H, N)
        mu = a.mean(-1, keepdim=True)
        sd = a.std(-1, keepdim=True).clamp_min(eps)
        z = (a - mu) / sd
        w_hn = torch.sigmoid(z)
        if gamma != 1.0:
            w_hn = w_hn.pow(gamma)

        a_sum = a.sum(-1, keepdim=True) + eps
        a_norm = a / a_sum
        entropy = -(a_norm * a_norm.clamp_min(1e-8).log()).sum(-1) / math.log(N + 1e-8)
        s_h = (w_hn.mean(-1) * (1.0 - entropy)).clamp_min(eps)
        return w_hn, s_h

    def forward(self, x_tuple, return_maps: bool = False):
        """Forward.

        Args:
            x_pair: (x, attn) with x∈R^{B×C×H×W}, attn∈R^{B×H×H×W}.
            return_maps: If True, also return effective maps and residuals.

        Returns:
            Tensor | Tuple: g ∈ R^{B×(K·C)} or (g, eff_map, v_kc, None, None).
        """
        x, attn = x_tuple
        B, C, Hh, Ww = x.shape
        N = Hh * Ww

        if self.normalize_input: # set false for DiVLAD
            x = F.normalize(x, p=2, dim=1)  # L2 normalize across descriptor dim

        # Assignment a_{k|n}
        logits = self.assign(x).view(B, self.K, N)
        a_kn = F.softmax(logits, dim=1)

        # Head quality and gates
        w_hn, s_h = self._head_quality(attn, self.gamma)  # (B,H,N), (B,H)
        G_pos = F.softplus(self.G_raw)                    # (K,H) ≥ 0
        g_bkh = torch.softmax(torch.log(G_pos.unsqueeze(0) * s_h.unsqueeze(1) + self.eps), dim=-1)

        # Token quality per cluster
        m_kn = torch.einsum("bkh,bhn->bkn", g_bkh, w_hn)  # (B,K,N)

        # VLAD residuals
        x_nC = x.view(B, C, N)                              # (B,C,N)
        w_kn = a_kn * m_kn                                  # (B,K,N)
        v_x = torch.einsum("bkn,bcn->bkc", w_kn, x_nC)  # (B,K,C)
        a_sum = w_kn.sum(-1, keepdim=True)                   # (B,K,1)
        v_kc = v_x - a_sum * self.centers.unsqueeze(0)    # (B,K,C)

        # Normalize and flatten
        v_kc = F.normalize(v_kc, p=2, dim=-1)
        g = F.normalize(v_kc.reshape(B, -1), p=2, dim=1)
        g = self.bn(g)

        if return_maps:
            assign_map = w_kn.view(B, self.K, Hh, Ww)
            return g, assign_map, v_kc, None, None
        return g

    @property
    def output_dim(self):
        return self.C * self.K

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True