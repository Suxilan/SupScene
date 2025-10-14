import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NetVLAD(nn.Module):
    """
    NetVLAD layer implementation for aggregating local features into global descriptors
    
    Args:
        num_clusters (int): K.
        in_dim (int): C.
        alpha (float): Assignment scale (higher → harder).
        normalize_input (bool): L2 normalize inputs per descriptor.
    """

    def __init__(
        self, 
        in_dim: int,
        K: int = 32,
        alpha=100.0, 
        normalize_input=True, 
    ):
        super(NetVLAD, self).__init__()
        self.K = K
        self.C = in_dim
        self.alpha = float(alpha)
        self.normalize_input = bool(normalize_input)

        # Soft assignment
        self.assign = nn.Conv2d(in_dim, K, kernel_size=1, bias=False)

        # Cluster centers
        self.centers = nn.Parameter(torch.rand(K, in_dim))

        self.bn = nn.BatchNorm1d(K * in_dim)

        # K‑means init placeholders
        self.clsts = None
        self.traindescs = None

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.centers)
        nn.init.xavier_uniform_(self.assign.weight)
        with torch.no_grad():
            self.assign.weight.data *= self.alpha

    def init_from_clusters(self, clsts, traindescs) -> None:
        """Init with k‑means results.

        Args:
            clsts (ndarray): (K,C) cluster centers.
            traindescs (ndarray): (M,C) training descriptors.
        """
        import numpy as np

        self.clsts = clsts
        self.traindescs = traindescs

        clsts_unit = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clsts_unit, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]
        self.alpha = float((-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item())

        with torch.no_grad():
            self.centers.copy_(torch.from_numpy(clsts))
            self.assign.weight.copy_(torch.from_numpy(self.alpha * clsts_unit)[:, :, None, None])

    def forward(self, x, return_maps=False):
        """Forward.

        Args:
            x (Tensor): (B,C,H,W) or (B,C,L)
            return_maps (bool): If True, also return effective maps.

        Returns:
            Tensor | Tuple: g ∈ R^{B×(K·C)} or (g, a_map, v_kc, None, None).
        """
        B, C = x.shape[:2]
        if x.ndim == 3:  # (B,C,L)
            x = x.unsqueeze(-1)
            
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # L2 normalize across descriptor dim

        # Assignment a_{k|n}
        a = self.assign(x)                     # (B,K,H,W)
        a = F.softmax(a, dim=1)

        # Flatten
        Hh, Ww = x.shape[-2:]
        N = Hh * Ww
        x_nC = x.view(B, C, N)                  # (B,C,N)
        a_kn = a.view(B, self.K, N)             # (B,K,N)

        # Aggregation
        v_x = torch.einsum("bkn,bcn->bkc", a_kn, x_nC)  # Σ_n a·x
        a_sum = a_kn.sum(-1, keepdim=True)               # Σ_n a
        v_kc = v_x - a_sum * self.centers.unsqueeze(0)   # residuals
        
        # Intra-normalization
        v_kc = F.normalize(v_kc, p=2, dim=-1)
        g = F.normalize(v_kc.reshape(B, -1), p=2, dim=1)
        g = self.bn(g)

        if return_maps:
            assign_map = a_kn.view(B, self.K, Hh, Ww)
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
