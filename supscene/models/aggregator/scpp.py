import torch
import torch.nn as nn
import torch.nn.functional as F


class SCPP(nn.Module):
    """Structural Confidence Probe Pooling aggregator.

    Returns a 2C descriptor by concatenating two confidence-weighted power pools.
    """

    def __init__(
        self,
        in_dim,
        p_s=1.3,
        p_a=4.6,
        eps=1e-6,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.p_s = float(p_s)
        self.p_a = float(p_a)
        self.eps = float(eps)

        self.dwconv = nn.Conv2d(
            self.in_dim,
            self.in_dim,
            kernel_size=3,
            padding=1,
            groups=self.in_dim,
            bias=True,
        )
        self.confidence_proj_low = nn.Conv2d(self.in_dim, 1, kernel_size=1, bias=True)
        self.confidence_proj_high = nn.Conv2d(self.in_dim, 1, kernel_size=1, bias=True)

    def _power_pool(self, x: torch.Tensor, conf_map: torch.Tensor, p: float) -> torch.Tensor:
        conf_norm = conf_map / conf_map.mean(dim=(2, 3), keepdim=True).clamp_min(self.eps)

        x_abs = x.abs().clamp_min(self.eps)
        x_pow = x_abs.pow(p)

        weighted_sum = (conf_norm * x_pow).sum(dim=(2, 3))
        weight_sum = conf_norm.sum(dim=(2, 3)).clamp_min(self.eps)
        pooled = (weighted_sum / weight_sum).pow(1.0 / p)
        return pooled, conf_norm

    def forward(self, x, return_maps=False):
        b, c, h, w = x.shape

        h_feat = x + self.dwconv(x)
        conf_map_low = torch.sigmoid(self.confidence_proj_low(h_feat))
        conf_map_high = torch.sigmoid(self.confidence_proj_high(h_feat))

        pooled_support, _ = self._power_pool(x, conf_map_low, self.p_s)
        pooled_anchor, _ = self._power_pool(x, conf_map_high, self.p_a)

        desc = torch.cat(
            [
                F.normalize(pooled_support, p=2, dim=1),
                F.normalize(pooled_anchor, p=2, dim=1),
            ],
            dim=1,
        )
        desc = F.normalize(desc, p=2, dim=1)

        if return_maps:
            assign_map = conf_map_low.expand(-1, c, -1, -1)
            return desc, assign_map, None, None, None

        return desc

    @property
    def output_dim(self):
        return self.in_dim * 2
