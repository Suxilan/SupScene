# viz_gtp.py
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ---------- util ----------

def _to_numpy_img(img: torch.Tensor):
    """
    img: [3,H,W], [0,1] or standard normalized.
    return: HxWx3 uint8
    """
    if img.ndim == 3 and img.shape[0] in (1,3):
        x = img.detach().cpu().float()
        if x.max() <= 1.0 and x.min() >= 0.0:
            pass
        else:
            # try unnormalize DINO/Imagenet stats if it's obviously normalized
            # (不清楚分布就简单clip)
            x = x
        x = x.clamp(0,1)
        x = x.permute(1,2,0).numpy()
        x = (x*255.0 + 0.5).astype(np.uint8)
        return x
    raise ValueError("expect img shape [3,H,W]")

def _norm_to_px(mu_xy, sigma_xy, H, W):
    """
    把 GTP 的 [-1,1] 归一化坐标转像素坐标
    mu_xy: [K,2]  (mux, muy in [-1,1])
    sigma_xy: [K,2] (sigx,sigy in norm space)
    """
    mu_px = torch.empty_like(mu_xy)
    mu_px[:,0] = (mu_xy[:,0] + 1.0) * 0.5 * (W - 1)
    mu_px[:,1] = (mu_xy[:,1] + 1.0) * 0.5 * (H - 1)

    # sigma 也按像素尺度映射（近似按各向同性缩放）
    sig_px = torch.empty_like(sigma_xy)
    sig_px[:,0] = sigma_xy[:,0] * 0.5 * (W - 1)
    sig_px[:,1] = sigma_xy[:,1] * 0.5 * (H - 1)
    return mu_px, sig_px

def _draw_ellipse(ax, cx, cy, rx, ry, color, lw=2, alpha=0.9, nsig=2.0):
    """
    画 nsig-σ 的椭圆（轴对齐版本）
    """
    e = Ellipse((cx, cy), width=2*nsig*rx, height=2*nsig*ry,
                angle=0.0, fill=False, edgecolor=color, linewidth=lw, alpha=alpha)
    ax.add_patch(e)

def _cmaps(idx, K):
    # 固定色表
    cmap = plt.get_cmap('tab20')
    return cmap(idx % 20)

# ---------- 1) 单图可视化 ----------

@torch.no_grad()
def visualize_gtp_tokens(
    image: torch.Tensor,   # [3,H,W], 0-1
    A: torch.Tensor,       # [K,H,W]  or [1,K,H,W]
    mu: torch.Tensor,      # [K,2]
    sigma: torch.Tensor,   # [K,2]
    out_path: str = "gtp_vis.png",
    show_topk_heatmaps: int = 6,
):
    if A.dim() == 4:  # [1,K,H,W]
        A = A[0]
    K, H, W = A.shape
    img = _to_numpy_img(image)

    mu_px, sig_px = _norm_to_px(mu, sigma, H, W)

    # 主图：原图 + 椭圆
    fig = plt.figure(figsize=(10, 6))
    ax_main = fig.add_subplot(2, max(3, math.ceil(show_topk_heatmaps/2)), 1)
    ax_main.imshow(img)
    ax_main.set_title("Image with GTP ellipses")
    ax_main.axis('off')

    # 每个 token 的质量（权重和）
    mass = A.view(K, -1).sum(dim=1)
    order = torch.argsort(mass, descending=True)

    # 叠加椭圆与中心
    for rank, k in enumerate(order):
        c = _cmaps(rank, K)
        cx, cy = mu_px[k,0].item(), mu_px[k,1].item()
        rx, ry = sig_px[k,0].item(), sig_px[k,1].item()
        _draw_ellipse(ax_main, cx, cy, rx, ry, color=c, lw=2, nsig=2.0)
        ax_main.scatter([cx], [cy], s=20, c=[c], marker='x')

    # 绘制 top-k attention map
    show_topk_heatmaps = min(show_topk_heatmaps, K)
    ncols = max(3, math.ceil((show_topk_heatmaps + 1) / 2))
    for i in range(show_topk_heatmaps):
        ax = fig.add_subplot(2, ncols, i + 2)
        ax.imshow(img)
        ax.imshow(A[order[i]].cpu().numpy(), cmap='jet', alpha=0.45)
        ax.set_title(f"token#{order[i]} mass={mass[order[i]].item():.2f}")
        ax.axis('off')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[viz] saved: {out_path}")

# ---------- 2) 图对：位置感知 OT 可视化 ----------

def cosine_sim(T1, T2):
    T1 = F.normalize(T1, dim=-1)
    T2 = F.normalize(T2, dim=-1)
    return T1 @ T2.t()  # [K1,K2]

def sinkhorn(C, epsilon=0.05, niter=100, tol=1e-6):
    """
    最小化 <P, C> - epsilon * H(P)
    返回 transport plan P （大小与 C 一致），行列近似均匀。
    """
    # 转化为 K = exp(-C/eps)
    K = torch.exp(-C / epsilon).clamp_min(1e-12)
    u = torch.ones(C.size(0), device=C.device) / C.size(0)
    v = torch.ones(C.size(1), device=C.device) / C.size(1)

    for _ in range(niter):
        u_prev = u
        u = (1.0 / (K @ v)).clamp_min(1e-12)
        v = (1.0 / (K.t() @ u)).clamp_min(1e-12)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break
    P = torch.diag(u) @ K @ torch.diag(v)
    return P

@torch.no_grad()
def visualize_pair_ot(
    img1: torch.Tensor, T1: torch.Tensor, mu1: torch.Tensor, sigma1: torch.Tensor,   # [3,H,W], [K,C], [K,2], [K,2]
    img2: torch.Tensor, T2: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor,
    out_path: str = "gtp_pair_ot.png",
    alpha=0.7, beta=0.3, epsilon=0.05,
):
    # 图像准备
    I1 = _to_numpy_img(img1)
    I2 = _to_numpy_img(img2)
    H, W = I1.shape[:2]
    mu1_px, _ = _norm_to_px(mu1, sigma1, H, W)
    mu2_px, _ = _norm_to_px(mu2, sigma2, H, W)

    # 余弦相似 + 位置代价
    S = cosine_sim(T1, T2)        # [K1,K2]
    Dpos = torch.cdist(mu1, mu2, p=2)  # [-1,1] 空间距离
    Dpos = Dpos / 0.5             # 归一成 ~[0,2]
    # 代价矩阵
    C = alpha * (1 - S.clamp(-1,1)) + beta * Dpos

    # Sinkhorn OT
    P = sinkhorn(C, epsilon=epsilon)  # [K1,K2]
    # 对齐相似度（可当图对分数）
    score = (P * S).sum().item()

    # 画图：把两图拼横，连中心
    Wpad = 20
    canvas = np.ones((H, W*2 + Wpad, 3), dtype=np.uint8) * 255
    canvas[:, :W] = I1
    canvas[:, W+Wpad:] = I2
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.imshow(canvas)
    ax.axis('off')
    ax.set_title(f"Position-aware OT match, sim={score:.3f}")

    # 画 token 中心 & 连接线（粗细/透明度 ∝ P）
    K1, K2 = P.shape
    for i in range(K1):
        for j in range(K2):
            w = P[i,j].item()
            if w < 1e-3: 
                continue
            c = _cmaps(i, K1)
            x1, y1 = mu1_px[i,0].item(), mu1_px[i,1].item()
            x2, y2 = mu2_px[j,0].item() + W + Wpad, mu2_px[j,1].item()
            ax.plot([x1, x2], [y1, y2], color=c, linewidth=1.5 + 4*w, alpha=min(0.9, 0.2 + 1.5*w))
            ax.scatter([x1], [y1], c=[c], s=30, marker='o')
            ax.scatter([x2], [y2], c=[c], s=30, marker='x')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[viz] saved: {out_path}")

# ---------- demo（可删） ----------
if __name__ == "__main__":
    # 制作一对假数据演示
    H, W, C, K = 280, 420, 256, 6
    img1 = torch.rand(3,H,W)
    img2 = torch.rand(3,H,W)
    # 假 tokens/参数
    T1 = F.normalize(torch.randn(K, C), dim=-1)
    T2 = F.normalize(torch.randn(K, C), dim=-1)
    mu1 = torch.rand(K,2)*2-1
    mu2 = torch.rand(K,2)*2-1
    sigma1 = torch.rand(K,2)*0.2 + 0.08
    sigma2 = torch.rand(K,2)*0.2 + 0.08
    x_dummy = torch.randn(K, H, W)
    A_dummy = torch.softmax(x_dummy.view(K, -1), dim=-1).view(K, H, W)

    visualize_gtp_tokens(img1, A_dummy, mu1, sigma1, "demo_gtp_single.png")
    visualize_pair_ot(img1, T1, mu1, sigma1, img2, T2, mu2, sigma2, "demo_gtp_pair.png")
