import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import torch
from torch.utils.data import DataLoader

from supscene.datasets import GL3DSubgraphDataset, SubgraphSampler, make_pad_collate


def main():
    root = "data"
    split = os.path.join(root, "dataset_split/train.txt")

    ds = GL3DSubgraphDataset(
        root_dir=root,
        split_txt=split,
        n_sub=128,
        sampler=SubgraphSampler(mode="anchor_expand", iou_th=0.1, topk_per_hop=64),
        load_images=False,
        scenes_per_epoch=None,  # 默认=全部场景
        samples_per_scene=None,    # 每个场景抽 1 个子图
    )

    collate_fn = make_pad_collate(diag_weight=0.1)

    dl = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,          # 便于一致性检查
        num_workers=0,          # smoke，先用单进程
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # 每轮训练前，重排计划
    ds.reshuffle_epoch(epoch=0)

    total_batches = 0
    for batch in dl:
        total_batches += 1
        B = len(batch["scene_ids"])  # 实际 batch size（最后一个 batch 可能不足）
        overlap = batch["overlap"]        # [B, N, N]
        node_mask = batch["node_mask"]    # [B, N]
        pair_mask = batch["pair_mask"]    # [B, N, N]
        pw = batch.get("pair_weight", None)

        # 形状检查
        assert overlap.ndim == 3 and overlap.shape[0] == B
        assert node_mask.ndim == 2 and node_mask.shape[0] == B and node_mask.shape[1] == overlap.shape[1]
        assert pair_mask.ndim == 3 and pair_mask.shape == overlap.shape

        # 掩码一致性：pair_mask 的有效行列应与 node_mask 对应
        n_valid = node_mask.sum(dim=1)
        for b in range(B):
            n = int(n_valid[b].item())
            # 有效区域为 n×n，必须 True；padding 区域必须为 False
            assert pair_mask[b, :n, :n].all().item(), "有效区域 pair_mask 应为 True"
            assert not pair_mask[b, n:, :].any().item(), "padding 行应为 False"
            assert not pair_mask[b, :, n:].any().item(), "padding 列应为 False"
            # 对角应在 [0,1] 且 >= 其他最小值（O 的构造中对角为 1）
            diag = overlap[b, torch.arange(n), torch.arange(n)]
            assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        if pw is not None:
            assert pw.shape == overlap.shape

    print(f"OK: {len(ds)} samples, {total_batches} batches passed basic checks.")


if __name__ == "__main__":
    main()

