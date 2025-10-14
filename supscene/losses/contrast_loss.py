import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with overlap-aware positive weighting.
    Supports soft/hard modes and masking for invalid pairs.
    
    Args:
        tau (float): Scaling factor for logits. Default is 0.1.
        mode (str): "soft" for soft positive weights, "hard" for binary positives. Default is "soft".
        gamma (float): Exponent for soft weighting. Default is 0.8.
        pos_th (float): Threshold for positive pairs. Default is 0.3.
        exclude_self (bool): Whether to exclude self-comparisons. Default is True.
        eps (float): Small value to avoid division by zero. Default is 1e-8.
    """
    
    def __init__(self, 
                 tau=0.1,
                 mode="soft",           # "soft" | "hard"
                 gamma=0.7,           
                 pos_th=0.3,           # Positive threshold
                 exclude_self=True,    
                 eps=1e-8):
        super().__init__()
        assert mode in ("soft", "hard")
        self.tau = float(tau)
        self.mode = mode
        self.gamma = float(gamma)
        self.pos_th = float(pos_th)
        self.exclude_self = exclude_self
        self.eps = float(eps)

    def _pack_blockdiag(self, o, pair_mask):
        B, N, _ = o.shape
        device, dtype = o.device, o.dtype

        BN = B * N
        o_full = torch.zeros((1, BN, BN), dtype=dtype, device=device)
        pair_mask_full = torch.ones((1, BN, BN), dtype=torch.bool, device=device)

        for b in range(B):
            i0, i1 = b * N, (b + 1) * N
            o_full[:, i0:i1, i0:i1] = o[b:b+1]
            pair_mask_full[:, i0:i1, i0:i1] = pair_mask[b:b+1]

        return o_full, pair_mask_full  # (1, BN, BN)

    def _logsumexp(self, x, keep_mask=None, add_one=True, dim=1):
        """Masked logsumexp for numerical stability"""
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, float('-inf'))
        if add_one:
            shape = list(x.shape)
            shape[dim] = 1
            zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, zeros], dim=dim)
        out = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            out = out.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return out

    def _compute_loss(self, s, o, mask):
        """Main loss computation"""
        B, N, _ = s.shape
        device = s.device

        # Base mask setup
        base_mask = mask.bool()
        if self.exclude_self:
            eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
            base_mask = base_mask & ~eye

        # Positive weights (soft/hard)
        if self.mode == "soft":
            W = torch.where(
                (o - self.pos_th) > 0,
                o.pow(self.gamma),
                o.pow(1/self.gamma)
            )
        else:
            W = (o >= self.pos_th).float()
        W = W * base_mask.float()

        # Valid row check
        has_any = base_mask.any(dim=2)
        has_pos = (W > 0).any(dim=2)
        valid_row = has_any & has_pos
        if not valid_row.any():
            return s.new_tensor(0.0, requires_grad=True)

        # Temperature scaling with stability
        mat = s / max(self.tau, 1e-6)
        mat_masked = mat.masked_fill(~base_mask, float('-inf'))
        row_max = mat_masked.max(dim=2, keepdim=True).values
        row_has_valid = has_any.unsqueeze(-1)
        row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
        mat = torch.where(row_has_valid, mat - row_max.detach(), torch.zeros_like(mat))

        # Denominator and log probabilities
        denom = self._logsumexp(mat, keep_mask=base_mask, add_one=False, dim=2)
        log_prob = mat - denom
        log_prob = log_prob.masked_fill(~base_mask, 0.0)

        # Weighted loss computation
        Z = (W.sum(dim=2).float() + self.eps).clamp_min(1e-8)
        per_anchor = -(W * log_prob).sum(dim=2) / Z

        loss = per_anchor[valid_row].mean()
        return loss

    def forward(self, 
                x,      
                o, 
                pair_mask,
                accelerator=None,):
        """Forward pass
        Args:
            x: Feature tensor of shape (B, N, D)
            o: Overlap tensor of shape (B, N, N)
            pair_mask: Validity mask of shape (B, N, N)
        Returns:
            loss: Computed loss value
        """
        # Normalize and compute similarity
        x = F.normalize(x, p=2, dim=-1, eps=1e-6)
        B, N, D = x.shape
        if accelerator is not None and accelerator.num_processes > 1:
            x_flat = x.reshape(B * N, D)                 # (BN_local, D)
            BN_local = x_flat.size(0)
            device = x.device

            rank = accelerator.process_index
            world = accelerator.num_processes

            local_len = torch.tensor([BN_local], device=device)
            all_len = accelerator.gather(local_len).tolist()   # [len_0, ..., len_{w-1}]
            max_len = int(max(all_len))
            BN_all = int(sum(all_len))

            pad_len = max_len - BN_local
            if pad_len > 0:
                x_pad_local = torch.cat([x_flat, x_flat.new_zeros(pad_len, D)], dim=0)  # (max_len, D)
            else:
                x_pad_local = x_flat

            x_gather_det = accelerator.gather(x_pad_local.detach()) 
            x_gather = x_gather_det.clone()
            start = rank * max_len
            x_gather[start : start + BN_local] = x_flat              

            keep = torch.zeros(world * max_len, dtype=torch.bool, device=device)
            for r, L in enumerate(all_len):
                if L > 0:
                    keep[r * max_len : r * max_len + int(L)] = True
            x_all = x_gather[keep]                                       # (BN_all, D)

            offsets = [0]
            for L in all_len[:-1]:
                offsets.append(offsets[-1] + int(L))
            offset = offsets[rank]

            o_blk, m_blk = self._pack_blockdiag(o, pair_mask)            # (1,BN_local,BN_local)
            o_full = torch.zeros((1, BN_all, BN_all), dtype=o.dtype, device=device)
            m_full = torch.ones((1, BN_all, BN_all), dtype=torch.bool, device=device)
            o_full[:, offset:offset+BN_local, offset:offset+BN_local] = o_blk
            m_full[:, offset:offset+BN_local, offset:offset+BN_local] = m_blk

            s_full = (x_all @ x_all.t()).unsqueeze(0)                    # (1, BN_all, BN_all)
            return self._compute_loss(s_full, o_full, m_full)

        else:
            s = torch.bmm(x, x.transpose(1, 2))      # (B, N, N)
            final_mask = pair_mask.bool()
            return self._compute_loss(s, o, final_mask)


# class InfoNCELoss(nn.Module):
#     """
#     Overlap-aware InfoNCE Loss with soft/hard target options.
#     Maintains same interface as SupConLoss.

#     Args:
#         tau (float): Scaling factor for logits. Default is 0.07.
#         mode (str): "soft" for soft positive weights, "hard" for binary positives. Default is "hard".
#         gamma (float): Exponent for soft weighting. Default is 0.8.
#         pos_th (float): Threshold for positive pairs. Default is 0.3.
#         exclude_self (bool): Whether to exclude self-comparisons. Default is True.
#         eps (float): Small value to avoid division by zero. Default is 1e-8.
#     """
    
#     def __init__(self,
#                  tau=0.07,
#                  mode="hard",        # "soft" | "hard"
#                  gamma=0.8,
#                  pos_th=0.3,
#                  exclude_self=True,
#                  eps=1e-8):
#         super().__init__()
#         assert mode in ("soft", "hard")
#         self.tau = float(tau)
#         self.mode = mode
#         self.gamma = float(gamma)
#         self.pos_th = float(pos_th)
#         self.exclude_self = exclude_self
#         self.eps = float(eps)
#         self.ce = nn.CrossEntropyLoss(reduction='mean')

#     @staticmethod
#     def _cosine_sim(feat):
#         """Compute cosine similarity matrix"""
#         fn = F.normalize(feat, dim=-1, eps=1e-6)
#         return fn @ fn.transpose(1, 2)

#     def forward(self, x, o, pair_mask):
#         device = x.device
#         B, N, _ = x.shape

#         # Similarity and masks
#         s = self._cosine_sim(x)
#         mask = pair_mask.bool()
#         if self.exclude_self:
#             eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
#             mask = mask & (~eye)

#         # Logits with invalid row handling
#         logits = s / max(self.tau, 1e-6)
#         logits = logits.masked_fill(~mask, float("-inf"))
#         row_has_any = mask.any(dim=-1)
#         logits = torch.where(row_has_any.unsqueeze(-1), logits, torch.zeros_like(logits))

#         # Positive mask
#         if self.mode == "soft":
#             pos_w = torch.where(
#                 (o - self.pos_th) > 0,
#                 o.pow(self.gamma),
#                 o.pow(1/self.gamma)
#             )
#             pos_mask = (pos_w > 0) & mask
#         else:
#             pos_mask = (o >= self.pos_th) & mask

#         has_pos = pos_mask.any(dim=-1)
#         if not has_pos.any():
#             return s.new_tensor(0.0, requires_grad=True)

#         # Loss computation
#         logits_flat = logits.view(B * N, N)
#         valid_idx = has_pos.view(-1)

#         if self.mode == "hard":
#             masked_o = o.masked_fill(~pos_mask, -1.0)
#             T = masked_o.argmax(dim=-1).view(B * N)
#             loss = self.ce(logits_flat[valid_idx], T[valid_idx])
#         else:
#             # Soft target distribution
#             T = (pos_w * pos_mask.float())
#             Z = (T.sum(dim=-1, keepdim=True) + self.eps)
#             T = T / Z
#             T_flat = T.view(B * N, N)
            
#             logp = F.log_softmax(torch.nan_to_num(logits_flat[valid_idx], nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
#             mask_flat = mask.view(B * N, N)[valid_idx]
#             logp = logp.masked_fill(~mask_flat, 0.0)
#             T_valid = T_flat[valid_idx].masked_fill(~mask_flat, 0.0)
#             loss = F.kl_div(logp, T_valid, reduction='batchmean')
            
#         return loss