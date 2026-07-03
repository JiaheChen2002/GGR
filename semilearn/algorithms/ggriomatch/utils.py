
import torch
import torch.nn.functional as F

# === IOMatch multi-binary supervised loss (from OP_Match) ===
def mb_sup_loss(logits_ova, label):
    """
    logits_ova: [B, 2K] from multi-binary classifier
    label: [B] in [0, K-1]
    """
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)  # [B,2,K]
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, 1)  # softmax over {neg,pos}
    label_s_sp = torch.zeros((batch_size, num_classes), device=label.device, dtype=torch.long)
    label_range = torch.arange(0, batch_size, device=label.device, dtype=torch.long)
    label_s_sp[label_range[label < num_classes], label[label < num_classes]] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(probs_ova[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(probs_ova[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    return open_loss_neg + open_loss


def soft_ce_loss(logits, targets):
    """
    Cross-entropy with soft targets.
    logits: [B,C]
    targets: [B,C] probabilities
    """
    logp = F.log_softmax(logits, dim=-1)
    return -(targets * logp).sum(dim=-1)


def masked_soft_ce(logits, targets, mask, norm_mode="mean"):
    """
    logits: [B,C]
    targets: soft [B,C]
    mask: [B] float in [0,1]
    norm_mode:
      - "mean": average over full batch size (stable when mask is sparse)
      - "sum": average over accepted samples (sum / mask.sum)
    """
    per = soft_ce_loss(logits, targets)
    mask = mask.float()
    if norm_mode == "mean":
        return (per * mask).mean()
    elif norm_mode == "sum":
        return (per * mask).sum() / (mask.sum().clamp_min(1e-12))
    else:
        raise ValueError(f"Unknown norm_mode={norm_mode}")
