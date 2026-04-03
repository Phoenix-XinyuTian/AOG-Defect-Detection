import torch

def dice_iou_from_logits(logits, target, thr=0.5, eps=1e-7):
    """
    logits: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    """
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).float()

    # flatten
    pred_f = pred.view(pred.size(0), -1)
    tgt_f  = target.view(target.size(0), -1)

    inter = (pred_f * tgt_f).sum(dim=1)
    union = (pred_f + tgt_f - pred_f * tgt_f).sum(dim=1)

    dice = (2 * inter + eps) / (pred_f.sum(dim=1) + tgt_f.sum(dim=1) + eps)
    iou  = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

def area_ratio_from_mask(mask01):
    """
    mask01: [B,1,H,W] in {0,1}
    returns mean area ratio in batch
    """
    return mask01.mean().item()  # because mean over pixels == white ratio
