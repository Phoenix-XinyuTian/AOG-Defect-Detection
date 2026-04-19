import torch
import cv2

def dice_iou_from_logits(logits, target, thr=0.5, eps=1e-7):
    """
    从 logits 计算 Dice 和 IoU 指标。
    Compute Dice and IoU metrics from raw logits.
    logits: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    """
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).float()

    # 展平为一维向量方便计算 / Flatten for element-wise computation
    pred_f = pred.view(pred.size(0), -1)
    tgt_f  = target.view(target.size(0), -1)

    inter = (pred_f * tgt_f).sum(dim=1)
    union = (pred_f + tgt_f - pred_f * tgt_f).sum(dim=1)

    dice = (2 * inter + eps) / (pred_f.sum(dim=1) + tgt_f.sum(dim=1) + eps)
    iou  = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

def f1_precision_recall_from_logits(logits, target, thr=0.5, eps=1e-7):
    """
    从 logits 计算 F1-score、Precision 和 Recall。
    Compute F1-score, Precision, and Recall from raw logits.
    logits: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    返回 / Returns: (f1, precision, recall) — 批次均值标量 / batch-mean scalars
    """
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).float()

    # 展平为一维向量方便计算 / Flatten for element-wise computation
    pred_f = pred.view(pred.size(0), -1)
    tgt_f  = target.view(target.size(0), -1)

    tp = (pred_f * tgt_f).sum(dim=1)                      # 真正例 / True Positives
    fp = (pred_f * (1 - tgt_f)).sum(dim=1)                # 假正例 / False Positives
    fn = ((1 - pred_f) * tgt_f).sum(dim=1)                # 假负例 / False Negatives

    precision = (tp + eps) / (tp + fp + eps)               # 精确率 / Precision
    recall    = (tp + eps) / (tp + fn + eps)               # 召回率 / Recall
    f1        = 2 * precision * recall / (precision + recall + eps)  # F1-score

    return f1.mean().item(), precision.mean().item(), recall.mean().item()


def area_ratio_from_mask(mask01):
    """
    计算批次中掩码的平均面积占比。
    Compute mean area ratio of foreground in the batch.
    mask01: [B,1,H,W] in {0,1}
    returns mean area ratio in batch
    """
    return mask01.mean().item()  # 像素均値即等于白色像素占比 / Pixel mean equals foreground ratio

def count_aog_regions(pred_uint8):
    """
    计算预测掩码中 AOG 连通域（独立晶粒）的数量。
    Count the number of connected AOG regions (individual grains) in a predicted mask.
    pred_uint8: [H,W] uint8，值为 0 或 255 / uint8 array with values 0 or 255
    返回 / Returns: int — AOG 区域数量 / number of distinct AOG regions
    """
    # connectedComponents 返回 (数量, 标签图)，背景标签为 0 需减去
    # connectedComponents returns (count, label_map); subtract 1 to exclude background label
    num_labels, _ = cv2.connectedComponents(pred_uint8)
    return num_labels - 1  # 减去背景 / subtract background