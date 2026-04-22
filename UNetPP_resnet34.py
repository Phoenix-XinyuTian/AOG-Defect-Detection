import os
import random
import json
import argparse
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.unetpp_data_policy import preprocess_bgr_unetpp, augment_bgr_mask_unetpp


EVAL_THRESHOLD = 0.45
MIN_COMPONENT_AREA = 0


def _threshold_tag(threshold):
    """Convert numeric threshold to a folder-safe tag, e.g. 0.4 -> 0p4."""
    value = f"{threshold:.4f}".rstrip("0").rstrip(".")
    return value.replace(".", "p")


def save_experiment_config(output_folder, config):
    os.makedirs(output_folder, exist_ok=True)
    cfg_path = os.path.join(output_folder, "experiment_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Saved config: {cfg_path}")


class AOGDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256), augment=False):
        valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".bmp")
        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(valid_extensions)
        ])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to load: {img_name}")

        image = preprocess_bgr_unetpp(image, img_size=self.img_size, clahe=self._clahe)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Training-only geometric augmentation (shared transform for image/mask)
        if self.augment:
            image, mask = augment_bgr_mask_unetpp(image, mask)

        image = image.transpose(2, 0, 1) / 255.0
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def calculate_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8).flatten()
    gt = (gt_mask > 0).astype(np.uint8).flatten()
    intersection = np.sum(pred * gt)
    pred_area = np.sum(pred)
    gt_area = np.sum(gt)
    eps = 1e-6

    iou = (intersection + eps) / (pred_area + gt_area - intersection + eps)
    dice = (2.0 * intersection + eps) / (pred_area + gt_area + eps)
    precision = (intersection + eps) / (pred_area + eps)
    recall = (intersection + eps) / (gt_area + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return iou, dice, precision, recall, f1


def loss_fn(prob, target, eps=1e-6):
    # Tversky loss: alpha=0.45, beta=0.55.
    alpha = 0.45
    beta = 0.55
    prob_f = prob.view(prob.size(0), -1)
    tgt_f = target.view(target.size(0), -1)

    tp = (prob_f * tgt_f).sum(dim=1)
    fp = (prob_f * (1.0 - tgt_f)).sum(dim=1)
    fn = ((1.0 - prob_f) * tgt_f).sum(dim=1)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    tversky_loss = 1.0 - tversky.mean()

    # Binary focal loss on probabilities: gamma=2.0, pos_weight=10.0.
    gamma = 2.0
    pos_weight = 7.0
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = pos_weight * target + (1.0 - target)
    focal = -alpha_t * torch.pow(1.0 - p_t, gamma) * torch.log(p_t.clamp(min=eps, max=1.0 - eps))
    focal_loss = focal.mean()

    return focal_loss + tversky_loss


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, val_loader=None, epochs=20, best_ckpt_path="best_model_unetpp_resnet34.pth"):
    device = get_device()
    if device.type == "mps":
        print("Apple Silicon GPU detected, using MPS")
    elif device.type == "cuda":
        print("NVIDIA GPU detected, using CUDA")
    else:
        print("No GPU detected, using CPU")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = loss_fn
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print(f"Training on {device}...")

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loader is not None:
            model.eval()
            val_dices, val_ious, val_f1s, val_precs, val_recs = [], [], [], [], []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    preds = model(images).squeeze(1).cpu().numpy()
                    masks_np = masks.squeeze(1).numpy()
                    for p, g in zip(preds, masks_np):
                        pred_mask = (p > EVAL_THRESHOLD).astype(np.uint8) * 255
                        gt_mask = (g * 255).astype(np.uint8)
                        iou, dice, prec, rec, f1 = calculate_metrics(pred_mask, gt_mask)
                        val_ious.append(iou)
                        val_dices.append(dice)
                        val_precs.append(prec)
                        val_recs.append(rec)
                        val_f1s.append(f1)

            val_dice = sum(val_dices) / len(val_dices)
            val_iou = sum(val_ious) / len(val_ious)
            val_f1 = sum(val_f1s) / len(val_f1s)
            val_prec = sum(val_precs) / len(val_precs)
            val_rec = sum(val_recs) / len(val_recs)

            print(
                f"Epoch {epoch + 1:03d}/{epochs} | train_loss={train_loss:.4f} | lr={current_lr:.6e} "
                f"| Dice={val_dice:.4f} | IoU={val_iou:.4f} "
                f"| F1={val_f1:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f}"
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_ckpt_path)
                print(
                    f"  New best model saved: {best_ckpt_path} "
                    f"(epoch={best_epoch}, val_dice={best_val_dice:.4f})"
                )
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, lr={current_lr:.6e}")

        scheduler.step()

    if val_loader is not None:
        print(f"Training finished. Best val Dice = {best_val_dice:.4f} at epoch {best_epoch:03d}")
    return best_val_dice, best_epoch


def _save_overlay(ori_img_bgr, mask_uint8, out_path, alpha=0.5):
    overlay = ori_img_bgr.copy()
    overlay[mask_uint8 > 0] = (0, 0, 255)
    blended = cv2.addWeighted(overlay, alpha, ori_img_bgr, 1 - alpha, 0)
    ok = cv2.imwrite(out_path, blended)
    if not ok:
        print(f"Warning: failed to save overlay: {out_path}")


def _write_metrics_files(output_folder, rows, summary):
    per_image_csv = os.path.join(output_folder, "metrics_per_image.csv")
    summary_txt = os.path.join(output_folder, "metrics_summary.txt")

    with open(per_image_csv, "w", encoding="utf-8") as f:
        f.write("filename,iou,dice,f1,precision,recall,aog_area_percent,aog_count\n")
        for r in rows:
            f.write(
                f"{r['filename']},{r['iou']:.6f},{r['dice']:.6f},{r['f1']:.6f},"
                f"{r['precision']:.6f},{r['recall']:.6f},{r['aog_area_percent']:.6f},{r['aog_count']}\n"
            )

    with open(summary_txt, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def _save_pr_curve(y_true, y_score, out_path):
    if y_true.size == 0:
        return

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)
    pos = max(1, int(np.sum(y_true == 1)))

    recall = tp_cum / pos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (Pixel-level)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_confusion_matrix_plot(tp, fp, tn, fn, out_path):
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["GT 0", "GT 1"])
    plt.title("Confusion Matrix (Pixel-level)")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_gt_pred_compare(output_folder, rows):
    compare_csv = os.path.join(output_folder, "gt_pred_area_count_compare.csv")
    compare_png = os.path.join(output_folder, "gt_pred_area_count_compare.png")

    with open(compare_csv, "w", encoding="utf-8") as f:
        f.write("filename,gt_area_percent,pred_area_percent,gt_count,pred_count\n")
        for r in rows:
            f.write(
                f"{r['filename']},{r['gt_aog_area_percent']:.6f},{r['aog_area_percent']:.6f},"
                f"{r['gt_aog_count']},{r['aog_count']}\n"
            )

    gt_area = np.array([r["gt_aog_area_percent"] for r in rows], dtype=np.float32)
    pred_area = np.array([r["aog_area_percent"] for r in rows], dtype=np.float32)
    gt_count = np.array([r["gt_aog_count"] for r in rows], dtype=np.float32)
    pred_count = np.array([r["aog_count"] for r in rows], dtype=np.float32)

    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.scatter(gt_area, pred_area, alpha=0.8)
    max_area = float(max(np.max(gt_area), np.max(pred_area), 1.0))
    plt.plot([0, max_area], [0, max_area], "r--", linewidth=1)
    plt.xlabel("GT AOG area %")
    plt.ylabel("Pred AOG area %")
    plt.title("GT vs Pred Area %")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(gt_count, pred_count, alpha=0.8)
    max_count = float(max(np.max(gt_count), np.max(pred_count), 1.0))
    plt.plot([0, max_count], [0, max_count], "r--", linewidth=1)
    plt.xlabel("GT AOG count")
    plt.ylabel("Pred AOG count")
    plt.title("GT vs Pred Count")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(compare_png, dpi=200)
    plt.close()


def _save_gt_pred_image_compare(items, out_path, max_items=12):
    """Save a qualitative GT-vs-Pred mask panel (worst Dice samples first)."""
    if not items:
        return

    # Show harder cases first so qualitative errors are easy to inspect.
    items_sorted = sorted(items, key=lambda x: x["dice"])
    show_items = items_sorted[:max_items]

    n = len(show_items)
    fig, axes = plt.subplots(n, 2, figsize=(8, max(2.5 * n, 4)))
    if n == 1:
        axes = np.array([axes])

    for i, item in enumerate(show_items):
        gt_mask = item["gt_mask"]
        pred_mask = item["pred_mask"]
        name = item["filename"]
        dice = item["dice"]

        axes[i, 0].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"GT | {name}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title(f"Pred | Dice={dice:.3f}", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle("GT vs Prediction Mask Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _remove_small_components(mask_uint8, min_area=MIN_COMPONENT_AREA):
    if min_area <= 0:
        return mask_uint8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def batch_process_and_evaluate(model, input_folder, gt_folder, output_folder):
    device = get_device()
    model.to(device)
    model.eval()

    masks_dir = os.path.join(output_folder, "masks")
    overlays_dir = os.path.join(output_folder, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    print(f"Output masks -> {masks_dir}")
    print(f"Output overlays -> {overlays_dir}")

    valid_ext = (".png", ".jpg", ".jpeg", ".tif")
    img_list = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)])

    total_metrics = {"iou": 0, "dice": 0, "prec": 0, "rec": 0, "f1": 0}
    count = 0
    total_aog_count = 0
    total_gt_aog_count = 0
    rows = []
    gt_pred_vis_items = []

    all_y_true = []
    all_y_score = []
    cm_tp = cm_fp = cm_tn = cm_fn = 0

    print("\n" + "=" * 108)
    print(
        f"{'Filename':<30} | {'IoU':<8} | {'Dice':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AOG area %':<10} | {'AOG count':<9}"
    )
    print("-" * 108)

    for img_name in img_list:
        img_path = os.path.join(input_folder, img_name)
        gt_path = os.path.join(gt_folder, img_name)
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_folder, os.path.splitext(img_name)[0] + ".png")
        if not os.path.exists(gt_path):
            continue

        ori_img = cv2.imread(img_path)
        h, w = ori_img.shape[:2]
        proc = preprocess_bgr_unetpp(ori_img, img_size=(256, 256))

        img_tensor = proc.transpose(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_prob = model(img_tensor).squeeze().cpu().numpy()

        res_mask = (pred_prob > EVAL_THRESHOLD).astype(np.uint8) * 255
        res_mask = cv2.resize(res_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        res_mask = _remove_small_components(res_mask, min_area=MIN_COMPONENT_AREA)

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        iou, dice, prec, rec, f1 = calculate_metrics(res_mask, gt_img)

        gt_bin = (gt_img > 0).astype(np.uint8)
        prob_resized = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        all_y_true.append(gt_bin.flatten())
        all_y_score.append(prob_resized.flatten())

        pred_bin = (res_mask > 0).astype(np.uint8)
        cm_tp += int(np.sum((pred_bin == 1) & (gt_bin == 1)))
        cm_fp += int(np.sum((pred_bin == 1) & (gt_bin == 0)))
        cm_tn += int(np.sum((pred_bin == 0) & (gt_bin == 0)))
        cm_fn += int(np.sum((pred_bin == 0) & (gt_bin == 1)))

        aog_area = (np.count_nonzero(res_mask) / res_mask.size) * 100
        num_labels, _ = cv2.connectedComponents(res_mask)
        aog_count = num_labels - 1

        gt_area = (np.count_nonzero(gt_img) / gt_img.size) * 100
        gt_num_labels, _ = cv2.connectedComponents((gt_img > 0).astype(np.uint8) * 255)
        gt_aog_count = gt_num_labels - 1

        print(
            f"{img_name:<30} | {iou:.4f}   | {dice:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}   | {aog_area:>8.2f}%   | {aog_count:>9d}"
        )

        stem = os.path.splitext(img_name)[0]
        rows.append({
            "filename": stem,
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "aog_area_percent": aog_area,
            "aog_count": aog_count,
            "gt_aog_area_percent": gt_area,
            "gt_aog_count": gt_aog_count,
        })
        gt_pred_vis_items.append({
            "filename": stem,
            "dice": dice,
            "gt_mask": gt_img.copy(),
            "pred_mask": res_mask.copy(),
        })

        total_metrics["iou"] += iou
        total_metrics["dice"] += dice
        total_metrics["prec"] += prec
        total_metrics["rec"] += rec
        total_metrics["f1"] += f1
        total_aog_count += aog_count
        total_gt_aog_count += gt_aog_count
        count += 1

        mask_out = os.path.join(masks_dir, f"{stem}_pred.png")
        ok = cv2.imwrite(mask_out, res_mask)
        if not ok:
            print(f"Warning: failed to save mask: {mask_out}")

        overlay_out = os.path.join(overlays_dir, f"{stem}_overlay.png")
        _save_overlay(ori_img, res_mask, overlay_out)

    if count > 0:
        print("-" * 108)
        print(
            f"{'Mean (all images)':<30} | {total_metrics['iou']/count:.4f}   | {total_metrics['dice']/count:.4f}   | "
            f"{total_metrics['f1']/count:.4f}   | {total_metrics['prec']/count:.4f}   | "
            f"{total_metrics['rec']/count:.4f}   | -          | {total_aog_count/count:>9.1f}"
        )
        print("=" * 108)

        summary = {
            # UNet++ specific model label
            "model": "UNetPP_resnet34",
            "num_images": count,
            "mean_iou": f"{total_metrics['iou']/count:.6f}",
            "mean_dice": f"{total_metrics['dice']/count:.6f}",
            "mean_f1": f"{total_metrics['f1']/count:.6f}",
            "mean_precision": f"{total_metrics['prec']/count:.6f}",
            "mean_recall": f"{total_metrics['rec']/count:.6f}",
            "mean_aog_area_percent": f"{sum([r['aog_area_percent'] for r in rows])/count:.6f}",
            "mean_aog_count": f"{total_aog_count/count:.6f}",
            "threshold": EVAL_THRESHOLD,
            "image_size": 256,
        }

        _write_metrics_files(output_folder, rows, summary)

        y_true = np.concatenate(all_y_true).astype(np.uint8)
        y_score = np.concatenate(all_y_score).astype(np.float32)
        _save_pr_curve(y_true, y_score, os.path.join(output_folder, "pr_curve.png"))
        _save_confusion_matrix_plot(
            cm_tp, cm_fp, cm_tn, cm_fn, os.path.join(output_folder, "confusion_matrix.png")
        )

        with open(os.path.join(output_folder, "confusion_matrix.txt"), "w", encoding="utf-8") as f:
            f.write(f"TN: {cm_tn}\n")
            f.write(f"FP: {cm_fp}\n")
            f.write(f"FN: {cm_fn}\n")
            f.write(f"TP: {cm_tp}\n")

        _write_gt_pred_compare(output_folder, rows)
        _save_gt_pred_image_compare(
            gt_pred_vis_items,
            os.path.join(output_folder, "gt_pred_image_compare.png"),
        )

        print(f"Saved PR curve: {os.path.join(output_folder, 'pr_curve.png')}")
        print(f"Saved confusion matrix: {os.path.join(output_folder, 'confusion_matrix.png')}")
        print(
            f"Saved GT-vs-Pred area/count comparison: {os.path.join(output_folder, 'gt_pred_area_count_compare.png')}"
        )
        print(f"Saved GT-vs-Pred image comparison: {os.path.join(output_folder, 'gt_pred_image_compare.png')}")
        print(f"Saved metrics: {os.path.join(output_folder, 'metrics_per_image.csv')}")
        print(f"Saved summary: {os.path.join(output_folder, 'metrics_summary.txt')}")
    else:
        print("No matching image pairs found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate UNet++ (ResNet34) for AOG segmentation.")
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=EVAL_THRESHOLD,
        help="Binarization threshold for validation/test metrics and masks.",
    )
    args = parser.parse_args()

    # Allow threshold override from CLI for multi-threshold experiments.
    EVAL_THRESHOLD = float(args.eval_threshold)
    epochs = 100

    train_imgs = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/images/"
    train_gt = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/GT/"

    test_imgs = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/"
    test_gt = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/"

    base_results_dir = os.path.join("outputs", "unet_results")
    os.makedirs(base_results_dir, exist_ok=True)
    print(f"Using eval threshold: {EVAL_THRESHOLD}")

    # UNet++ specific experiment folder naming
    run_name = (
        f"unetpp_resnet34_thr{_threshold_tag(EVAL_THRESHOLD)}_"
        f"train_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result_folder = os.path.join(base_results_dir, run_name)
    print("=" * 60)
    print(f"UNetPP_resnet34 result folder: {result_folder}")
    print("=" * 60)

    if not os.path.exists(train_imgs):
        print("Error: training path does not exist.")
    else:
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        full_dataset = AOGDataset(img_dir=train_imgs, mask_dir=train_gt, augment=False)
        n = len(full_dataset)
        idxs = list(range(n))
        random.shuffle(idxs)
        split = int(n * 0.8)

        train_dataset = AOGDataset(img_dir=train_imgs, mask_dir=train_gt, augment=True)
        val_dataset = AOGDataset(img_dir=train_imgs, mask_dir=train_gt, augment=False)

        train_sub = Subset(train_dataset, idxs[:split])
        val_sub = Subset(val_dataset, idxs[split:])
        train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=16, shuffle=False)

        # UNet++ specific model definition
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

        best_ckpt_path = os.path.join(result_folder, "best_model_unetpp_resnet34.pth")

        experiment_config = {
            "script": "UNetPP_resnet34.py",
            "model": {
                "name": "UNetPP_resnet34",
                "arch": "smp.UnetPlusPlus",
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "classes": 1,
                "activation": "sigmoid",
            },
            "training": {
                "seed": seed,
                "epochs": epochs,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "optimizer": "Adam",
                "loss": "Focal (gamma=2.0,pos_weight=10.0) + Tversky (alpha=0.4,beta=0.6)",
                "eval_threshold": EVAL_THRESHOLD,
                "min_component_area": MIN_COMPONENT_AREA,
                "val_ratio": 0.2,
                "train_augment": True,
                "val_augment": False,
                "device_policy": "MPS > CUDA > CPU",
            },
            "data": {
                "train_images_dir": train_imgs,
                "train_masks_dir": train_gt,
                "test_images_dir": test_imgs,
                "test_masks_dir": test_gt,
                "img_size": 256,
                "num_total_samples": n,
                "num_train_samples": len(idxs[:split]),
                "num_val_samples": len(idxs[split:]),
                "preprocess": "median blur + CLAHE per channel",
            },
            "outputs": {
                "result_folder": result_folder,
                "best_checkpoint": best_ckpt_path,
                "artifacts": [
                    "masks/",
                    "overlays/",
                    "metrics_per_image.csv",
                    "metrics_summary.txt",
                    "pr_curve.png",
                    "confusion_matrix.png",
                    "confusion_matrix.txt",
                    "gt_pred_area_count_compare.csv",
                    "gt_pred_area_count_compare.png",
                    "gt_pred_image_compare.png",
                ],
            },
        }
        save_experiment_config(result_folder, experiment_config)

        train_model(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=epochs,
            best_ckpt_path=best_ckpt_path,
        )

        if os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path, map_location=get_device()))
            print(f"Loaded best checkpoint for final test: {best_ckpt_path}")
        else:
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

        batch_process_and_evaluate(model, test_imgs, test_gt, result_folder)
