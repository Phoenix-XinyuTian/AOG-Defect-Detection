import csv
import json
import os
import random
from datetime import datetime

import cv2
import numpy as np


SEED = 42
IMG_SIZE = 256
VAL_RATIO = 0.2

# Intensity model parameter search space
THRESHOLD_CANDIDATES = list(range(70, 181, 5))
MIN_AREA_CANDIDATES = [0, 10, 20, 40, 80, 120, 200]
KERNEL_SIZE = 3
USE_MEDIAN_BLUR = True
USE_CLAHE = True

# Paths aligned with UNet/UNet1
TRAIN_IMAGES_DIR = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/images/"
TRAIN_MASKS_DIR = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/GT/"
TEST_IMAGES_DIR = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/"
TEST_MASKS_DIR = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/"

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT)])


def ensure_mask_path(mask_dir, img_name):
    cand1 = os.path.join(mask_dir, img_name)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")
    if os.path.exists(cand2):
        return cand2
    return None


def preprocess_gray(gray):
    out = gray.copy()
    if USE_MEDIAN_BLUR:
        out = cv2.medianBlur(out, 3)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
    return out


def postprocess_binary(mask_uint8, min_area, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if min_area <= 0:
        return out

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    cleaned = np.zeros_like(out)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def predict_mask(gray, threshold, min_area, kernel_size):
    proc = preprocess_gray(gray)
    raw = (proc > threshold).astype(np.uint8) * 255
    return postprocess_binary(raw, min_area=min_area, kernel_size=kernel_size)


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


def save_overlay(gray, mask_uint8, out_path, alpha=0.5):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    overlay[mask_uint8 > 0] = (0, 0, 255)
    blend = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
    cv2.imwrite(out_path, blend)


def tune_params(val_items):
    best = {
        "dice": -1.0,
        "threshold": None,
        "min_area": None,
    }
    rows = []

    for thr in THRESHOLD_CANDIDATES:
        for min_area in MIN_AREA_CANDIDATES:
            dices = []
            for gray, gt in val_items:
                pred = predict_mask(gray, threshold=thr, min_area=min_area, kernel_size=KERNEL_SIZE)
                _, d, _, _, _ = calculate_metrics(pred, gt)
                dices.append(d)
            mean_dice = float(np.mean(dices)) if dices else 0.0
            rows.append((thr, min_area, mean_dice))
            if mean_dice > best["dice"]:
                best = {
                    "dice": mean_dice,
                    "threshold": thr,
                    "min_area": min_area,
                }

    rows.sort(key=lambda x: x[2], reverse=True)
    return best, rows


def load_gray_mask_pairs(images_dir, masks_dir, img_names, img_size):
    pairs = []
    for name in img_names:
        img_path = os.path.join(images_dir, name)
        gt_path = ensure_mask_path(masks_dir, name)
        if gt_path is None:
            continue
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gray is None or gt is None:
            continue
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        pairs.append((name, gray, gt))
    return pairs


def write_metrics_files(output_folder, rows, summary):
    per_image_csv = os.path.join(output_folder, "metrics_per_image.csv")
    summary_txt = os.path.join(output_folder, "metrics_summary.txt")

    with open(per_image_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "iou", "dice", "f1", "precision", "recall", "aog_area_percent", "aog_count"
        ])
        for r in rows:
            writer.writerow([
                r["filename"],
                f"{r['iou']:.6f}",
                f"{r['dice']:.6f}",
                f"{r['f1']:.6f}",
                f"{r['precision']:.6f}",
                f"{r['recall']:.6f}",
                f"{r['aog_area_percent']:.6f}",
                int(r["aog_count"]),
            ])

    with open(summary_txt, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def write_config(output_folder, config):
    with open(os.path.join(output_folder, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def write_tuning(output_folder, tuning_rows):
    path = os.path.join(output_folder, "tuning_results.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "min_area", "mean_val_dice"])
        for thr, min_area, d in tuning_rows:
            writer.writerow([thr, min_area, f"{d:.6f}"])


def evaluate_and_save(test_items, threshold, min_area, output_folder):
    masks_dir = os.path.join(output_folder, "masks")
    overlays_dir = os.path.join(output_folder, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    rows = []
    totals = {"iou": 0.0, "dice": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    total_aog_count = 0

    print("\n" + "=" * 108)
    print(f"{'Filename':<30} | {'IoU':<8} | {'Dice':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AOG area %':<10} | {'AOG count':<9}")
    print("-" * 108)

    for name, gray, gt in test_items:
        pred = predict_mask(gray, threshold=threshold, min_area=min_area, kernel_size=KERNEL_SIZE)
        iou, dice, prec, rec, f1 = calculate_metrics(pred, gt)
        aog_area = (np.count_nonzero(pred) / pred.size) * 100.0
        n_labels, _ = cv2.connectedComponents(pred)
        aog_count = n_labels - 1

        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(masks_dir, f"{stem}_pred.png"), pred)
        save_overlay(gray, pred, os.path.join(overlays_dir, f"{stem}_overlay.png"))

        print(f"{name:<30} | {iou:.4f}   | {dice:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}   | {aog_area:>8.2f}%   | {aog_count:>9d}")

        rows.append({
            "filename": stem,
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "aog_area_percent": aog_area,
            "aog_count": aog_count,
        })

        totals["iou"] += iou
        totals["dice"] += dice
        totals["f1"] += f1
        totals["precision"] += prec
        totals["recall"] += rec
        total_aog_count += aog_count

    n = len(rows)
    if n == 0:
        raise RuntimeError("No valid test pairs found.")

    print("-" * 108)
    print(
        f"{'Mean (all images)':<30} | {totals['iou']/n:.4f}   | {totals['dice']/n:.4f}   | "
        f"{totals['f1']/n:.4f}   | {totals['precision']/n:.4f}   | {totals['recall']/n:.4f}   | "
        f"{sum(r['aog_area_percent'] for r in rows)/n:>8.2f}%   | {total_aog_count/n:>9.1f}"
    )
    print("=" * 108)

    summary = {
        "model": "intensity_threshold_baseline",
        "num_images": n,
        "mean_iou": f"{totals['iou']/n:.6f}",
        "mean_dice": f"{totals['dice']/n:.6f}",
        "mean_f1": f"{totals['f1']/n:.6f}",
        "mean_precision": f"{totals['precision']/n:.6f}",
        "mean_recall": f"{totals['recall']/n:.6f}",
        "mean_aog_area_percent": f"{sum(r['aog_area_percent'] for r in rows)/n:.6f}",
        "mean_aog_count": f"{total_aog_count/n:.6f}",
        "threshold": threshold,
        "min_area": min_area,
        "image_size": IMG_SIZE,
    }

    write_metrics_files(output_folder, rows, summary)
    return summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    all_train_names = list_images(TRAIN_IMAGES_DIR)
    random.shuffle(all_train_names)

    split = int(len(all_train_names) * (1 - VAL_RATIO))
    fit_names = all_train_names[:split]
    val_names = all_train_names[split:]

    fit_items = load_gray_mask_pairs(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, fit_names, IMG_SIZE)
    val_items = [(g, m) for _, g, m in load_gray_mask_pairs(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, val_names, IMG_SIZE)]
    test_items = load_gray_mask_pairs(TEST_IMAGES_DIR, TEST_MASKS_DIR, list_images(TEST_IMAGES_DIR), IMG_SIZE)

    if len(val_items) == 0:
        raise RuntimeError("Validation set is empty. Check train paths and masks.")

    best, tuning_rows = tune_params(val_items)
    print(
        f"Best params on val -> threshold={best['threshold']}, min_area={best['min_area']}, "
        f"val_dice={best['dice']:.6f}"
    )

    run_name = f"intensity_train_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_folder = os.path.join("outputs", "unet_results", run_name)
    os.makedirs(result_folder, exist_ok=True)

    config = {
        "script": "intensity_model.py",
        "model": {
            "name": "intensity_threshold_baseline",
            "type": "non-learning",
            "in_channels": 1,
            "preprocess": {
                "median_blur": USE_MEDIAN_BLUR,
                "clahe": USE_CLAHE,
            },
            "morphology": {
                "kernel_size": KERNEL_SIZE,
                "open_close": True,
            },
        },
        "training": {
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "parameter_search": {
                "threshold_candidates": THRESHOLD_CANDIDATES,
                "min_area_candidates": MIN_AREA_CANDIDATES,
                "selection_metric": "mean_val_dice",
            },
            "selected_params": {
                "threshold": best["threshold"],
                "min_area": best["min_area"],
                "best_val_dice": best["dice"],
            },
        },
        "data": {
            "train_images_dir": TRAIN_IMAGES_DIR,
            "train_masks_dir": TRAIN_MASKS_DIR,
            "test_images_dir": TEST_IMAGES_DIR,
            "test_masks_dir": TEST_MASKS_DIR,
            "img_size": IMG_SIZE,
            "num_total_train_pool": len(all_train_names),
            "num_fit_samples": len(fit_items),
            "num_val_samples": len(val_items),
            "num_test_samples": len(test_items),
        },
        "outputs": {
            "result_folder": result_folder,
            "artifacts": [
                "masks/",
                "overlays/",
                "metrics_per_image.csv",
                "metrics_summary.txt",
                "experiment_config.json",
                "tuning_results.csv",
            ],
        },
    }

    write_config(result_folder, config)
    write_tuning(result_folder, tuning_rows)
    summary = evaluate_and_save(
        test_items=test_items,
        threshold=best["threshold"],
        min_area=best["min_area"],
        output_folder=result_folder,
    )

    print(f"Saved config: {os.path.join(result_folder, 'experiment_config.json')}")
    print(f"Saved tuning: {os.path.join(result_folder, 'tuning_results.csv')}")
    print(f"Saved summary: {os.path.join(result_folder, 'metrics_summary.txt')}")
    print(f"Intensity result folder: {result_folder}")
    print(f"Mean Dice on test: {summary['mean_dice']}")


if __name__ == "__main__":
    main()
