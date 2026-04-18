import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".bmp")


def find_masks(gt_dir: Path):
    return sorted([p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])


def mean_stats_from_masks(mask_paths):
    if not mask_paths:
        raise RuntimeError("No GT mask files found.")

    area_percents = []
    counts = []

    for p in mask_paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {p}")

        binary = (mask > 0).astype(np.uint8)
        area_percent = float(np.count_nonzero(binary) / binary.size * 100.0)
        num_labels, _ = cv2.connectedComponents(binary * 255)
        aog_count = int(num_labels - 1)

        area_percents.append(area_percent)
        counts.append(aog_count)

    return {
        "num_images": len(mask_paths),
        "mean_aog_area_percent": float(np.mean(area_percents)),
        "mean_aog_count": float(np.mean(counts)),
    }


def read_summary(path: Path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def latest_run(results_root: Path, prefix: str):
    if not results_root.exists():
        return None
    runs = [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def compare_with_models(results_root: Path, gt_stats):
    model_prefixes = {
        "UNet": "unet_train_eval_",
        "UNet1_resnet34": "unet1_train_eval_",
        "UNetPP_resnet34": "unetpp_resnet34_train_eval_",
        "Intensity": "intensity_train_eval_",
    }

    rows = []
    gt_area = gt_stats["mean_aog_area_percent"]
    gt_count = gt_stats["mean_aog_count"]

    for model_name, prefix in model_prefixes.items():
        run_dir = latest_run(results_root, prefix)
        if run_dir is None:
            continue

        summary_path = run_dir / "metrics_summary.txt"
        if not summary_path.exists():
            continue

        s = read_summary(summary_path)
        try:
            pred_area = float(s.get("mean_aog_area_percent", "nan"))
            pred_count = float(s.get("mean_aog_count", "nan"))
        except ValueError:
            continue

        rows.append({
            "model": s.get("model", model_name),
            "run_dir": str(run_dir),
            "pred_mean_aog_area_percent": pred_area,
            "pred_mean_aog_count": pred_count,
            "gt_mean_aog_area_percent": gt_area,
            "gt_mean_aog_count": gt_count,
            "abs_error_area_percent": abs(pred_area - gt_area),
            "abs_error_count": abs(pred_count - gt_count),
        })

    return rows


def save_outputs(out_dir: Path, gt_stats, compare_rows):
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_json = out_dir / "gt_test_reference_stats.json"
    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(gt_stats, f, indent=2, ensure_ascii=False)

    gt_txt = out_dir / "gt_test_reference_stats.txt"
    with open(gt_txt, "w", encoding="utf-8") as f:
        f.write(f"num_images: {gt_stats['num_images']}\n")
        f.write(f"mean_aog_area_percent: {gt_stats['mean_aog_area_percent']:.6f}\n")
        f.write(f"mean_aog_count: {gt_stats['mean_aog_count']:.6f}\n")

    cmp_csv = out_dir / "model_vs_gt_area_count_latest.csv"
    if compare_rows:
        headers = [
            "model",
            "run_dir",
            "pred_mean_aog_area_percent",
            "pred_mean_aog_count",
            "gt_mean_aog_area_percent",
            "gt_mean_aog_count",
            "abs_error_area_percent",
            "abs_error_count",
        ]
        with open(cmp_csv, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in compare_rows:
                f.write(
                    f"{r['model']},{r['run_dir']},{r['pred_mean_aog_area_percent']:.6f},"
                    f"{r['pred_mean_aog_count']:.6f},{r['gt_mean_aog_area_percent']:.6f},"
                    f"{r['gt_mean_aog_count']:.6f},{r['abs_error_area_percent']:.6f},"
                    f"{r['abs_error_count']:.6f}\n"
                )

    return gt_json, gt_txt, cmp_csv


def main():
    parser = argparse.ArgumentParser(
        description="Compute ground-truth reference mean_aog_area_percent and mean_aog_count on test GT masks."
    )
    parser.add_argument(
        "--test-gt-dir",
        default="/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/",
        help="Directory of test GT masks.",
    )
    parser.add_argument(
        "--results-root",
        default="outputs/unet_results",
        help="Root of experiment result folders for optional model-vs-GT comparison.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/unet_results",
        help="Output directory for reference stats and comparison files.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.test_gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Test GT dir not found: {gt_dir}")

    mask_paths = find_masks(gt_dir)
    gt_stats = mean_stats_from_masks(mask_paths)

    results_root = Path(args.results_root)
    compare_rows = compare_with_models(results_root, gt_stats)

    gt_json, gt_txt, cmp_csv = save_outputs(Path(args.out_dir), gt_stats, compare_rows)

    print("=" * 64)
    print("Ground-Truth Reference (Test Set)")
    print("=" * 64)
    print(f"num_images: {gt_stats['num_images']}")
    print(f"mean_aog_area_percent: {gt_stats['mean_aog_area_percent']:.6f}")
    print(f"mean_aog_count: {gt_stats['mean_aog_count']:.6f}")
    print("=" * 64)
    print(f"Saved: {gt_json}")
    print(f"Saved: {gt_txt}")
    if compare_rows:
        print(f"Saved: {cmp_csv}")
    else:
        print("No model summary found for model-vs-GT comparison.")


if __name__ == "__main__":
    main()
