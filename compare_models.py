import csv
import os
from pathlib import Path


RESULTS_ROOT = Path("outputs/unet_results")
MODEL_PREFIXES = {
    "UNet": "unet_train_eval_",
    "UNet1_resnet34": "unet1_train_eval_",
    "UNetPP_resnet34": "unetpp_resnet34_train_eval_",
    "Intensity": "intensity_train_eval_",
}


def latest_run(prefix):
    if not RESULTS_ROOT.exists():
        return None
    runs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def read_summary(summary_path):
    out = {}
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    rows = []
    missing = []

    for model_name, prefix in MODEL_PREFIXES.items():
        run_dir = latest_run(prefix)
        if run_dir is None:
            missing.append(model_name)
            continue

        summary_path = run_dir / "metrics_summary.txt"
        if not summary_path.exists():
            missing.append(model_name)
            continue

        s = read_summary(summary_path)
        rows.append({
            "model": s.get("model", model_name),
            "run_dir": str(run_dir),
            "mean_iou": s.get("mean_iou", ""),
            "mean_dice": s.get("mean_dice", ""),
            "mean_f1": s.get("mean_f1", ""),
            "mean_precision": s.get("mean_precision", ""),
            "mean_recall": s.get("mean_recall", ""),
            "mean_aog_area_percent": s.get("mean_aog_area_percent", ""),
            "mean_aog_count": s.get("mean_aog_count", ""),
            "num_images": s.get("num_images", ""),
        })

    out_csv = RESULTS_ROOT / "model_comparison_latest.csv"
    out_md = RESULTS_ROOT / "model_comparison_latest.md"

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    headers = [
        "model",
        "run_dir",
        "num_images",
        "mean_iou",
        "mean_dice",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "mean_aog_area_percent",
        "mean_aog_count",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Model Comparison (Latest Runs)\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")

        if missing:
            f.write("\nMissing models: " + ", ".join(missing) + "\n")

    print(f"Saved comparison CSV: {out_csv}")
    print(f"Saved comparison Markdown: {out_md}")
    if missing:
        print("Missing models:", ", ".join(missing))


if __name__ == "__main__":
    main()
