import csv
import glob
import json
import os
import re
import argparse
from pathlib import Path


DEFAULT_BASE_ROOT = Path("outputs/unet_results")
DEFAULT_COMPARE_SUBDIR = "Compare2"


def parse_threshold_from_dirname(name: str):
    m = re.search(r"_thr([0-9p]+)_", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


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


def load_gt_ref(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_latest_runs_by_threshold(results_root: Path):
    runs = [
        Path(p)
        for p in glob.glob(str(results_root / "unetpp_resnet34_thr*_train_eval_*"))
        if os.path.isdir(p)
    ]

    latest = {}
    for run in runs:
        thr = parse_threshold_from_dirname(run.name)
        if thr is None:
            continue
        if thr not in latest or run.stat().st_mtime > latest[thr].stat().st_mtime:
            latest[thr] = run

    return latest


def as_float(s, default=float("nan")):
    try:
        return float(s)
    except Exception:
        return default


def resolve_gt_ref(results_root: Path):
    # Prefer GT reference in current report folder, then parent folder.
    local = results_root / "gt_test_reference_stats.json"
    if local.exists():
        return local

    parent = results_root.parent / "gt_test_reference_stats.json"
    if parent.exists():
        return parent

    base = DEFAULT_BASE_ROOT / "gt_test_reference_stats.json"
    if base.exists():
        return base

    return local


def main():
    parser = argparse.ArgumentParser(
        description="Generate UNetPP threshold comparison report from a target experiment folder."
    )
    parser.add_argument(
        "--results-root",
        default="",
        help="Target folder containing UNetPP threshold runs. If empty, auto-uses outputs/unet_results/Compare2 when present.",
    )
    args = parser.parse_args()

    if args.results_root:
        results_root = Path(args.results_root)
    else:
        compare2 = DEFAULT_BASE_ROOT / DEFAULT_COMPARE_SUBDIR
        results_root = compare2 if compare2.exists() else DEFAULT_BASE_ROOT

    results_root.mkdir(parents=True, exist_ok=True)

    out_csv = results_root / "unetpp_threshold_comparison_latest.csv"
    out_md = results_root / "unetpp_threshold_comparison_latest.md"
    gt_ref_path = resolve_gt_ref(results_root)

    latest = collect_latest_runs_by_threshold(results_root)
    if not latest:
        print(f"No UNetPP threshold runs found under: {results_root}")
        return

    gt_ref = load_gt_ref(gt_ref_path)
    gt_area = gt_ref.get("mean_aog_area_percent") if gt_ref else None
    gt_count = gt_ref.get("mean_aog_count") if gt_ref else None

    rows = []
    for thr in sorted(latest.keys()):
        run_dir = latest[thr]
        summary_path = run_dir / "metrics_summary.txt"
        if not summary_path.exists():
            continue

        s = read_summary(summary_path)
        row = {
            "threshold": thr,
            "run_dir": str(run_dir),
            "mean_iou": as_float(s.get("mean_iou", "nan")),
            "mean_dice": as_float(s.get("mean_dice", "nan")),
            "mean_f1": as_float(s.get("mean_f1", "nan")),
            "mean_precision": as_float(s.get("mean_precision", "nan")),
            "mean_recall": as_float(s.get("mean_recall", "nan")),
            "mean_aog_area_percent": as_float(s.get("mean_aog_area_percent", "nan")),
            "mean_aog_count": as_float(s.get("mean_aog_count", "nan")),
        }

        if gt_area is not None and gt_count is not None:
            row["gt_mean_aog_area_percent"] = float(gt_area)
            row["gt_mean_aog_count"] = float(gt_count)
            row["abs_error_area_percent"] = abs(row["mean_aog_area_percent"] - float(gt_area))
            row["abs_error_count"] = abs(row["mean_aog_count"] - float(gt_count))
        else:
            row["gt_mean_aog_area_percent"] = float("nan")
            row["gt_mean_aog_count"] = float("nan")
            row["abs_error_area_percent"] = float("nan")
            row["abs_error_count"] = float("nan")

        rows.append(row)

    if not rows:
        print("No threshold runs with metrics_summary.txt found.")
        return

    best_dice = max(rows, key=lambda r: r["mean_dice"])["threshold"]
    best_iou = max(rows, key=lambda r: r["mean_iou"])["threshold"]

    has_gt = gt_area is not None and gt_count is not None
    if has_gt:
        best_area_err = min(rows, key=lambda r: r["abs_error_area_percent"])["threshold"]
        best_count_err = min(rows, key=lambda r: r["abs_error_count"])["threshold"]
    else:
        best_area_err = None
        best_count_err = None

    for r in rows:
        r["best_dice"] = "YES" if r["threshold"] == best_dice else ""
        r["best_iou"] = "YES" if r["threshold"] == best_iou else ""
        r["best_area_error"] = "YES" if has_gt and r["threshold"] == best_area_err else ""
        r["best_count_error"] = "YES" if has_gt and r["threshold"] == best_count_err else ""

    headers = [
        "threshold",
        "run_dir",
        "mean_iou",
        "mean_dice",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "mean_aog_area_percent",
        "mean_aog_count",
        "gt_mean_aog_area_percent",
        "gt_mean_aog_count",
        "abs_error_area_percent",
        "abs_error_count",
        "best_dice",
        "best_iou",
        "best_area_error",
        "best_count_error",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(v):
        if isinstance(v, float):
            if str(v) == "nan":
                return ""
            return f"{v:.6f}"
        return str(v)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# UNetPP Threshold Comparison (Latest Runs)\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[h]) for h in headers) + " |\n")

        f.write("\n")
        f.write(f"- Best Dice threshold: {best_dice}\n")
        f.write(f"- Best IoU threshold: {best_iou}\n")
        if has_gt:
            f.write(f"- Best area-error threshold (vs GT): {best_area_err}\n")
            f.write(f"- Best count-error threshold (vs GT): {best_count_err}\n")
        else:
            f.write("- GT reference not found; area/count error ranking skipped.\n")

    print(f"Report root: {results_root}")
    print(f"GT reference: {gt_ref_path}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
