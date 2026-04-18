# SEM-based AOG Segmentation Project

This project benchmarks multiple approaches for AOG region segmentation on SEM images:

- Pure U-Net (custom implementation)
- U-Net with ResNet34 encoder using segmentation-models-pytorch (named UNet1)
- U-Net++ with ResNet34 encoder using segmentation-models-pytorch (named UNetPP_resnet34)
- Intensity-threshold baseline

The task is binary segmentation (AOG vs background), with unified metric and artifact outputs for fair comparison.

## 1. Current Project Structure

- `train.py`: Train/evaluate pure U-Net pipeline (from `unet.py` + `dataset.py`)
- `infer.py`: Inference/evaluation utility for pure U-Net outputs
- `unet.py`: Pure U-Net model definition
- `dataset.py`: Grayscale SEM dataset loader + preprocessing + augmentation (for pure U-Net)
- `metrics.py`: Shared metric helpers (Dice, IoU, F1, Precision, Recall, AOG count)

- `UNet1.py`: UNet (SMP) + ResNet34 encoder pipeline
- `UNetPP_resnet34.py`: UNet++ (SMP) + ResNet34 encoder pipeline, supports `--eval-threshold`

- `intensity_model.py`: Intensity-threshold baseline with val-set tuning

- `compare_models.py`: Compare latest runs across UNet / UNet1 / UNetPP / Intensity
- `compute_test_gt_reference.py`: Compute GT reference mean AOG area/count on test set
- `unetpp_threshold_report.py`: Build threshold sweep comparison table for UNetPP runs

- `outputs/unet_results/`: Experiment artifacts and comparison reports

## 2. Environment Setup

### 2.1 Python Environment

Use Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Dependencies

Key packages (from `requirements.txt`):

- torch, torchvision
- opencv-python
- numpy
- segmentation-models-pytorch
- albumentations
- matplotlib
- scikit-image

## 3. Data Paths and Important Note

Several scripts currently use absolute local paths in code, for example:

- `/Desktop/AOGs Detection/Train and Test/train/images/`
- `/Desktop/AOGs Detection/Train and Test/train/GT/`
- `/Desktop/AOGs Detection/Train and Test/test/images/`
- `/Desktop/AOGs Detection/Train and Test/test/GT/`

Before running on another machine or dataset location, update the path variables in:

- `train.py`
- `UNet1.py`
- `UNetPP_resnet34.py`
- `intensity_model.py`
- `compute_test_gt_reference.py` (or pass CLI args)

## 4. Unified Output Convention

Each experiment run creates a timestamped folder under:

- `outputs/unet_results/`

Typical artifacts:

- `experiment_config.json`
- `best_model_*.pth` (for trainable models)
- `metrics_summary.txt`
- `metrics_per_image.csv`
- `masks/`
- `overlays/`
- `pr_curve.png`
- `confusion_matrix.png`
- `confusion_matrix.txt`
- `gt_pred_area_count_compare.csv`
- `gt_pred_area_count_compare.png`
- `gt_pred_image_compare.png`

## 5. Run Experiments

### 5.1 Pure U-Net

```bash
python train.py
```

### 5.2 UNet1 (SMP Unet + ResNet34)

```bash
python UNet1.py
```

### 5.3 UNet++ (SMP UnetPlusPlus + ResNet34)

Default threshold is defined in script, but can be overridden:

```bash
python UNetPP_resnet34.py --eval-threshold 0.4
```

Run multi-threshold sweep (example):

```bash
for t in 0.3 0.4 0.5 0.6 0.7; do
  python UNetPP_resnet34.py --eval-threshold "$t"
done
```

Run folder names include threshold tags (for example `thr0p4`), so results do not overwrite each other.

### 5.4 Intensity Baseline

```bash
python intensity_model.py
```

## 6. Reporting and Comparison

### 6.1 Compare Latest Model Runs

```bash
python compare_models.py
```

Outputs:

- `outputs/unet_results/model_comparison_latest.csv`
- `outputs/unet_results/model_comparison_latest.md`

### 6.2 Compute GT Reference for Area/Count

```bash
python compute_test_gt_reference.py
```

Optional args:

```bash
python compute_test_gt_reference.py \
  --test-gt-dir "<test_gt_dir>" \
  --results-root "outputs/unet_results" \
  --out-dir "outputs/unet_results"
```

Outputs:

- `gt_test_reference_stats.json`
- `gt_test_reference_stats.txt`
- `model_vs_gt_area_count_latest.csv`

### 6.3 UNetPP Threshold Sweep Report

```bash
python unetpp_threshold_report.py
```

Outputs:

- `outputs/unet_results/unetpp_threshold_comparison_latest.csv`
- `outputs/unet_results/unetpp_threshold_comparison_latest.md`

The report auto-marks:

- best Dice threshold
- best IoU threshold
- best area error threshold (vs GT, if GT reference exists)
- best count error threshold (vs GT, if GT reference exists)

## 7. Metric Definitions

Main segmentation metrics:

- IoU
- Dice
- F1
- Precision
- Recall

AOG-specific structure metrics:

- mean AOG area percent
- mean AOG connected-component count

## 8. Device Policy

Training scripts prioritize devices in this order:

1. Apple MPS
2. CUDA
3. CPU

So running in a background terminal still keeps hardware acceleration if MPS/CUDA is available.

## 9. Reproducibility Notes

- Random seed is set in core training scripts (typically 42)
- Validation split is done from train set (commonly 80/20)
- Best checkpoint is selected by validation Dice
- Final test evaluation uses best checkpoint, not the last epoch checkpoint

## 10. Suggested Next Improvements

- Move all hardcoded paths to command-line args or a single config file
- Add a single experiment launcher script to run all models and generate all reports
- Add automatic threshold optimization based on validation PR/F-beta criterion
- Add unit checks for missing masks/path mismatches before training starts
