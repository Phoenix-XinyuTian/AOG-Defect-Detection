# Safe Migration Notes (Phase 1)

This repository was reorganized with a non-breaking strategy.

## What changed

- Added structured folders:
  - `src/data`, `src/models`, `src/eval`, `src/reporting`, `src/baselines`
  - `scripts`
  - `apps`
  - `archive/legacy`
- Copied existing stable code into the new layout.
- Kept original root scripts unchanged for backward compatibility.
- Added wrapper entry scripts under `scripts/` to run existing workflows safely.

## Old -> New mapping (copies)

- `dataset.py` -> `src/data/dataset.py`
- `unet.py` -> `src/models/unet_basic.py`
- `metrics.py` -> `src/eval/metrics.py`
- `infer.py` -> `src/eval/infer.py`
- `intensity_model.py` -> `src/baselines/intensity.py`
- `compare_models.py` -> `src/reporting/compare_models.py`
- `unetpp_threshold_report.py` -> `src/reporting/threshold_report.py`
- `compute_test_gt_reference.py` -> `src/reporting/gt_reference.py`
- `sem_aog_comparison_gui.py` -> `apps/sem_aog_comparison_gui.py`
- `Intensity Detect.py` -> `archive/legacy/intensity_detect_legacy.py`
- `Intersity Google.py` -> `archive/legacy/intensity_google_legacy.py`
- `GT area.py` -> `archive/legacy/gt_area_legacy.py`

## New preferred entry points

- Basic UNet: `python scripts/train_unet_basic.py`
- ResNet34 UNet: `python scripts/train_unet_resnet34.py`
- ResNet34 UNet++: `python scripts/train_unetpp_resnet34.py`
- Intensity baseline: `python scripts/run_intensity_baseline.py`
- Compare latest models: `python scripts/compare_latest_models.py`
- UNetPP threshold report: `python scripts/build_unetpp_threshold_report.py`
- GT reference stats: `python scripts/compute_test_gt_reference.py`
- Figure GUI: `python scripts/launch_comparison_gui.py`

## Risk profile

- Low risk: no original file was moved or deleted.
- Existing old commands still work.
- New structure is ready for Phase 2 refactor (deduplicate imports and centralize common logic).

## Phase 2 progress

- Script entrypoints below now call new structured files directly:
  - `scripts/run_intensity_baseline.py` -> `src/baselines/intensity.py`
  - `scripts/compare_latest_models.py` -> `src/reporting/compare_models.py`
  - `scripts/build_unetpp_threshold_report.py` -> `src/reporting/threshold_report.py`
  - `scripts/compute_test_gt_reference.py` -> `src/reporting/gt_reference.py`
- Training wrappers remain backward-compatible and still target original training files for safety:
  - `scripts/train_unet_basic.py` -> `train.py`
  - `scripts/train_unet_resnet34.py` -> `UNet1.py`
  - `scripts/train_unetpp_resnet34.py` -> `UNetPP_resnet34.py`
