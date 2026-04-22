"""
Run test evaluation on a saved UNetPP_resnet34 checkpoint at multiple thresholds.
All results are written under one parent folder, one subfolder per threshold.

Usage:
    python test_multi_threshold.py
    python test_multi_threshold.py --checkpoint path/to/best_model.pth
    python test_multi_threshold.py --thresholds 0.5 0.6 0.7
"""

import argparse
import os
import sys
from datetime import datetime

import segmentation_models_pytorch as smp
import torch

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(__file__))

import UNetPP_resnet34 as M  # reuse helpers; __main__ block won't run on import

DEFAULT_CHECKPOINT = (
    "outputs/unet_results/"
    "unetpp_resnet34_thr0p45_train_eval_20260422_075629/"
    "best_model_unetpp_resnet34.pth"
)
TEST_IMGS = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/"
TEST_GT   = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/"


def main():
    parser = argparse.ArgumentParser(description="Multi-threshold test for UNetPP_resnet34.")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to the .pth checkpoint (default: latest thr0p45 run)."
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7],
        help="List of thresholds to evaluate (default: 0.5 0.6 0.7)."
    )
    parser.add_argument(
        "--min-area", type=int, default=0,
        help="Minimum connected-component area to keep (pixels). 0 = disabled."
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_out = os.path.join(
        os.path.dirname(__file__),
        "outputs", "unet_results",
        f"threshold_sweep_{timestamp}",
    )
    os.makedirs(parent_out, exist_ok=True)
    print(f"\nOutput root: {parent_out}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Thresholds : {args.thresholds}")
    print(f"Min area   : {args.min_area}")

    device = M.get_device()
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    # Override module-level globals so batch_process_and_evaluate picks them up
    M.MIN_COMPONENT_AREA = args.min_area

    for thr in args.thresholds:
        M.EVAL_THRESHOLD = thr
        tag = M._threshold_tag(thr)
        sub_out = os.path.join(parent_out, f"thr{tag}")
        os.makedirs(sub_out, exist_ok=True)

        print("=" * 70)
        print(f"  Threshold = {thr}  →  {sub_out}")
        print("=" * 70)
        M.batch_process_and_evaluate(model, TEST_IMGS, TEST_GT, sub_out)

    print(f"\nDone. All results in:\n  {parent_out}")


if __name__ == "__main__":
    main()
