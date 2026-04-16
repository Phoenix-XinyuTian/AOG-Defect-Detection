import os
import cv2
import torch
import numpy as np
from datetime import datetime

from unet import UNet
from dataset import SEMSegDataset
from metrics import dice_iou_from_logits, f1_precision_recall_from_logits, count_aog_regions

EVAL_THRESHOLD = 0.4

def save_overlay(gray01, pred01, out_path, alpha=0.5):
    """
    将预测掩码叠加到原始灰度图上并保存。
    Overlay the prediction mask on the original grayscale image and save.
    gray01: [H,W] 0..1
    pred01: [H,W] 0/1
    """
    gray = (gray01 * 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = rgb.copy()
    red[pred01 == 1] = (0, 0, 255)  # 将 AOG 区域标色为红色 (BGR) / Mark AOG regions red (BGR)
    overlay = cv2.addWeighted(red, alpha, rgb, 1-alpha, 0)
    cv2.imwrite(out_path, overlay)


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


def batch_process_and_evaluate(model, images_dir, masks_dir, output_folder, img_size=256, thr=0.5):
    os.makedirs(output_folder, exist_ok=True)
    masks_out_dir = os.path.join(output_folder, "masks")
    overlays_out_dir = os.path.join(output_folder, "overlays")
    os.makedirs(masks_out_dir, exist_ok=True)
    os.makedirs(overlays_out_dir, exist_ok=True)

    print(f"Output masks    -> {masks_out_dir}")
    print(f"Output overlays -> {overlays_out_dir}")

    device = next(model.parameters()).device
    model.eval()

    has_gt = masks_dir is not None and os.path.isdir(masks_dir)

    if has_gt:
        ds = SEMSegDataset(images_dir, masks_dir, img_size=img_size)
        dices, ious, f1s, precs, recs, aog_areas, aog_counts = [], [], [], [], [], [], []
        rows = []

        print("\n" + "=" * 108)
        print(f"{'Filename':<30} | {'IoU':<8} | {'Dice':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AOG area %':<10} | {'AOG count':<9}")
        print("-" * 108)

        with torch.no_grad():
            for img, mask, name in ds:
                img_t = img.unsqueeze(0).to(device)
                mask_t = mask.unsqueeze(0).to(device)

                logits = model(img_t)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                pred = (prob >= thr).astype(np.uint8)

                d, j = dice_iou_from_logits(logits, mask_t, thr=thr)
                f1, prec, rec = f1_precision_recall_from_logits(logits, mask_t, thr=thr)

                aog_area = (pred.sum() / pred.size) * 100
                aog_count = count_aog_regions((pred * 255).astype(np.uint8))

                dices.append(d)
                ious.append(j)
                f1s.append(f1)
                precs.append(prec)
                recs.append(rec)
                aog_areas.append(aog_area)
                aog_counts.append(aog_count)

                stem = os.path.splitext(name)[0]
                print(f"{stem:<30} | {j:.4f}   | {d:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}   | {aog_area:>8.2f}%   | {aog_count:>9d}")

                mask_path = os.path.join(masks_out_dir, f"{stem}_pred.png")
                overlay_path = os.path.join(overlays_out_dir, f"{stem}_overlay.png")
                cv2.imwrite(mask_path, pred * 255)
                gray01 = img[0].cpu().numpy()
                save_overlay(gray01, pred, overlay_path)

                rows.append({
                    "filename": stem,
                    "iou": j,
                    "dice": d,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "aog_area_percent": aog_area,
                    "aog_count": aog_count,
                })

        n = len(dices)
        print("-" * 108)
        print(f"{'Mean (all images)':<30} | {sum(ious)/n:.4f}   | {sum(dices)/n:.4f}   | "
              f"{sum(f1s)/n:.4f}   | {sum(precs)/n:.4f}   | {sum(recs)/n:.4f}   | {sum(aog_areas)/n:>8.2f}%   | {sum(aog_counts)/n:>9.1f}")
        print("=" * 108)

        summary = {
            "num_images": n,
            "mean_iou": f"{sum(ious)/n:.6f}",
            "mean_dice": f"{sum(dices)/n:.6f}",
            "mean_f1": f"{sum(f1s)/n:.6f}",
            "mean_precision": f"{sum(precs)/n:.6f}",
            "mean_recall": f"{sum(recs)/n:.6f}",
            "mean_aog_area_percent": f"{sum(aog_areas)/n:.6f}",
            "mean_aog_count": f"{sum(aog_counts)/n:.6f}",
            "threshold": thr,
            "image_size": img_size,
        }
        _write_metrics_files(output_folder, rows, summary)

    else:
        ds = SEMSegDataset(images_dir, masks_dir=None, img_size=img_size)
        with torch.no_grad():
            for img, name in ds:
                img_t = img.unsqueeze(0).to(device)
                logits = model(img_t)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                pred = (prob >= thr).astype(np.uint8)

                stem = os.path.splitext(name)[0]
                mask_path = os.path.join(masks_out_dir, f"{stem}_pred.png")
                overlay_path = os.path.join(overlays_out_dir, f"{stem}_overlay.png")
                cv2.imwrite(mask_path, pred * 255)

                gray01 = img[0].cpu().numpy()
                save_overlay(gray01, pred, overlay_path)

        print("Inference done (no GT). Saved masks & overlays.")

def main():
    images_dir = "data/images"
    masks_dir  = "data/masks"  # 若无 GT 掩码请设为 None / Set to None if no GT masks
    img_size = 256
    thr = EVAL_THRESHOLD

    base_results_dir = os.path.join("outputs", "unet_results")
    os.makedirs(base_results_dir, exist_ok=True)
    run_name = f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_folder = os.path.join(base_results_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型 / Load trained model
    model = UNet(base=32).to(device)
    model.load_state_dict(torch.load("outputs/unet_best.pth", map_location=device))
    batch_process_and_evaluate(model, images_dir, masks_dir, result_folder, img_size=img_size, thr=thr)
    print(f"Results saved to: {result_folder}")

if __name__ == "__main__":
    main()
