import os
import cv2
import torch
import numpy as np

from unet import UNet
from dataset import SEMSegDataset
from metrics import dice_iou_from_logits, f1_precision_recall_from_logits

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

def main():
    images_dir = "data/images"
    masks_dir  = "data/masks"  # 若无 GT 掩码请设为 None / Set to None if no GT masks
    img_size = 256
    thr = 0.5

    os.makedirs("outputs/preds", exist_ok=True)
    os.makedirs("outputs/overlays", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型 / Load trained model
    model = UNet(base=32).to(device)
    model.load_state_dict(torch.load("outputs/unet_best.pth", map_location=device))
    model.eval()

    has_gt = masks_dir is not None and os.path.isdir(masks_dir)

    if has_gt:
        ds = SEMSegDataset(images_dir, masks_dir, img_size=img_size)
        dices, ious, f1s, precs, recs = [], [], [], [], []
        with torch.no_grad():
            for img, mask, name in ds:
                img_t = img.unsqueeze(0).to(device)   # 扩展 batch 维度 [1,1,H,W] / Add batch dimension [1,1,H,W]
                mask_t = mask.unsqueeze(0).to(device)

                logits = model(img_t)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                pred = (prob >= thr).astype(np.uint8)

                # 计算指标 / Compute metrics
                d, j = dice_iou_from_logits(logits, mask_t, thr=thr)
                f1, prec, rec = f1_precision_recall_from_logits(logits, mask_t, thr=thr)
                dices.append(d); ious.append(j)
                f1s.append(f1); precs.append(prec); recs.append(rec)

                # 保存预测掩码 / Save predicted mask
                cv2.imwrite(f"outputs/preds/{os.path.splitext(name)[0]}_pred.png", pred*255)

                # 保存叠加可视化图 / Save overlay visualization
                gray01 = img[0].cpu().numpy()
                save_overlay(gray01, pred, f"outputs/overlays/{os.path.splitext(name)[0]}_overlay.png")

        print(f"Test on labeled set: Dice={sum(dices)/len(dices):.4f} | IoU={sum(ious)/len(ious):.4f} "
              f"| F1={sum(f1s)/len(f1s):.4f} | Prec={sum(precs)/len(precs):.4f} | Rec={sum(recs)/len(recs):.4f}")

    else:
        ds = SEMSegDataset(images_dir, masks_dir=None, img_size=img_size)
        with torch.no_grad():
            for img, name in ds:
                img_t = img.unsqueeze(0).to(device)
                logits = model(img_t)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                pred = (prob >= thr).astype(np.uint8)

                cv2.imwrite(f"outputs/preds/{os.path.splitext(name)[0]}_pred.png", pred*255)
                gray01 = img[0].cpu().numpy()
                save_overlay(gray01, pred, f"outputs/overlays/{os.path.splitext(name)[0]}_overlay.png")

        print("Inference done (no GT). Saved masks & overlays to outputs/.")

if __name__ == "__main__":
    main()
