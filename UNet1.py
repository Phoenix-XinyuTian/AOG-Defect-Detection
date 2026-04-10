import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import cv2
import numpy as np
import segmentation_models_pytorch as smp


# ================= 1. 数据集 / Dataset =================
class AOGDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(valid_extensions)
        ])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        # CLAHE 与主管线保持一致 / CLAHE consistent with main pipeline
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # 掩码文件名统一使用 .png 扩展名 / Mask filename always uses .png extension
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: raise FileNotFoundError(f"Failed to load: {img_name}")
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 与主管线一致的预处理：中值滤波 + CLAHE（逐通道）
        # Preprocessing consistent with main pipeline: median blur + CLAHE (per channel)
        image_proc = np.zeros_like(image)
        for c in range(3):
            ch = cv2.medianBlur(image[:, :, c], 3)
            image_proc[:, :, c] = self._clahe.apply(ch)
        image = image_proc

        image = image.transpose(2, 0, 1) / 255.0
        mask = np.expand_dims(mask, axis=0) / 255.0
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# ================= 2. 评价指标 / Metrics =================
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
    f1 = (2.0 * precision * recall) / (precision + recall + eps)  # F1-score
    return iou, dice, precision, recall, f1


def get_device():
    """优先使用 MPS（Apple GPU），其次 CUDA，最后 CPU。
    Prefer MPS (Apple GPU), then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ================= 3. 模型初始化与训练 / Model init & training =================
# 注意：模型在入口处创建，避免 import 时触发权重下载
# Note: model is instantiated in __main__ to avoid downloading weights on import


def train_model(model, train_loader, val_loader=None, epochs=20):
    device = get_device()
    if device.type == "mps":
        print("🚀 Apple Silicon GPU detected, using MPS")
    elif device.type == "cuda":
        print("🚀 NVIDIA GPU detected, using CUDA")
    else:
        print("🐢 No GPU detected, using CPU")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    print(f"Training on {device}...")
    for epoch in range(epochs):
        # ---- 训练阶段 / Train ----
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        # ---- 验证阶段 / Validation ----
        if val_loader is not None:
            model.eval()
            val_dices, val_ious, val_f1s, val_precs, val_recs = [], [], [], [], []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    preds = model(images).squeeze(1).cpu().numpy()  # 概率图 / probability maps
                    masks_np = masks.squeeze(1).numpy()
                    for p, g in zip(preds, masks_np):
                        pred_mask = (p > 0.5).astype(np.uint8) * 255
                        gt_mask   = (g * 255).astype(np.uint8)
                        iou, dice, prec, rec, f1 = calculate_metrics(pred_mask, gt_mask)
                        val_ious.append(iou); val_dices.append(dice)
                        val_precs.append(prec); val_recs.append(rec); val_f1s.append(f1)
            print(f"Epoch {epoch+1:03d}/{epochs} | train_loss={train_loss:.4f} "
                  f"| Dice={sum(val_dices)/len(val_dices):.4f} | IoU={sum(val_ious)/len(val_ious):.4f} "
                  f"| F1={sum(val_f1s)/len(val_f1s):.4f} | Prec={sum(val_precs)/len(val_precs):.4f} "
                  f"| Rec={sum(val_recs)/len(val_recs):.4f}")
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")


# ================= 4. 批量推理与逐图结果输出 / Batch inference & per-image output =================
def _save_overlay(ori_img_bgr, mask_uint8, out_path, alpha=0.5):
    """
    将预测掩码以红色叠加到原图上并保存。
    Overlay the prediction mask in red on the original image and save.
    """
    overlay = ori_img_bgr.copy()
    overlay[mask_uint8 > 0] = (0, 0, 255)  # 标红 AOG 区域 / Mark AOG regions red (BGR)
    blended = cv2.addWeighted(overlay, alpha, ori_img_bgr, 1 - alpha, 0)
    ok = cv2.imwrite(out_path, blended)
    if not ok:
        print(f"  ⚠️  Failed to save overlay: {out_path}")


def batch_process_and_evaluate(model, input_folder, gt_folder, output_folder):
    device = get_device()
    model.to(device)
    model.eval()

    # 创建掩码和叠加图输出子目录 / Create subdirs for masks and overlays
    masks_dir    = os.path.join(output_folder, "masks")
    overlays_dir = os.path.join(output_folder, "overlays")
    os.makedirs(masks_dir,    exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    print(f"Output masks   → {masks_dir}")
    print(f"Output overlays → {overlays_dir}")

    valid_ext = ('.png', '.jpg', '.jpeg', '.tif')
    img_list = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)])

    total_metrics = {"iou": 0, "dice": 0, "prec": 0, "rec": 0, "f1": 0}
    count = 0

    print("\n" + "=" * 95)
    # 表头 / Table header
    print(f"{'Filename':<30} | {'IoU':<8} | {'Dice':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AOG area %':<10}")
    print("-" * 95)

    for img_name in img_list:
        img_path = os.path.join(input_folder, img_name)
        gt_path = os.path.join(gt_folder, img_name)
        # GT 掩码统一尝试 .png 扩展名 / Try .png extension for GT mask if not found
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_folder, os.path.splitext(img_name)[0] + ".png")
        if not os.path.exists(gt_path): continue

        # 1. 模型推理 / Inference
        ori_img = cv2.imread(img_path)
        h, w = ori_img.shape[:2]
        resized = cv2.resize(ori_img, (256, 256))
        # 推理预处理与训练一致：中值滤波 + CLAHE / Inference preprocessing consistent with training
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc = np.zeros_like(resized)
        for c in range(3):
            ch = cv2.medianBlur(resized[:, :, c], 3)
            proc[:, :, c] = _clahe.apply(ch)
        img_tensor = proc.transpose(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()



        # 2. 二值化掩码 / Binary mask
        res_mask = (pred > 0.4).astype(np.uint8) * 255
        # 二值化阈値，可根据 precision/recall 调整 / Binarization threshold; tune using precision/recall


        res_mask = cv2.resize(res_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. 计算指标 / Metrics
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        iou, dice, prec, rec, f1 = calculate_metrics(res_mask, gt_img)

        # 4. 当前图像的 AOG 面积占比 / AOG area fraction for this image
        aog_area = (np.count_nonzero(res_mask) / res_mask.size) * 100

        # 5. 输出每张图像结果行 / Per-image result row
        print(f"{img_name:<30} | {iou:.4f}   | {dice:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}   | {aog_area:>8.2f}%")

        # 累加全数据集均値 / Accumulate for mean over dataset
        total_metrics["iou"]  += iou
        total_metrics["dice"] += dice
        total_metrics["prec"] += prec
        total_metrics["rec"]  += rec
        total_metrics["f1"]   += f1
        count += 1

        # 保存预测掩码 / Save predicted mask
        stem = os.path.splitext(img_name)[0]
        mask_out = os.path.join(masks_dir, f"{stem}_pred.png")
        ok = cv2.imwrite(mask_out, res_mask)
        if not ok:
            print(f"  ⚠️  Failed to save mask: {mask_out}")

        # 保存叠加可视化图 / Save overlay visualization
        overlay_out = os.path.join(overlays_dir, f"{stem}_overlay.png")
        _save_overlay(ori_img, res_mask, overlay_out)

    # 6. 汇总统计 / Summary
    if count > 0:
        print("-" * 95)
        print(f"{'Mean (all images)':<30} | {total_metrics['iou']/count:.4f}   | {total_metrics['dice']/count:.4f}   | "
              f"{total_metrics['f1']/count:.4f}   | {total_metrics['prec']/count:.4f}   | "
              f"{total_metrics['rec']/count:.4f}   | -")
        print("=" * 95)
    else:
        print("No matching image pairs found.")


# ================= 5. 入口 / Entry point =================
if __name__ == "__main__":
    # --- 第一步：训练数据路径 / Step 1: training data paths ---
    # 用于训练的图像和 GT 标注 / Images and GT used for training
    train_imgs = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/images/'
    train_gt = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/GT/'

    # --- 第二步：测试数据路径 / Step 2: test data paths ---
    # 用于评估的保留测试集 / Held-out images for evaluation
    test_imgs = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/'
    test_gt = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/'

    # 预测结果输出目录 / Output directory for predictions
    result_folder = '/Users/phoenix/Desktop/AOGs Detection/Test Results/410 epoch100/'
    # ------------------------------------------------------------

    # 1. 仅在训练集上训练 / Train on training set only
    if not os.path.exists(train_imgs):
        print("❌ Error: training path does not exist.")
    else:
        random.seed(42)  # 固定随机种子，保证可复现 / Fix seed for reproducibility
        full_dataset = AOGDataset(img_dir=train_imgs, mask_dir=train_gt)
        n = len(full_dataset)
        idxs = list(range(n))
        random.shuffle(idxs)
        split = int(n * 0.8)  # 80% 训练 / 20% 验证 / 80% train, 20% val
        train_sub = Subset(full_dataset, idxs[:split])
        val_sub   = Subset(full_dataset, idxs[split:])
        train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
        val_loader   = DataLoader(val_sub,   batch_size=16, shuffle=False)

        # 模型在此处创建，避免 import 时触发权重下载 / Model created here to avoid download on import
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation='sigmoid')

        # 训练循环 / Training loop
        train_model(model, train_loader, val_loader=val_loader, epochs=100)

        # 2. 在测试集上批量预测与评分 / Batch predict & score on test set
        batch_process_and_evaluate(model, test_imgs, test_gt, result_folder)
