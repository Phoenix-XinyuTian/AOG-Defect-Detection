import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp


# ================= 1. Dataset =================
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

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: raise FileNotFoundError(f"Failed to load: {img_name}")
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        image = image.transpose(2, 0, 1) / 255.0
        mask = np.expand_dims(mask, axis=0) / 255.0
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# ================= 2. Metrics =================
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
    return iou, dice, precision, recall


def get_device():
    """Prefer MPS (Apple GPU), then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ================= 3. Model init & training =================
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation='sigmoid')


def train_model(model, train_loader, epochs=20):
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
    model.train()
    print(f"Training on {device}...")
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# ================= 4. Batch inference & per-image output =================
def batch_process_and_evaluate(model, input_folder, gt_folder, output_folder):
    device = get_device()
    model.to(device)
    model.eval()

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    valid_ext = ('.png', '.jpg', '.jpeg', '.tif')
    img_list = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)])

    total_metrics = {"iou": 0, "dice": 0, "prec": 0, "rec": 0}
    count = 0

    print("\n" + "=" * 80)
    # Table header
    print(f"{'Filename':<20} | {'IoU':<8} | {'Dice':<8} | {'AOG area %':<12}")
    print("-" * 80)

    for img_name in img_list:
        img_path = os.path.join(input_folder, img_name)
        gt_path = os.path.join(gt_folder, img_name)

        if not os.path.exists(gt_path): continue

        # 1. Inference
        ori_img = cv2.imread(img_path)
        h, w = ori_img.shape[:2]
        img_tensor = cv2.resize(ori_img, (256, 256)).transpose(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()



        # 2. Binary mask
        res_mask = (pred > 0.4).astype(np.uint8) * 255
        # Binarization threshold; tune using precision/recall


        res_mask = cv2.resize(res_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Metrics
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        iou, dice, prec, rec = calculate_metrics(res_mask, gt_img)

        # 4. AOG area fraction for this image
        aog_area = (np.count_nonzero(res_mask) / res_mask.size) * 100

        # 5. Per-image row
        print(f"{img_name:<20} | {iou:.4f} | {dice:.4f} | {aog_area:>10.2f}%")

        # Accumulate for mean over dataset
        total_metrics["iou"] += iou
        total_metrics["dice"] += dice
        total_metrics["prec"] += prec
        total_metrics["rec"] += rec
        count += 1

        # Save mask
        cv2.imwrite(os.path.join(output_folder, f"mask_{img_name}"), res_mask)

    # 6. Summary
    if count > 0:
        print("-" * 80)
        print(
            f"{'Mean (all images)':<20} | {total_metrics['iou'] / count:.4f} | {total_metrics['dice'] / count:.4f} | done")
        print("=" * 80)
        print(f"Mean Precision: {total_metrics['prec'] / count:.4f} | Mean Recall: {total_metrics['rec'] / count:.4f}")
    else:
        print("No matching image pairs found.")


# ================= 5. Entry point =================
if __name__ == "__main__":
    # --- Step 1: training data paths ---
    # Images and GT used for optimization
    train_imgs = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/images/'
    train_gt = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/GT/'

    # --- Step 2: test data paths ---
    # Held-out images for evaluation
    test_imgs = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/'
    test_gt = '/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/'

    # Output directory for predictions
    result_folder = '/Users/phoenix/Desktop/AOGs Detection/Test Results/38Test 249Train epoch50/'
    # ------------------------------------------------------------

    # 1. Train (training set only)
    if not os.path.exists(train_imgs):
        print("❌ Error: training path does not exist.")
    else:
        dataset = AOGDataset(img_dir=train_imgs, mask_dir=train_gt)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Training loop
        train_model(model, loader, epochs=100)

        # 2. Batch predict & score on test set
        batch_process_and_evaluate(model, test_imgs, test_gt, result_folder)
