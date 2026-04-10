import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from unet import UNet
from dataset import SEMSegDataset
from metrics import dice_iou_from_logits, f1_precision_recall_from_logits

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    # ======= 路径配置 / Paths =======
    images_dir = "data/images"
    masks_dir  = "data/masks"
    os.makedirs("outputs", exist_ok=True)

    # ======= 超参数配置 / Hyperparameters =======
    img_size = 256
    batch_size = 4
    epochs = 30
    lr = 1e-3
    val_ratio = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SEMSegDataset(images_dir, masks_dir, img_size=img_size)
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idxs[:split], idxs[split:]

    # 训练集启用数据增强，验证集不启用，确保评估指标可靠
    # Training set uses augmentation; validation set does not for reliable evaluation
    train_dataset = SEMSegDataset(images_dir, masks_dir, img_size=img_size, augment=True)
    val_dataset   = SEMSegDataset(images_dir, masks_dir, img_size=img_size, augment=False)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(val_dataset,   val_idx),   batch_size=batch_size, shuffle=False)

    model = UNet(base=32).to(device)  # base=32: 轻量化模型，加快训练 / Lighter model for faster runs
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = -1.0

    for ep in range(1, epochs + 1):
        # ---- 训练阶段 / Train ----
        model.train()
        total_loss = 0.0
        for img, mask, _ in train_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = criterion(logits, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))

        # ---- 验证阶段 / Validation ----
        model.eval()
        val_loss = 0.0
        dices, ious, f1s, precs, recs = [], [], [], [], []
        with torch.no_grad():
            for img, mask, _ in val_loader:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                loss = criterion(logits, mask)
                val_loss += loss.item()
                d, j = dice_iou_from_logits(logits, mask)
                f1, prec, rec = f1_precision_recall_from_logits(logits, mask)
                dices.append(d)
                ious.append(j)
                f1s.append(f1)
                precs.append(prec)
                recs.append(rec)

        val_loss = val_loss / max(1, len(val_loader))
        val_dice = sum(dices) / max(1, len(dices))
        val_iou  = sum(ious)  / max(1, len(ious))
        val_f1   = sum(f1s)   / max(1, len(f1s))
        val_prec = sum(precs) / max(1, len(precs))
        val_rec  = sum(recs)  / max(1, len(recs))

        print(f"Epoch {ep:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
              f"| Dice={val_dice:.4f} | IoU={val_iou:.4f} "
              f"| F1={val_f1:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f}")

        # ---- 保存最佳模型 / Save best model ----
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "outputs/unet_best.pth")
            print("  ✓ Saved: outputs/unet_best.pth")

    print(f"Done. Best val Dice = {best_val_dice:.4f}")

if __name__ == "__main__":
    main()
