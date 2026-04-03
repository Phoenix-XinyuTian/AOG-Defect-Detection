import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from unet import UNet
from dataset import SEMSegDataset
from metrics import dice_iou_from_logits

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    # ======= paths =======
    images_dir = "data/images"
    masks_dir  = "data/masks"
    os.makedirs("outputs", exist_ok=True)

    # ======= hyperparams =======
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

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = UNet(base=32).to(device)  # base=32: lighter model for faster runs
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = -1.0

    for ep in range(1, epochs + 1):
        # ---- train ----
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

        # ---- val ----
        model.eval()
        val_loss = 0.0
        dices, ious = [], []
        with torch.no_grad():
            for img, mask, _ in val_loader:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                loss = criterion(logits, mask)
                val_loss += loss.item()
                d, j = dice_iou_from_logits(logits, mask)
                dices.append(d)
                ious.append(j)

        val_loss = val_loss / max(1, len(val_loader))
        val_dice = sum(dices) / max(1, len(dices))
        val_iou  = sum(ious) / max(1, len(ious))

        print(f"Epoch {ep:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | Dice={val_dice:.4f} | IoU={val_iou:.4f}")

        # ---- save best ----
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "outputs/unet_best.pth")
            print("  ✓ Saved: outputs/unet_best.pth")

    print(f"Done. Best val Dice = {best_val_dice:.4f}")

if __name__ == "__main__":
    main()
