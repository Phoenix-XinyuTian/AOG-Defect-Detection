import os
import random
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from unet import UNet
from dataset import SEMSegDataset
from metrics import dice_iou_from_logits, f1_precision_recall_from_logits
from infer import batch_process_and_evaluate

EVAL_THRESHOLD = 0.4


def get_device():
    """Prefer MPS, then CUDA, else CPU to match UNet1 runtime policy."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_experiment_config(output_folder, config):
    os.makedirs(output_folder, exist_ok=True)
    cfg_path = os.path.join(output_folder, "experiment_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Saved config: {cfg_path}")


def bce_dice_loss(prob, target, eps=1e-6):
    """Binary segmentation combined loss: BCE + Dice."""
    bce = nn.functional.binary_cross_entropy(prob, target)
    prob_f = prob.view(prob.size(0), -1)
    tgt_f = target.view(target.size(0), -1)
    inter = (prob_f * tgt_f).sum(dim=1)
    dice = (2.0 * inter + eps) / (prob_f.sum(dim=1) + tgt_f.sum(dim=1) + eps)
    dice_loss = 1.0 - dice.mean()
    return bce + dice_loss

def main():
    seed = 42
    set_seed(seed)

    # ======= 路径配置 / Paths =======
    images_dir = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/images/"
    masks_dir  = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/train/GT/"
    test_images_dir = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/"
    test_masks_dir  = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/"
    os.makedirs("outputs", exist_ok=True)

    # ======= 超参数配置 / Hyperparameters =======
    img_size = 256
    batch_size = 16
    epochs = 100
    lr = 1e-4
    val_ratio = 0.2

    device = get_device()
    if device.type == "mps":
        print("🚀 Apple Silicon GPU detected, using MPS")
    elif device.type == "cuda":
        print("🚀 NVIDIA GPU detected, using CUDA")
    else:
        print("🐢 No GPU detected, using CPU")

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

    run_name = f"unet_train_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_folder = os.path.join("outputs", "unet_results", run_name)

    experiment_config = {
        "script": "train.py",
        "model": {
            "name": "UNet",
            "backbone": "none",
            "base_channels": 32,
            "in_channels": 1,
            "out_channels": 1,
        },
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss": "BCE + Dice (on sigmoid(logits))",
            "eval_threshold": EVAL_THRESHOLD,
            "val_ratio": val_ratio,
            "train_augment": True,
            "val_augment": False,
            "device": str(device),
        },
        "data": {
            "train_images_dir": images_dir,
            "train_masks_dir": masks_dir,
            "test_images_dir": test_images_dir,
            "test_masks_dir": test_masks_dir,
            "img_size": img_size,
            "num_total_samples": n,
            "num_train_samples": len(train_idx),
            "num_val_samples": len(val_idx),
            "split_indices_seeded_shuffle": True,
            "preprocess": "median blur + CLAHE",
        },
        "outputs": {
            "result_folder": result_folder,
            "best_weight_path": os.path.join(result_folder, "best_model_unet.pth"),
        },
    }
    save_experiment_config(result_folder, experiment_config)

    model = UNet(base=32).to(device)  # base=32: 轻量化模型，加快训练 / Lighter model for faster runs
    criterion = bce_dice_loss
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = -1.0
    best_ckpt_path = os.path.join(result_folder, "best_model_unet.pth")

    for ep in range(1, epochs + 1):
        # ---- 训练阶段 / Train ----
        model.train()
        total_loss = 0.0
        for img, mask, _ in train_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            probs = torch.sigmoid(logits)
            loss = criterion(probs, mask)
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
                probs = torch.sigmoid(logits)
                loss = criterion(probs, mask)
                val_loss += loss.item()
                d, j = dice_iou_from_logits(logits, mask, thr=EVAL_THRESHOLD)
                f1, prec, rec = f1_precision_recall_from_logits(logits, mask, thr=EVAL_THRESHOLD)
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
            torch.save(model.state_dict(), best_ckpt_path)
            torch.save(model.state_dict(), "outputs/unet_best.pth")  # backward compatibility for infer.py
            print(f"  ✓ Saved best: {best_ckpt_path}")

    print(f"Done. Best val Dice = {best_val_dice:.4f}")

    # ---- 测试集评估并导出与 UNet1 对齐的结果 / Test-set evaluation with UNet1-style outputs ----
    if os.path.isdir(test_images_dir) and os.path.isdir(test_masks_dir):
        best_model = UNet(base=32).to(device)
        best_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        batch_process_and_evaluate(
            best_model,
            test_images_dir,
            test_masks_dir,
            result_folder,
            img_size=img_size,
            thr=EVAL_THRESHOLD,
        )
        print(f"UNet test results saved to: {result_folder}")
    else:
        print("Skip test evaluation: data/test/images or data/test/masks not found.")

if __name__ == "__main__":
    main()
