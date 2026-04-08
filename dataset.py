import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SEMSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        self.img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])
        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]

        if self.masks_dir is None:
            return img, name

        mask_path = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"GT mask not found: {mask_path}")

        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)  # 二值化掩码 {0,1} / Binarize mask to {0,1}
        mask = torch.from_numpy(mask).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]

        return img, mask, name
