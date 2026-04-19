import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from src.data.unetpp_data_policy import preprocess_gray_unetpp, augment_gray_mask_unetpp


class SEMSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, img_size=256, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.augment = augment

        # CLAHE：限制对比度的自适应直方图均衡，增强 SEM 局部对比度
        # CLAHE: Contrast Limited Adaptive Histogram Equalization for local SEM contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])
        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.img_files)

    def _preprocess_image(self, img):
        """
        对所有集合（训练/验证/测试）统一执行的图像预处理。
        Preprocessing applied to all splits (train / val / test).
        """
        return preprocess_gray_unetpp(img, img_size=(self.img_size, self.img_size), clahe=self._clahe)

    def __getitem__(self, idx):
        name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, name)

        # 以灰度模式读取 SEM 图像 / Read SEM image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 统一预处理：UNetPP 风格的 resize + median blur + CLAHE。
        # Unified preprocessing with UNetPP policy.
        img = self._preprocess_image(img)

        if self.masks_dir is None:
            # 推理模式，无掩码 / Inference mode, no mask
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]
            return img, name

        # 读取 GT 掩码 / Load GT mask
        mask_path = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"GT mask not found: {mask_path}")

        # 最近邻插值缩放，保持二值边界清晰 / Nearest-neighbor resize to preserve binary boundaries
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 数据增强（训练集专用，遵循 UNetPP 当前策略）
        # Training-only augmentation with UNetPP-style policy.
        if self.augment:
            img, mask = augment_gray_mask_unetpp(img, mask)

        # 归一化到 [0,1] / Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]

        # 二值化掩码 {0,1} / Binarize mask to {0,1}
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]

        return img, mask, name
