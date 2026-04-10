import os
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset


def _build_aug_pipeline():
    """
    构建训练期间的数据增强流水线（仅作用于训练集）。
    Build augmentation pipeline applied only during training.
    所有空间变换同步作用于图像和掩码 / All spatial transforms are applied jointly to image and mask.
    """
    return A.Compose([
        # ---- 空间变换 / Spatial transforms ----
        # SEM 图像无方向依赖，水平/垂直翻转安全 / SEM images are orientation-agnostic
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # 90° 随机旋转，保持晶粒形状不变形 / Random 90° rotation; preserves grain geometry
        A.RandomRotate90(p=0.5),
        # 小角度随机旋转，模拟样品微小偏转 / Small-angle rotation to simulate slight sample tilt
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),

        # ---- 强度/对比度变换（仅作用于图像）/ Intensity transforms (image only) ----
        # 随机亮度/对比度抖动，模拟 SEM 束流强度波动 / Simulate SEM beam intensity variation
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # 随机 Gamma 校正，处理不同加速电压下的灰度偏差 / Random gamma for voltage-dependent gray shift
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        # 高斯噪声，模拟 SEM 探测器散粒噪声 / Simulate SEM detector shot noise
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
        # 轻微高斯模糊，模拟焦距偏差 / Slight blur to simulate focus variation
        A.GaussianBlur(blur_limit=(3, 3), p=0.3),
    ], additional_targets={"mask": "mask"})  # 掩码同步变换 / Sync mask with image


class SEMSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, img_size=256, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.augment = augment

        # CLAHE：限制对比度的自适应直方图均衡，增强 SEM 局部对比度
        # CLAHE: Contrast Limited Adaptive Histogram Equalization for local SEM contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 仅训练集启用增强 / Augmentation pipeline, enabled for training set only
        self._aug = _build_aug_pipeline() if augment else None

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
        # 1. 中值滤波：去除 SEM 椒盐噪声，同时保留边缘
        #    Median blur: remove SEM salt-and-pepper noise while preserving edges
        img = cv2.medianBlur(img, 3)

        # 2. CLAHE：自适应对比度增强，改善局部曝光不均匀问题
        #    CLAHE: adaptive contrast enhancement to correct uneven local illumination
        img = self._clahe.apply(img)

        return img

    def __getitem__(self, idx):
        name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, name)

        # 以灰度模式读取 SEM 图像 / Read SEM image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 缩放至统一尺寸（INTER_AREA 适合缩小，抗混叠）
        # Resize to fixed size (INTER_AREA suited for downscaling, anti-aliasing)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # 统一预处理：去噪 + 对比度增强 / Uniform preprocessing: denoise + contrast enhancement
        img = self._preprocess_image(img)

        if self.masks_dir is None:
            # 推理模式，无掩码 / Inference mode, no mask
            if self._aug is not None:
                img = self._aug(image=img)["image"]
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

        # 数据增强（训练集专用，图像与掩码同步）
        # Augmentation for training set only; image and mask transformed jointly
        if self._aug is not None:
            result = self._aug(image=img, mask=mask)
            img, mask = result["image"], result["mask"]

        # 归一化到 [0,1] / Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]

        # 二值化掩码 {0,1} / Binarize mask to {0,1}
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # 增加通道维度 [1,H,W] / Add channel dimension [1,H,W]

        return img, mask, name
