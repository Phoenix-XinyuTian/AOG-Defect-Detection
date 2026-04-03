import cv2
import numpy as np

# 1. 读取 GT 黑白图像（灰度）
gt_path = "/Users/phoenix/Desktop/AOGs Detection/Dataset with GT/GT/S8-FOV2-X10_Bilayer_10KA_256x256_layer0.png"
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

# 2. 二值化（确保是 0/1）
# 假设白色是 255
binary_gt = (gt > 0).astype(np.uint8)

# 3. 计算面积
white_pixels = np.sum(binary_gt)        # 白色像素数
total_pixels = binary_gt.size            # 总像素数

area_ratio = white_pixels / total_pixels

print(f"AOG area ratio: {area_ratio:.4f} ({area_ratio*100:.2f}%)")
