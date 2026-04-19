import cv2
import numpy as np

# 1. Load ground-truth mask (grayscale)
gt_path = ('/Users/phoenix/Desktop/AOGs Detection/Dataset with GT/GT/S8-FOV2-X10_Bilayer_10KA_256x256_layer1.png')
gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

if gt_img is None:
    print("Could not read file; check the path.")
else:
    # 2. Count foreground (white) pixels
    # In a binary mask, foreground is typically 255 or any value > 0
    white_pixels = np.count_nonzero(gt_img)

    # 3. Total pixel count (rows * cols)
    total_pixels = gt_img.size

    # 4. Area fraction
    area_ratio = (white_pixels / total_pixels) * 100

    print(f"Foreground pixels: {white_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"AOG area fraction (GT): {area_ratio:.2f}%")
