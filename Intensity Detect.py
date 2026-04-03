import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import morphology

# =========================
# 用户可调参数（重点）
# =========================
IMAGE_PATH = os.path.expanduser("/Users/phoenix/Desktop/AOGs Detection/Dataset with GT/GT/S8-FOV2-X10_Bilayer_10KA_256x256_layer0.png")

INTENSITY_THRESHOLD = 116   # ⭐ 手动阈值（越大 → AOG 越少）
MIN_AREA =20000              # ⭐ 最小晶粒面积（越大 → 碎点越少）

KERNEL_SIZE = 3             # 形态学核大小
USE_BLUR = True             # 是否先降噪

# =========================
# 1. 读取图像
# =========================
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found.")

# =========================
# 2. 预处理（降噪）
# =========================
img_proc = img.copy()

if USE_BLUR:
    img_proc = cv2.medianBlur(img_proc, 3)
    img_proc = cv2.GaussianBlur(img_proc, (5, 5), 0)

# =========================
# 3. 手动灰度阈值
# =========================
binary = img_proc > INTENSITY_THRESHOLD
binary = binary.astype(bool)

# =========================
# 4. 形态学操作（减少碎点）
# =========================
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (KERNEL_SIZE, KERNEL_SIZE)
)

binary_uint8 = (binary * 255).astype(np.uint8)

# 开运算：去碎点
binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel)

# 闭运算：填补小孔
binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)

# =========================
# 5. 连通域面积过滤（最关键）
# =========================
lab = label(binary_uint8 > 0)
clean = np.zeros_like(binary_uint8)

for r in regionprops(lab):
    if r.area >= MIN_AREA:
        clean[lab == r.label] = 255

# =========================
# 6. AOG 面积计算
# =========================
aog_pixels = np.sum(clean == 255)
total_pixels = clean.size
aog_area_percent = 100.0 * aog_pixels / total_pixels

# =========================
# 7. 可视化
# =========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original SEM")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("AOG Mask (Manual Threshold)")
plt.imshow(clean, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay[clean == 255] = [255, 0, 0]
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

# =========================
# 8. 输出
# =========================
print(f"Manual intensity threshold: {INTENSITY_THRESHOLD}")
print(f"Min region area: {MIN_AREA}")
print(f"AOG area percentage: {aog_area_percent:.2f}%")

cv2.imwrite("AOG_mask_manual_threshold.png", clean)
