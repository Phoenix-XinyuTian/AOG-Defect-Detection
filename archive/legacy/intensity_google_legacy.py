import cv2
import numpy as np


def nothing(x):
    pass


# 1. 读取图像
input_path = '/Users/phoenix/Desktop/AOGs Detection/Dataset with GT/images/S8-FOV2-X10_Bilayer_10KA_256x256_layer0.png'  # 请替换为您的图片路径
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("错误：无法读取图片，请检查路径。")
else:
    # 2. 预处理：稍微加强去噪，有助于得到平滑的边缘
    # 如果图像还是太模糊，可以将 5 改为 3
    blurred = cv2.medianBlur(img, 5)

    # 3. 创建交互窗口
    cv2.namedWindow('AOG Threshold Tuning', cv2.WINDOW_NORMAL)

    # 4. 创建滑动条
    # 参数：名称, 窗口名, 默认值, 最大值, 回调函数
    cv2.createTrackbar('Threshold', 'AOG Threshold Tuning', 127, 255, nothing)
    cv2.createTrackbar('Min_Area', 'AOG Threshold Tuning', 10, 500, nothing)  # 可选：过滤微小噪点

    print("操作提示：")
    print(" - 拖动滑块调整阈值")
    print(" - 按 's' 键保存当前结果并退出")
    print(" - 按 'q' 键直接退出")

    while True:
        # 获取滑动条当前的数值
        t_value = cv2.getTrackbarPos('Threshold', 'AOG Threshold Tuning')
        min_area = cv2.getTrackbarPos('Min_Area', 'AOG Threshold Tuning')

        # 执行阈值分割 (这里使用常用的 Binary_Inv，即暗区变白)
        # 如果您的目标是亮的，请去掉 _INV
        _, thresh = cv2.threshold(blurred, t_value, 255, cv2.THRESH_BINARY_INV)

        # 形态学处理：进一步平滑
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填孔

        # 计算面积
        total_pixels = mask.size
        aog_pixels = np.count_nonzero(mask)
        ratio = (aog_pixels / total_pixels) * 100

        # 在图像上实时显示占比文字
        display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_img, f"Area Ratio: {ratio:.2f}%", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, f"Threshold: {t_value}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示预览
        cv2.imshow('AOG Threshold Tuning', display_img)

        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 保存
            cv2.imwrite('optimized_aog_mask.png', mask)
            print(f"结果已保存！最终阈值: {t_value}, 面积占比: {ratio:.2f}%")
            break
        elif key == ord('q'):  # 退出
            break

    cv2.destroyAllWindows()