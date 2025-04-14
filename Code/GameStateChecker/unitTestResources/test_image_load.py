# -*- coding: utf-8 -*-


import cv2

# 指定文件路径
file_path = "p1.png"  # 如果文件和脚本在同一个目录下，可以直接使用相对路径

# 尝试加载图像
img = cv2.imread(file_path)

# 检查是否加载成功
if img is None:
    print("Failed to load the image. Please check the file path or file integrity.")
else:
    print("Image loaded successfully!")
    print(f"Image shape: {img.shape}")  # 打印图像的维度信息
