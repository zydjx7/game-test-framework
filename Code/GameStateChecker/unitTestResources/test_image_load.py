# -*- coding: utf-8 -*-


import cv2

# ָ���ļ�·��
file_path = "p1.png"  # ����ļ��ͽű���ͬһ��Ŀ¼�£�����ֱ��ʹ�����·��

# ���Լ���ͼ��
img = cv2.imread(file_path)

# ����Ƿ���سɹ�
if img is None:
    print("Failed to load the image. Please check the file path or file integrity.")
else:
    print("Image loaded successfully!")
    print(f"Image shape: {img.shape}")  # ��ӡͼ���ά����Ϣ
