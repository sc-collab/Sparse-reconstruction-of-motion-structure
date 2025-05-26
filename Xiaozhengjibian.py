import cv2
import numpy as np
import os

# 相机内参矩阵
K = np.array([[3.11660392e+03, 0.00000000e+00, 2.02906352e+03],
              [0.00000000e+00, 2.95632155e+03, 1.39631132e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# 畸变系数
dist_coeffs = np.array([0.22870696, -1.20894266, 0.00219493, 0.00248504, 2.18832691])

# 文件夹路径
input_folder = "Datasets/Anteman"  # 替换为你的图片文件夹路径
output_folder = "D:/3Dceshi/undistorted_images"  # 校正后的图片保存路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图片
for filename in os.listdir(input_folder):
    # 检查文件扩展名是否为图片格式
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)

        # 加载图像
        img = cv2.imread(file_path)

        # 检查图像是否加载成功
        if img is None:
            print(f"无法加载图像: {file_path}")
            continue

        # 校正畸变
        undistorted_img = cv2.undistort(img, cameraMatrix=K, distCoeffs=dist_coeffs)

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存校正后的图像
        cv2.imwrite(output_path, undistorted_img)
        print(f"校正后的图像已保存到: {output_path}")

print("所有图像处理完成！")