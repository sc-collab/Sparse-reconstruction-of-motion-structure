import cv2
import numpy as np
import glob
import os

# 设置棋盘格大小（9x6）
CHECKERBOARD = (9, 6)

# 假设每格大小为1单位（如1cm）
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 存储3D点
imgpoints = []  # 存储2D点

# 设置图像文件夹路径（可根据需要修改）
img_dir = r"xiangcan/Aotemanxiangcan"  # ← ← ← 修改成你自己的文件夹路径
image_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

if not image_paths:
    print("未找到任何图像，请检查路径和文件格式")
    exit()

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(3, 30, 0.001))
        imgpoints.append(corners2)
        # 可视化角点（可选）
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('Corners', img)
        # cv2.waitKey(200)

# cv2.destroyAllWindows()

# 执行相机标定
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 输出和保存
print("Camera intrinsic matrix (K):")
print(K)
print("Distortion coefficients:")
print(dist)

np.savetxt("K.txt", K)
np.savetxt("distortion.txt", dist)
