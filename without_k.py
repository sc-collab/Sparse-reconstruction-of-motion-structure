import numpy as np
import cv2

obj_point = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
image_point = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]], dtype=np.float32)

if len(obj_point) < 4 or len(image_point) < 4:
    raise ValueError("输入的点数必须至少为4个")
if obj_point.dtype not in [np.float32, np.float64]:
    obj_point = obj_point.astype(np.float32)
if image_point.dtype not in [np.float32, np.float64]:
    image_point = image_point.astype(np.float32)

K = np.eye(3)
dist_coeff = np.zeros((4, 1))
_, rot_vector, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff)