import os
import cv2
import numpy as np
from scipy.optimize import least_squares

class Image_loader:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = self.load_images()

    def load_images(self):
        image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        return [cv2.imread(os.path.join(self.img_dir, file)) for file in image_files]

class Sfm:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = Image_loader(img_dir).images
        self.K = np.loadtxt(os.path.join(self.img_dir, 'Datasets/Anteman/K.txt'))
        self.keypoints = []
        self.descriptors = []
        self.matches = []
        self.poses = []
        self.points_3d = []

    def feature_extract(self):
        sift = cv2.SIFT_create()
        for img in self.images:
            kp, des = sift.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)

    def feature_match(self):
        bf = cv2.BFMatcher()
        for i in range(len(self.images) - 1):
            matches = bf.knnMatch(self.descriptors[i], self.descriptors[i + 1], k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            self.matches.append(good)

    # def triangulation(self, p1, p2, M1, M2):
    #     points_4d = cv2.triangulatePoints(M1, M2, p1, p2)
    #     return (points_4d / points_4d[3])[:3].T  # shape: Nx3
    def triangulation(self, p1, p2, M1, M2):
        """
        p1, p2: (N, 2) -- np.ndarray
        M1, M2: (3, 4) -- projection matrices
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        M1 = np.asarray(M1, dtype=np.float64)
        M2 = np.asarray(M2, dtype=np.float64)

        if p1.shape[0] < 2 or p2.shape[0] < 2:
            return np.empty((0, 3))

        # 转置为 (2, N)
        p1 = p1.T
        p2 = p2.T

        # 调用函数
        points_4d = cv2.triangulatePoints(M1, M2, p1, p2)

        # 齐次坐标归一化
        points_3d = (points_4d[:3] / points_4d[3]).T  # 变为 (N, 3)
        return points_3d

    def PnP(self, object_points, image_points):
        if len(object_points) < 4 or len(image_points) < 4:
            return None, None, False
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.K, None)
        if not retval:
            return None, None, False
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        return R, t, True

    def bundle_adjustment(self, points_3d, points_2d, K, R, t):
        def residuals(params):
            rvec, tvec = params[:3], params[3:6]
            proj_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
            return (proj_points.squeeze() - points_2d).ravel()

        rvec, _ = cv2.Rodrigues(R)
        params_init = np.hstack((rvec.ravel(), t.ravel()))
        result = least_squares(residuals, params_init)
        rvec_opt = result.x[:3]
        tvec_opt = result.x[3:6].reshape(3, 1)
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        return R_opt, tvec_opt

    def __call__(self):
        self.feature_extract()
        self.feature_match()

        # 初始化前两帧
        kp1 = self.keypoints[0]
        kp2 = self.keypoints[1]
        matches = self.matches[0]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        M2 = np.hstack((R, t))

        proj1 = self.K @ M1
        proj2 = self.K @ M2

        pts1_inliers = pts1[mask_pose.ravel() == 1]
        pts2_inliers = pts2[mask_pose.ravel() == 1]

        # points_3d = self.triangulation(pts1_inliers.T, pts2_inliers.T, proj1, proj2)
        points_3d = self.triangulation(pts1_inliers, pts2_inliers, proj1, proj2)

        self.poses.append((np.eye(3), np.zeros((3, 1))))
        self.poses.append((R, t))
        self.points_3d = points_3d

        # 后续帧逐帧加入
        for i in range(2, len(self.images)):
            kp = self.keypoints[i]
            des = self.descriptors[i]
            last_kp = self.keypoints[i - 1]
            last_des = self.descriptors[i - 1]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(last_des, des, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) < 4:
                print(f"帧 {i} 特征匹配太少，跳过。")
                continue

            pts_3d = []
            pts_2d = []
            for m in good:
                idx_3d = m.queryIdx
                if idx_3d >= len(self.points_3d):
                    continue
                pts_3d.append(self.points_3d[idx_3d])
                pts_2d.append(kp[m.trainIdx].pt)

            pts_3d = np.array(pts_3d)
            pts_2d = np.array(pts_2d)

            R, t, success = self.PnP(pts_3d, pts_2d)
            if not success:
                print(f"帧 {i} PnP 失败，跳过。")
                continue

            R, t = self.bundle_adjustment(pts_3d, pts_2d, self.K, R, t)
            self.poses.append((R, t))

        self.save_point_cloud()

    def save_point_cloud(self, filename='result.ply'):
        with open(filename, 'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(self.points_3d)))
            f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
            for p in self.points_3d:
                f.write(f'{p[0]} {p[1]} {p[2]}\n')
sfm = Sfm('Datasets\\IGBT')
sfm()

