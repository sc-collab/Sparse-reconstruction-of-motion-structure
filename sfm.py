import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt

class Image_loader():
    def __init__(self, img_dir:str, downscale_factor:float):
        # loading the Camera intrinsic parameters K

        # with open(img_dir + '\\K.txt') as f:
        #     self.K = np.array(list((map(lambda x:list(map(lambda x:float(x), x.strip().split(' '))),f.read().split('\n')))))
        #     self.image_list = []
        with open(img_dir + '\\K.txt') as f:
            self.K = np.array([
                list(map(float, line.strip().split()))
                for line in f
                if line.strip()  # 忽略空行
            ])
            self.image_list = []

        # Loading the set of images
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(img_dir + '\\' + image)
        
        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale()

    
    def downscale(self) -> None:
        '''
        Downscales the Image intrinsic parameter acc to the downscale factor
        '''
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor
    
    def downscale_image(self, image):
        for _ in range(1,int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image

class Sfm():
    def __init__(self, img_dir:str, downscale_factor:float = 2.0) -> None:
        '''
            Initialise and Sfm object.
        '''
        self.img_obj = Image_loader(img_dir,downscale_factor)

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
        '''
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, point cloud 
        '''
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    import numpy as np
    import cv2

    def PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector

        Args:
            obj_point (np.ndarray): 3D object points.
            image_point (np.ndarray): 2D image points.
            K (np.ndarray): Camera intrinsic matrix.
            dist_coeff (np.ndarray): Camera distortion coefficients.
            rot_vector (np.ndarray): Initial rotation vector.
            initial (int): Flag indicating whether it's the initial call.

        Returns:
            tuple: A tuple containing the rotation matrix, translation vector, image points, object points, and rotation vector.

        Raises:
            ValueError: If obj_point or image_point are not of the correct shape or have insufficient points.
        """

        # Ensure obj_point is a contiguous array of float32 type
        obj_point = np.ascontiguousarray(obj_point, dtype=np.float32)

        # Ensure image_point is a contiguous array of float32 type
        image_point = np.ascontiguousarray(image_point, dtype=np.float32)

        # Reshape obj_point to be a 2D array (number_of_points x 3)
        obj_point = obj_point.reshape(-1, 3)

        # Reshape image_point to be a 2D array (number_of_points x 2)
        image_point = image_point.reshape(-1, 2)

        # Reshape rot_vector to be a 1D array
        rot_vector = rot_vector.reshape(-1)

        # Check if obj_point and image_point have the same number of points
        if obj_point.shape[0] != image_point.shape[0]:
            raise ValueError("obj_point and image_point must have the same number of points")

        # Check if there are enough points for PnP
        if obj_point.shape[0] < 4:
            # 如果点数不足4个，重复现有的一些点以达到4个
            num_points = obj_point.shape[0]
            if num_points == 0:
                raise ValueError("No points provided for PnP")

            # 计算需要重复的次数以达到至少4个点
            repeat_times = max(1, (4 - num_points + num_points - 1) // num_points)

            # 重复点以达到至少4个点
            obj_point = np.tile(obj_point, (repeat_times + 1, 1))[:4]
            image_point = np.tile(image_point, (repeat_times + 1, 1))[:4]

            print(
                f"Warning: Only {num_points} points provided. Repeated points to meet the minimum requirement of 4 points.")

        # Use cv2.solvePnPRansac to find the object pose
        try:
            _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff,
                                                                         flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as e:
            print(f"An error occurred in solvePnPRansac: {e}")
            print("Returning default values.")
            return np.eye(3), np.zeros((3, 1), dtype=np.float32), image_point, obj_point, rot_vector

        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        # Filter inliers if available
        if inlier is not None:
            # Get indices of inlier points
            inlier_indices = inlier[:, 0]

            # Filter image_point to only include inliers
            image_point = image_point[inlier_indices]

            # Filter obj_point to only include inliers
            obj_point = obj_point[inlier_indices]

            # Filter rot_vector to only include inliers if initial == 1
            if initial == 1:
                rot_vector = rot_vector[inlier_indices]

        # Return the results
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector

        Args:
            obj_point (np.ndarray): 3D object points.
            image_point (np.ndarray): 2D image points.
            K (np.ndarray): Camera intrinsic matrix.
            dist_coeff (np.ndarray): Camera distortion coefficients.
            rot_vector (np.ndarray): Initial rotation vector.
            initial (int): Flag indicating whether it's the initial call.

        Returns:
            tuple: A tuple containing the rotation matrix, translation vector, image points, object points, and rotation vector.

        Raises:
            ValueError: If obj_point or image_point are not of the correct shape.
        """

        if initial == 1:
            # Ensure obj_point is a contiguous array of float32 type
            obj_point = np.ascontiguousarray(obj_point, dtype=np.float32)

            # Ensure image_point is a contiguous array of float32 type
            image_point = np.ascontiguousarray(image_point, dtype=np.float32)

            # Reshape obj_point to be a 2D array (number_of_points x 3)
            obj_point = obj_point.reshape(-1, 3)

            # Reshape image_point to be a 2D array (number_of_points x 2)
            image_point = image_point.reshape(-1, 2)

            # Reshape rot_vector to be a 1D array
            rot_vector = rot_vector.reshape(-1)

        # Check if obj_point and image_point have the same number of points
        if obj_point.shape[0] != image_point.shape[0]:
            raise ValueError("obj_point and image_point must have the same number of points")

        # Use cv2.solvePnPRansac to find the object pose
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff,
                                                                     flags=cv2.SOLVEPNP_ITERATIVE)

        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        # Filter inliers if available
        if inlier is not None:
            # Get indices of inlier points
            inlier_indices = inlier[:, 0]

            # Filter image_point to only include inliers
            image_point = image_point[inlier_indices]

            # Filter obj_point to only include inliers
            obj_point = obj_point[inlier_indices]

            # Filter rot_vector to only include inliers if initial == 1
            if initial == 1:
                rot_vector = rot_vector[inlier_indices]

        # Return the results
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector
    # def PnP(self, obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:
    #     '''
    #     Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
    #     returns rotational matrix, translational matrix, image points, object points, rotational vector
    #     '''
    #     if initial == 1:
    #         obj_point = obj_point[:, 0 ,:]
    #         image_point = image_point.T
    #         rot_vector = rot_vector.T
    #     _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
    #     # Converts a rotation matrix to a rotation vector or vice versa
    #     rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)
    #
    #     if inlier is not None:
    #         image_point = image_point[inlier[:, 0]]
    #         obj_point = obj_point[inlier[:, 0]]
    #         rot_vector = rot_vector[inlier[:, 0]]
    #     return rot_matrix, tran_vector, image_point, obj_point, rot_vector
    
    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
        '''
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        '''
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        calculates of the reprojection error during bundle adjustment
        returns error 
        '''
        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        '''
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        '''
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))

    def to_ply(self, path, point_cloud, colors) -> None:
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])


        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')


    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        '''
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3 
        '''
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2


    def find_features(self, image_0, image_1) -> tuple:
        '''
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        '''

        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])

    def __call__(self, enable_bundle_adjustment:boolean=False):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
    
        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4)) 
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        feature_0, feature_1 = self.find_features(image_0, image_1)

        # Essential matrix
        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]


        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
        #ideally error < 1
        print("REPROJECTION ERROR: ", error)
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2 
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.5
        for i in tqdm(range(total_images)):
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            features_cur, features_2 = self.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity = 0)
        
            
            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))
            # takes a long time to run
            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
                print("Bundle Adjusted error: ",error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector)) 
   


            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')

if __name__ == '__main__':
    sfm = Sfm("Datasets\\Aoteman01")
    sfm()

