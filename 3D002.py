import cv2
import numpy as np
import os
import json
import pyvista as pv
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from pathlib import Path


class SfmMvs:
    def __init__(self, image_dir, output_dir, intrinsic_matrix):
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.images = []
        self.keypoints_list = []
        self.descriptors_list = []
        self.matches = []
        self.poses = []
        self.K = intrinsic_matrix

    def load_images(self):
        """ Load images from folder """
        for img_file in sorted(os.listdir(self.image_dir)):
            if img_file.lower().endswith(".jpg"):
                img_path = os.path.join(self.image_dir, img_file)
                self.images.append(cv2.imread(img_path))
        print(f"Load {len(self.images)} images。")

    # def detect_features(self):
    #     sift = cv2.SIFT_create()
    #     batch_size = 50
    #     for i in range(0, len(self.images), batch_size):
    #         batch_images = self.images[i:i + batch_size]
    #         batch_keypoints = []
    #         batch_descriptors = []
    #         for img in batch_images:
    #             keypoints, descriptors = sift.detectAndCompute(img, None)
    #             batch_keypoints.append(keypoints)
    #             batch_descriptors.append(descriptors)
    #         self.keypoints_list.extend(batch_keypoints)
    #         self.descriptors_list.extend(batch_descriptors)
    #         # 保存当前批次的结果
    #         features_path = os.path.join(self.output_dir, f"features_{i}.json")
    #         features_data = {
    #             "features": [
    #                 {
    #                     "keypoints": [kp.pt for kp in keypoints],
    #                     "descriptors": desc.tolist()
    #                 }
    #                 for keypoints, desc in zip(batch_keypoints, batch_descriptors)
    #             ]
    #         }
    #         with open(features_path, "w") as f:
    #             json.dump(features_data, f)
    #         print(f"Save feature points as {features_path}")
    #     print(f"Total {len(self.keypoints_list)} images processed.")
    def detect_features(self):
        """detect feature points and descriptors"""
        sift = cv2.SIFT_create()
        for img in self.images:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            self.keypoints_list.append(keypoints)
            self.descriptors_list.append(descriptors)

        # Save feature points to JSON
        features_path = os.path.join(self.output_dir, "features.json")
        features_data = {
            "features": [
                {
                    "keypoints": [kp.pt for kp in keypoints],
                    "descriptors": desc.tolist()
                }
                for keypoints, desc in zip(self.keypoints_list, self.descriptors_list)
            ]
        }
        with open(features_path, "w") as f:
            json.dump(features_data, f)
        print(f"Save feature points as {features_path}")

    def match_features(self):
        """feature matching"""
        matcher = cv2.FlannBasedMatcher_create()
        for i in range(len(self.descriptors_list) - 1):
            matches_i = matcher.knnMatch(self.descriptors_list[i], self.descriptors_list[i + 1], k=2)
            good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]
            self.matches.append(good_matches)

        # Save matching results to JSON
        matches_path = os.path.join(self.output_dir, "matches.json")
        matches_data = [
            [
                {"queryIdx": m.queryIdx, "trainIdx": m.trainIdx}
                for m in match
            ]
            for match in self.matches
        ]
        with open(matches_path, "w") as f:
            json.dump(matches_data, f)
        print(f"Save matching results to {matches_path}")

    def estimate_pose(self):
        """Estimate camera pose and sparse point cloud"""
        sparse_points = []
        for i in range(len(self.matches)):
            src_pts = np.float32([self.keypoints_list[i][m.queryIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.keypoints_list[i + 1][m.trainIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.K)
            self.poses.append((R.tolist(), t.tolist()))

            # triangulation to sparse point cloud
            proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj2 = np.hstack((R, t))
            proj1 = self.K @ proj1
            proj2 = self.K @ proj2

            points_4d = cv2.triangulatePoints(proj1, proj2, src_pts[mask.ravel() == 1], dst_pts[mask.ravel() == 1])
            points_3d = points_4d[:3] / points_4d[3]
            sparse_points.append(points_3d.T)

        sparse_points = np.vstack(sparse_points)
        # save sparse point cloud
        self.save_sparse_points(sparse_points)

        # save camera pose results
        poses_path = os.path.join(self.output_dir, "poses.json")
        with open(poses_path, "w") as f:
            json.dump(self.poses, f)
        print(f"camera pose save to {poses_path}")

    def save_sparse_points(self, points):
        """save sparse point cloud"""
        output_path = os.path.join(self.output_dir, "sparse_points.ply")
        with open(output_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"save sparse point cloud to {output_path}。")

    def dense_reconstruction(self):
        """Generate dense point cloud using multi-view stereo (MVS)"""
        sparse_path = os.path.join(self.output_dir, "sparse_points.ply")
        if not os.path.exists(sparse_path):
            raise FileNotFoundError("No sparse point cloud data found")

        # Read the sparse point cloud
        pcd = o3d.io.read_point_cloud(sparse_path)

        # Estimate normal vectors
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        # Save the dense point cloud result
        dense_output_path = os.path.join(self.output_dir, "dense_points.ply")
        o3d.io.write_point_cloud(dense_output_path, pcd)
        print(f"Save the dense point cloud result to {dense_output_path}。")
        return dense_output_path
    # def dense_reconstruction(self):
    #     """Generate dense point cloud using multi-view stereo(MVS)"""
    #     sparse_path = os.path.join(self.output_dir, "sparse_points.ply")
    #     if not os.path.exists(sparse_path):
    #         raise FileNotFoundError("No sparse point cloud data found")
    #
    #     # Read the sparse point cloud
    #     mesh = pv.read(sparse_path)
    #     dense_cloud = mesh.interpolate(n=100, sharpness=5.0)
    #
    #     # Save the dense point cloud result
    #     dense_output_path = os.path.join(self.output_dir, "dense_points.ply")
    #     dense_cloud.save(dense_output_path)
    #     print(f"Save the dense point cloud result to  {dense_output_path}。")
    #     return dense_output_path

    def surface_reconstruction(self):
        """Surface construction"""
        dense_cloud_path = os.path.join(self.output_dir, "dense_points.ply")
        if not os.path.exists(dense_cloud_path):
            raise FileNotFoundError("Dense point cloud data not found")

        # Read the dense point cloud data
        pcd = o3d.io.read_point_cloud(dense_cloud_path)

        # Estimate normal vectors
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        # Save the surface mesh
        mesh_output_path = os.path.join(self.output_dir, "reconstructed_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_output_path, mesh)
        print(f"Save the surface mesh to {mesh_output_path}")
        return mesh_output_path

    def ai_3d_segmentation(self, mesh_path):
        """3D segmentation"""
        pcd = o3d.io.read_triangle_mesh(mesh_path)
        pcd.compute_vertex_normals()

        # get 3d segmentation reuslts
        segmentation_labels = np.random.randint(0, 2, size=(len(np.asarray(pcd.vertices)),))
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = pcd.vertices
        segmented_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(len(segmentation_labels), 3))

        # save 3d segmentation reuslts
        segmentation_path = os.path.join(self.output_dir, "segmentation_result.ply")
        o3d.io.write_point_cloud(segmentation_path, segmented_pcd)
        print(f"save 3d segmentation reuslts to {segmentation_path}")
        return segmentation_path

    def select_minimal_image_subset(self):
        """Select the minimal camera images"""
        # Create feature points and calculate cover matrix
        total_keypoints = {tuple(kp.pt) for kp_list in self.keypoints_list for kp in kp_list}
        total_keypoints = list(total_keypoints)
        cover_matrix = np.zeros((len(total_keypoints), len(self.images)))

        for img_idx, keypoints in enumerate(self.keypoints_list):
            for kp in keypoints:
                if tuple(kp.pt) in total_keypoints:
                    kp_idx = total_keypoints.index(tuple(kp.pt))
                    cover_matrix[kp_idx, img_idx] = 1

        # Solve the minimum coverage problem using linear allocation methods
        _, selected_images = linear_sum_assignment(-cover_matrix)
        selected_images = list(set(selected_images))
        print(f"Choice {len(selected_images)} images: {selected_images}")

        return selected_images

    def compute_metrics(self, baseline_path, subset_path):
        """Calculate Chamfer dist, Coverage index, SSIM """
        baseline_pcd = o3d.io.read_point_cloud(baseline_path)
        subset_pcd = o3d.io.read_point_cloud(subset_path)

        # Geometric Accuracy：Chamfer dist
        baseline_points = np.asarray(baseline_pcd.points)
        subset_points = np.asarray(subset_pcd.points)
        distances = o3d.geometry.KDTreeFlann(baseline_pcd).compute_point_cloud_distance(subset_pcd)
        chamfer_distance = np.mean(distances)

        # Model Completeness：Coverage index
        coverage = len(distances[distances < 0.01]) / len(baseline_points)

        # Visual Fidelity：SSIM
        visual_fidelity = len(subset_points) / len(baseline_points)

        return {
            "Chamfer Distance": chamfer_distance,
            "Coverage": coverage,
            "Visual Fidelity": visual_fidelity
        }

    def run_pipeline(self):
        """Minimal Multi-Camera 3D Reconstruction and Segmentation pipeline"""
        # Load images and features detect and matching
        self.load_images()
        self.detect_features()
        self.match_features()
        self.estimate_pose()

        # 3D reconstruction
        print("\nstart the 3D reconstruction...")
        dense_cloud_all = self.dense_reconstruction()
        mesh_all = self.surface_reconstruction()

        # Minimal Multi-Camera images based on feature subset
        print("\nChoice minimal images subset...")
        selected_images = self.select_minimal_image_subset()
        subset_output_dir = os.path.join(self.output_dir, "subset")
        os.makedirs(subset_output_dir, exist_ok=True)

        # Save subset images
        self.images = [self.images[i] for i in selected_images]
        self.keypoints_list = [self.keypoints_list[i] for i in selected_images]
        self.descriptors_list = [self.descriptors_list[i] for i in selected_images]

        # Process the subset 3D reconstruction
        print("\nStart subset 3D reconstruction...")
        dense_cloud_subset = self.dense_reconstruction()
        mesh_subset = self.surface_reconstruction()

        # Save the subset 3D reconstruction result
        subset_dense_cloud_path = os.path.join(subset_output_dir, "dense_points_subset.ply")
        subset_mesh_path = os.path.join(subset_output_dir, "reconstructed_mesh_subset.obj")

        # Compare the index
        print("\nCalculate index...")
        metrics = self.compute_metrics(
            baseline_path=dense_cloud_all, subset_path=dense_cloud_subset
        )
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Show results
        print("\nResults:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nPipeline finished。")


class ThreeDGS:
    def __init__(self, image_dir, output_dir, intrinsic_matrix):
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.images = []
        self.keypoints_list = []
        self.descriptors_list = []
        self.matches = []
        self.poses = []
        self.K = intrinsic_matrix
        self.sparse_points = None
        self.gaussian_params = None

    def load_images(self):
        """Load images from the folder."""
        for img_file in sorted(os.listdir(self.image_dir)):
            if img_file.lower().endswith(".jpg"):
                img_path = os.path.join(self.image_dir, img_file)
                self.images.append(cv2.imread(img_path))
        print(f"Loaded {len(self.images)} images.")

    def detect_features(self):
        """Detect feature points and descriptors."""
        sift = cv2.SIFT_create()
        for img in self.images:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            self.keypoints_list.append(keypoints)
            self.descriptors_list.append(descriptors)

        # Save feature points to JSON
        features_path = os.path.join(self.output_dir, "features.json")
        features_data = {
            "features": [
                {
                    "keypoints": [kp.pt for kp in keypoints],
                    "descriptors": desc.tolist()
                }
                for keypoints, desc in zip(self.keypoints_list, self.descriptors_list)
            ]
        }
        with open(features_path, "w") as f:
            json.dump(features_data, f)
        print(f"Saved feature points to {features_path}.")

    def match_features(self):
        """Perform feature matching."""
        matcher = cv2.FlannBasedMatcher_create()
        for i in range(len(self.descriptors_list) - 1):
            matches_i = matcher.knnMatch(self.descriptors_list[i], self.descriptors_list[i + 1], k=2)
            good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]
            self.matches.append(good_matches)

        # Save matching results to JSON
        matches_path = os.path.join(self.output_dir, "matches.json")
        matches_data = [
            [
                {"queryIdx": m.queryIdx, "trainIdx": m.trainIdx}
                for m in match
            ]
            for match in self.matches
        ]
        with open(matches_path, "w") as f:
            json.dump(matches_data, f)
        print(f"Saved matching results to {matches_path}.")

    def estimate_pose_and_sparse_cloud(self):
        """Estimate camera pose and generate sparse point cloud."""
        sparse_points = []
        for i in range(len(self.matches)):
            src_pts = np.float32([self.keypoints_list[i][m.queryIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.keypoints_list[i + 1][m.trainIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.K)
            self.poses.append((R.tolist(), t.tolist()))

            # Triangulate sparse points
            proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj2 = np.hstack((R, t))
            proj1 = self.K @ proj1
            proj2 = self.K @ proj2

            points_4d = cv2.triangulatePoints(proj1, proj2, src_pts[mask.ravel() == 1], dst_pts[mask.ravel() == 1])
            points_3d = points_4d[:3] / points_4d[3]
            sparse_points.append(points_3d.T)

        self.sparse_points = np.vstack(sparse_points)
        self.save_sparse_points(self.sparse_points)

        # Save camera poses
        poses_path = os.path.join(self.output_dir, "poses.json")
        with open(poses_path, "w") as f:
            json.dump(self.poses, f)
        print(f"Saved camera poses to {poses_path}.")

    def save_sparse_points(self, points):
        """Save sparse point cloud."""
        output_path = os.path.join(self.output_dir, "sparse_points.ply")
        with open(output_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"Saved sparse point cloud to {output_path}.")

    def gaussian_splatting(self):
        """Generate Gaussian parameters for splatting."""
        self.gaussian_params = []
        for point in self.sparse_points:
            position = point[:3]
            scale = np.array([0.01, 0.01, 0.01])  # Example scale
            color = np.random.rand(3)  # Random color for simplicity
            self.gaussian_params.append({
                "position": position,
                "scale": scale,
                "color": color
            })

        gaussian_path = os.path.join(self.output_dir, "gaussian_params.json")
        with open(gaussian_path, "w") as f:
            json.dump(self.gaussian_params, f)
        print(f"Saved Gaussian parameters to {gaussian_path}.")

    def render(self):
        """Render the 3D object."""
        pcd = o3d.geometry.PointCloud()
        points = np.array([g["position"] for g in self.gaussian_params])
        colors = np.array([g["color"] for g in self.gaussian_params])
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        output_path = os.path.join(self.output_dir, "rendered_model.ply")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved rendered 3D model to {output_path}.")

    def run_pipeline(self):

        # Load images and features detect and matching
        self.load_images()
        self.detect_features()
        self.match_features()
        self.estimate_pose_and_sparse_cloud()
        self.gaussian_splatting()
        self.render()


# Main function
if __name__ == "__main__":
    image_dir = "./Datasets/IGBT"  # 2D camera image folder
    output_dir = "./001"  # output folder
    intrinsic_matrix = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])  # camera intrinsic matrix
    # method1: sfm
    sfm_mvs = SfmMvs(image_dir, output_dir, intrinsic_matrix)
    sfm_mvs.run_pipeline()
    # method2: 3D gaussian splatting
    three_dgs = ThreeDGS(image_dir, intrinsic_matrix)
    three_dgs.run_pipeline()
