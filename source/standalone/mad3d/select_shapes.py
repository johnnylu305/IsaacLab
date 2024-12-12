import os
import glob
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def load_hollow_occ(path):
    """
    Load the hollow occupancy grid from a .npy file.
    """
    return np.load(path)

def occupancy_to_pointcloud(occ_grid):
    """
    Convert the 3D occupancy grid into a point cloud of occupied voxels.
    """
    return np.argwhere(occ_grid == 1)

def generate_sphere_points_from_grid(occ_grid, thickness=1.0):
    """
    Generate a spherical shell point cloud within the dimensions of the occupancy grid.
    """
    X, Y, Z = occ_grid.shape
    radius = (min(X, Y, Z) / 2.0) * 0.8
    cx, cy, cz = X / 2.0, Y / 2.0, Z / 2.0

    sphere_points = []
    r_min = radius - thickness
    r_max = radius

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if r_min <= dist <= r_max:
                    sphere_points.append([x, y, z])

    return np.array(sphere_points)

def chamfer_distance(pcA, pcB):
    """
    Compute the Chamfer distance between two point clouds pcA and pcB.
    """
    treeA = cKDTree(pcA)
    treeB = cKDTree(pcB)

    dist_A_to_B, _ = treeB.query(pcA)
    dist_B_to_A, _ = treeA.query(pcB)

    cd = np.mean(dist_A_to_B**2) + np.mean(dist_B_to_A**2)
    return cd

def extract_features(point_cloud):
    """
    Extract features from the point cloud using PCA.
    """
    pca = PCA(n_components=3)
    pca.fit(point_cloud)
    return pca.explained_variance_ratio_

def main():
    parser = argparse.ArgumentParser(description="Select a diverse subset of shapes and compute Chamfer distances.")
    parser.add_argument("--root_path", type=str, required=True, help="Root directory containing hollow_occ.npy files.")
    parser.add_argument("--num_shapes", type=int, default=200, help="Number of shapes to select.")
    args = parser.parse_args()

    # Find all hollow_occ.npy files
    hollow_occ_files = glob.glob(os.path.join(args.root_path, "**", "hollow_occ.npy"), recursive=True)
    if len(hollow_occ_files) == 0:
        print("No hollow_occ.npy files found in the specified directory.")
        return

    features = []
    file_paths = []

    # Extract features from each point cloud
    for file_path in hollow_occ_files:
        hollow_occ = load_hollow_occ(file_path)
        point_cloud = occupancy_to_pointcloud(hollow_occ)
        feature = extract_features(point_cloud)
        features.append(feature)
        file_paths.append(file_path)

    features = np.array(features)

    # Apply K-means clustering
    num_clusters = min(args.num_shapes, len(features))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    selected_files = []
    chamfer_distances = []

    # Generate a sphere point cloud
    sphere_occ = np.zeros_like(hollow_occ)
    sphere_pc = generate_sphere_points_from_grid(sphere_occ)

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(features[cluster_indices] - centroid, axis=1)
        representative_index = cluster_indices[np.argmin(distances)]
        selected_files.append(file_paths[representative_index])

        # Compute Chamfer distance for the representative shape
        hollow_occ = load_hollow_occ(file_paths[representative_index])
        hollow_pc = occupancy_to_pointcloud(hollow_occ)
        cd = chamfer_distance(hollow_pc, sphere_pc)
        chamfer_distances.append(cd)

    # Sort selected files and distances by Chamfer distance
    sorted_data = sorted(zip(selected_files, chamfer_distances), key=lambda x: x[1])

    # Save sorted file paths and Chamfer distances to the output file
    output_file = os.path.join(args.root_path, f"selected_shapes_{num_clusters}.txt")
    with open(output_file, 'w') as f:
        for file_path, cd in sorted_data:
            f.write(f"{file_path} {cd:.6f}\n")

    print(f"Selected and sorted {len(sorted_data)} shapes. Paths and Chamfer distances saved to {output_file}")

if __name__ == "__main__":
    main()

