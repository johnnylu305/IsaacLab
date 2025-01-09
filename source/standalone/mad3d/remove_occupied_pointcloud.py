import numpy as np
import pickle
import open3d as o3d
import argparse
import os
import glob

# Functions to be tested
def filter_point_cloud(point_cloud, occupancy_set):
    """
    Filters out points in the point cloud that are within the occupancy grid.

    Args:
        point_cloud (np.ndarray): Original point cloud (Nx3 array).
        occupancy_grid (np.ndarray): Occupancy grid points (Mx3 array).

    Returns:
        np.ndarray: Filtered point cloud.
    """
    # Convert occupancy grid to set of tuples for efficient lookup
    #occupancy_set = set(map(tuple, occupancy_grid))

    # Floor the point cloud to integer coordinates
    floored_points = np.floor(point_cloud).astype(int)

    # Check if floored points are in the occupancy grid
    filtered_indices = [i for i, point in enumerate(floored_points) if tuple(point) not in occupancy_set]

    # Return only the points not in the occupancy grid
    return point_cloud[filtered_indices]

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a point cloud based on an occupancy grid.")
    parser.add_argument("--pointcloud", required=True, help="Path to the folder containing the point cloud PLY files.")
    parser.add_argument("--pkl", required=True, help="Path to the folder containing the occupancy grid PKL files.")
    parser.add_argument("--output", required=True, help="Path to the folder to save the filtered point cloud PLY files.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Find all PLY files in the pointcloud directory and subdirectories
    ply_files = glob.glob(os.path.join(args.pointcloud, "**", "*.ply"), recursive=True)

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in directory: {args.pointcloud}")

    for ply_file in ply_files:
        # Derive corresponding PKL directory using glob to find matching subdirectory
        relative_path = os.path.relpath(ply_file, args.pointcloud)
        base_subdir = os.path.basename(os.path.dirname(relative_path))
        pkl_dirs = glob.glob(os.path.join(args.pkl, "**", base_subdir), recursive=True)

        if not pkl_dirs:
            print(f"Warning: No matching directory found for {base_subdir} in {args.pkl}. Skipping {ply_file}.")
            continue

        # Locate the fill_occ_set.pkl in the matched directory
        pkl_file = os.path.join(pkl_dirs[0], base_subdir, "fill_occ_set.pkl")

        if not os.path.isfile(pkl_file):
            print(f"Warning: No fill_occ_set.pkl found in {pkl_dirs[0]}. Skipping {ply_file}.")
            continue

        # Load point cloud from PLY file
        pcd = o3d.io.read_point_cloud(ply_file)
        point_cloud = np.asarray(pcd.points)

        # Load occupancy grid from PKL file
        with open(pkl_file, 'rb') as f:
            occupancy_set = pickle.load(f)
        #import pdb; pdb.set_trace()
        #occupancy_grid = np.array(occupancy_grid)
      
        # Run filter
        filtered_point_cloud = filter_point_cloud(point_cloud, occupancy_set)

        # Prepare output file path
        output_file_path = os.path.join(args.output, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Save filtered point cloud back to PLY file
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud)
        o3d.io.write_point_cloud(output_file_path, filtered_pcd)

        print(f"Filtered point cloud saved to {output_file_path}")
