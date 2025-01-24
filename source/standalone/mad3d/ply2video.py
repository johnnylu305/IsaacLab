import os
import open3d as o3d
import time
import numpy as np

def list_folders_and_files(root_dir):
    """
    List all folders in the root directory and find all .ply files in them.
    
    Parameters:
        root_dir (str): Root directory path.

    Returns:
        dict: Dictionary where keys are folder paths and values are lists of .ply file paths in those folders.
    """
    folder_ply_map = {}

    # Walk through the directory
    for subdir, _, files in os.walk(root_dir):
        ply_files = [os.path.join(subdir, file) for file in files if file.endswith(".ply") and not file.endswith("final.ply")]
        if ply_files:
            folder_ply_map[subdir] = ply_files

    return folder_ply_map

def register_and_merge(pcd_source, pcd_target):
    """
    Perform global registration to align the source point cloud to the target.

    Parameters:
        pcd_source (o3d.geometry.PointCloud): The source point cloud to align.
        pcd_target (o3d.geometry.PointCloud): The target point cloud to align to.

    Returns:
        o3d.geometry.PointCloud: Transformed source point cloud.
    """
    threshold = 0.02  # Distance threshold for registration
    transformation = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation

    # Apply the transformation to the source point cloud
    pcd_source.transform(transformation)
    return pcd_source

def merge_and_save(ply_files, output_folder, folder_name, voxel_size=0.02):
    """
    Merge .ply files incrementally and save each step as an image.

    Parameters:
        ply_files (list): List of .ply file paths.
        output_folder (str): Folder to save step images and final merged model.
        folder_name (str): Name of the folder being processed.
        voxel_size (float): Voxel size for downsampling point clouds.

    Returns:
        o3d.geometry.PointCloud: Final merged point cloud.
    """
    folder_output = os.path.join(output_folder, folder_name)
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    combined_cloud = o3d.geometry.PointCloud()

    for idx, file in enumerate(ply_files):
        # Load the next PLY file
        pcd = o3d.io.read_point_cloud(file)
        #pcd = register_and_merge(pcd, combined_cloud)
        # Combine with the existing point cloud
        combined_cloud += pcd

        # Optional: Downsample to remove duplicates
        #combined_cloud = combined_cloud.voxel_down_sample(voxel_size=voxel_size)

        # Save the combined state as an image
        step_image_path = os.path.join(folder_output, f"step_{idx + 1}.png")
        save_visualization(combined_cloud, step_image_path)

    # Save the final merged point cloud
    final_model_path = os.path.join(folder_output, "merged_model.ply")
    #o3d.io.write_point_cloud(final_model_path, combined_cloud)

    print(f"Final merged model for folder '{folder_name}' saved at: {final_model_path}")
    return combined_cloud

def save_visualization(point_cloud, filename):
    """
    Save a visualization of the point cloud as an image.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Point cloud to visualize.
        filename (str): Path to save the image.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    vis.add_geometry(point_cloud)
    vis.get_view_control().set_front([-1.0, 0.2, 0.2])
    vis.get_view_control().set_up([0.0, 0.0, 1.0])
    vis.get_view_control().set_zoom(0.8)
    vis.get_view_control().rotate(30.0, 0.0)  # Set a 60-degree rotation
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()

if __name__ == "__main__":
    # Specify the root directory containing folders with .ply files
    root_dir = "/projects/MAD3D/Zhuoli/IsaacLab/logs/sb3/MAD3D-v0/objaverse_data"

    # Specify the output folder for images and merged models
    output_folder = "output"

    # List all .ply files grouped by folder
    folder_ply_map = list_folders_and_files(root_dir)
    # Check if any folders with .ply files were found
    if not folder_ply_map:
        print("No .ply files found in the specified directory.")
    else:
        for folder, ply_files in folder_ply_map.items():
            if "ad4fe73f232840419e10a2b9e52cc729" not in folder:
                continue

            folder_name = os.path.basename(folder)
            print(f"Processing folder: {folder_name} with {len(ply_files)} .ply files...")

            # Merge and save for each folder
            merge_and_save(ply_files, output_folder, folder_name)

        print(f"Process complete. Results saved in: {output_folder}")

