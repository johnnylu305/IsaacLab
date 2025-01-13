import numpy as np
import open3d as o3d

# Load occupancy grids
#our_occ_grid = np.load('/home/dsr/Documents/mad3d/New_Dataset/temp_bfs/0002e50309b44e409c96f440202d90b3/modified_occ_grid.npy')
#depth_occ_grid = np.load('/home/dsr/Documents/mad3d/New_Dataset/temp_occ_from_depth_cam/0002e50309b44e409c96f440202d90b3/occ.npy')

our_occ_grid = np.load('/home/dsr/Documents/mad3d/New_Dataset/temp_bfs/0157b39c4b434ceba885ddace4f5b9bd/modified_occ_grid.npy')
depth_occ_grid = np.load('/home/dsr/Documents/mad3d/New_Dataset/temp_occ_from_depth_cam/0157b39c4b434ceba885ddace4f5b9bd/occ.npy')

#our_occ_grid = np.load('/home/dsr/Documents/mad3d/New_Dataset/temp_bfs/02010e17a28845cc921858cc24ece163/modified_occ_grid.npy')
#depth_occ_grid = np.load('/home/dsr/Documents/mad3d/New_Dataset/temp_occ_from_depth_cam/02010e17a28845cc921858cc24ece163/occ.npy')

our = np.sum(our_occ_grid[:, :, 1:])
print(np.unique(depth_occ_grid))
depth = np.sum(depth_occ_grid[:, :, 1:]-1)
overlap = np.sum(np.logical_and(our_occ_grid[:, :, 1:], depth_occ_grid[:, :, 1:]-1))
print(our, depth, overlap, overlap/our)

# Ensure grids are of size 20x20x20
assert our_occ_grid.shape == (20, 20, 20), "Unexpected shape for our_occ_grid"
assert depth_occ_grid.shape == (20, 20, 20), "Unexpected shape for depth_occ_grid"

def grid_to_point_cloud_with_overlap(our_grid, depth_grid):
    """
    Create a point cloud where:
    - Red indicates unique points in `our_grid`.
    - Blue indicates unique points in `depth_grid`.
    - Black indicates overlapping points.
    """
    # Identify unique and overlapping points
    our_only = np.logical_and(our_grid > 0, depth_grid == 0)
    depth_only = np.logical_and(depth_grid > 0, our_grid == 0)
    overlap = np.logical_and(our_grid > 0, depth_grid > 0)

    # Create point clouds for each category
    our_points = np.argwhere(our_only)
    depth_points = np.argwhere(depth_only)
    overlap_points = np.argwhere(overlap)

    # Combine all points and assign colors
    all_points = np.vstack([our_points, depth_points, overlap_points])
    all_colors = np.vstack([
        np.tile([1, 0, 0], (our_points.shape[0], 1)),  # Red for `our`
        np.tile([0, 0, 1], (depth_points.shape[0], 1)),  # Blue for `depth`
        np.tile([0, 0, 0], (overlap_points.shape[0], 1))  # Black for overlap
    ])

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors)

    return point_cloud

# Generate the point cloud with overlap
point_cloud = grid_to_point_cloud_with_overlap(our_occ_grid, depth_occ_grid-1)

# Visualize
o3d.visualization.draw([point_cloud])

