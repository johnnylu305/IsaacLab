import open3d as o3d
import numpy as np

# Parameters for the floor mesh
floor_width = 20  # width and length of the floor
z_value = 0  # z-coordinate of the floor
voxel_size = 1.0  # voxel size

# Create a mesh grid in the XY plane
half_width = floor_width // 2
x = np.arange(-half_width, half_width + 1, voxel_size)
y = np.arange(-half_width, half_width + 1, voxel_size)
xx, yy = np.meshgrid(x, y)
zz = np.full_like(xx, z_value)

# Combine to create points
points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)

# Convert to Open3D Point Cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Generate voxel grid from the point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

# Calculate world coordinates of the voxels
voxel_coordinates = np.array([voxel_grid.origin + voxel.grid_index * voxel_size for voxel in voxel_grid.get_voxels()])

# Visualize the voxel grid
o3d.visualization.draw_geometries([voxel_grid])

# Print first 10 voxel coordinates
print("First 10 voxel coordinates:")
print(voxel_coordinates[:10])

