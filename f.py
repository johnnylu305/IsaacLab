import torch
import torch.nn.functional as F

# Create a 3D tensor (e.g., a 10x10x10 grid with a single voxel set to 200)
grid = torch.zeros(1000, dtype=torch.float32).view(1, 1, 10, 10, 10)
grid[:, :, 1, 0, 0] = 200  # Set voxel at (5, 6, 7) to 200

# Reorder grid to (B, C, Z, Y, X) for grid_sample compatibility
grid_reordered = grid.permute(0, 1, 4, 3, 2)  # Reordering (B, C, X, Y, Z) -> (B, C, Z, Y, X)

# Define query coordinates in XYZ order (e.g., X=5, Y=6, Z=7)
query_coords = torch.tensor([[[1., 0., 0.]]], dtype=torch.float32)  # Shape: (1, 1, 3)

# Reorder query coordinates to (Z, Y, X)
#query_coords = query_coords[..., [2, 1, 0]]  # Reorder from (X, Y, Z) to (Z, Y, X)

# Normalize query points to [-1, 1] based on grid dimensions
query_coords[..., 0] = 2.0 * (query_coords[..., 0] / grid.shape[4]) - 1.0  # Normalize Z
query_coords[..., 1] = 2.0 * (query_coords[..., 1] / grid.shape[3]) - 1.0  # Normalize Y
query_coords[..., 2] = 2.0 * (query_coords[..., 2] / grid.shape[2]) - 1.0  # Normalize X

print("\nNormalized Query Coordinates (Z, Y, X order):\n", query_coords)

# Reshape and use grid_sample
query_coords = query_coords.view(1, 1, 1, 1, 3)  # Shape: (B, D=1, H=1, W=1, C=3)
interpolated_features = F.grid_sample(grid_reordered, query_coords, mode="bilinear", align_corners=True)

print("\nInterpolated Features:\n", interpolated_features)

