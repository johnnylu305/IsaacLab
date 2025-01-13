import torch
import torch.nn.functional as F

# Create a 3D tensor (e.g., a 3x3x3 grid with unique values for each voxel)
# Shape: (1, 1, X, Y, Z)
grid = torch.zeros(1000, dtype=torch.float32).view(1, 1, 10, 10, 10)
grid[:, :, 5, 6, 7] = 200
#grid = torch.arange(27, dtype=torch.float32).view(1, 1, 3, 3, 3)

# Reorder grid to (B, C, Z, Y, X) for grid_sample compatibility
grid_reordered = grid.permute(0, 1, 4, 3, 2)  

query_coords = torch.tensor([[[5., 6., 7.]]], dtype=torch.float32)  # Shape: (1, 1, 3)

# Normalize query points to [-1, 1] based on the original grid dimensions
query_coords[..., 0] = 2.0 * (query_coords[..., 0] / grid.shape[2]) - 1.0  # Normalize X
query_coords[..., 1] = 2.0 * (query_coords[..., 1] / grid.shape[3]) - 1.0  # Normalize Y
query_coords[..., 2] = 2.0 * (query_coords[..., 2] / grid.shape[4]) - 1.0  # Normalize Z
print("\nNormalized Query Coordinates (XYZ order):\n", query_coords)

# Reshape and use grid_sample
query_coords = query_coords.view(1, 1, 1, 1, 3)  # Shape: (B, D=1, H=1, W=1, C=3)

# Perform trilinear sampling
interpolated_features = F.grid_sample(grid_reordered, query_coords, mode="bilinear", align_corners=True)
print("\nInterpolated Features:\n", interpolated_features)

