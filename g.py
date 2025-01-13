import torch
import torch.nn as nn

# Create a 3D tensor (e.g., a 10x10x10 grid with a single voxel set to 200)
grid = torch.zeros(1, 1, 10, 10, 10)  # Shape: (B, C, D, H, W)
grid[:, :, 5, 6, 7] = 200  # Set voxel at (5, 6, 7) to 200

# Define a simple 3D convolutional layer
conv3d = nn.Conv3d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,  # 3x3x3 kernel
    stride=1,
    padding=1,  # To preserve input size
    bias=False
)

# Set kernel weights to a delta function (only center weight is 1.0, rest are 0.0)
with torch.no_grad():
    conv3d.weight[:] = 0.0
    conv3d.weight[0, 0, 1, 1, 1] = 1.0  # Center weight of the kernel

# Pass the grid through the convolutional layer
output = conv3d(grid)

# Find the value at the same voxel (5, 6, 7) in the output
voxel_value = output[0, 0, 5, 6, 7].item()

print("Input Grid:\n", grid)
print("\nKernel Weights:\n", conv3d.weight)
print("\nOutput Grid:\n", output)
print(f"\nValue at Voxel (5, 6, 7): {voxel_value}")

