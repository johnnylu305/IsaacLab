import torch

def get_hit_face_with_grid(ray_origin, grid_resolution, grid_origin, endpoint):
    """
    Calculate the hit face of a ray on a cube and find the grid index of the endpoint.

    Args:
        ray_origin (torch.Tensor): Shape (N, 3), origin of the rays.
        grid_resolution (float or torch.Tensor): The resolution of the grid (dx, dy, dz).
        grid_origin (torch.Tensor): Shape (3,), the origin of the grid.
        endpoint (torch.Tensor): Shape (N, 3), the coordinates of the endpoints.

    Returns:
        tuple:
            - torch.Tensor: Shape (N,), index of the hit face for each ray.
            - torch.Tensor: Shape (N, 3), the grid indices for each endpoint.
    """
    # Ensure grid_resolution is a tensor with the same shape as grid_origin
    if not isinstance(grid_resolution, torch.Tensor):
        grid_resolution = torch.tensor(grid_resolution, dtype=torch.float32, device=ray_origin.device)

    # Step 1: Compute the grid index for the endpoint
    grid_index = torch.floor((endpoint - grid_origin) / grid_resolution).long()

    # Step 2: Calculate ray direction based on endpoint and ray origin
    ray_direction = endpoint - ray_origin

    # Step 3: Define cube face normals and compute intersections
    normals = torch.tensor([
        [1, 0, 0], [-1, 0, 0],  # +X, -X
        [0, 1, 0], [0, -1, 0],  # +Y, -Y
        [0, 0, 1], [0, 0, -1]   # +Z, -Z
    ], dtype=torch.float32, device=ray_origin.device)

    # Compute face centers for the endpoint cube
    cube_center = grid_origin + grid_index * grid_resolution + grid_resolution / 2
    face_centers = cube_center.unsqueeze(1) + (grid_resolution / 2) * normals  # (N, 6, 3)

    # Compute t for each face
    denom = torch.sum(ray_direction.unsqueeze(1) * normals, dim=-1)  # (N, 6)
    valid = denom.abs() > 1e-8  # Avoid division by zero

    t = torch.where(
        valid,
        torch.sum(normals * (face_centers - ray_origin.unsqueeze(1)), dim=-1) / denom,
        torch.full_like(denom, float('inf'))  # Set invalid t to infinity
    )  # (N, 6)

    # Compute intersection points
    intersection = ray_origin.unsqueeze(1) + t.unsqueeze(-1) * ray_direction.unsqueeze(1)  # (N, 6, 3)

    # Check if the intersection point is within cube bounds
    bounds = torch.abs(intersection - cube_center.unsqueeze(1)) <= (grid_resolution / 2)
    within_bounds = torch.all(bounds, dim=-1)  # (N, 6)

    # Find the first valid intersection
    t[~within_bounds] = float('inf')  # Discard intersections outside the bounds
    hit_face = torch.argmin(t, dim=-1)  # Get the index of the closest valid face

    return hit_face, grid_index

# Example usage
ray_origin = torch.tensor([[0.5, 0.5, 0.5], [2.5, 3.5, -1.5]], dtype=torch.float32)
endpoint = torch.tensor([[3.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
grid_resolution = 1.0
grid_origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

hit_face, grid_index = get_hit_face_with_grid(ray_origin, grid_resolution, grid_origin, endpoint)
print("Hit Face:", hit_face)
print("Grid Index:", grid_index)

