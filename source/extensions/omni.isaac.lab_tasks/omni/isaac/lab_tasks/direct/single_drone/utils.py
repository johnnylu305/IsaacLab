import torch
import numpy as np
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Usd, Gf, Sdf


def _bresenhamline_nslope(slope, device):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = torch.amax(torch.abs(slope), dim=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = torch.ones(1, dtype=torch.long).to(device)
    normalizedslope = slope / scale
    normalizedslope[zeroslope] = torch.zeros(slope[0].shape).to(device)
    return normalizedslope

def _bresenhamlines(start, end, max_iter, device):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = torch.amax(torch.amax(torch.abs(end - start), dim=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start, device)

    # steps to iterate on
    stepseq = torch.arange(1, max_iter + 1).to(device)
    stepmat = stepseq.repeat(dim, 1) #np.tile(stepseq, (dim, 1)).T
    stepmat = stepmat.T

    # some hacks for broadcasting properly
    bline = start[:, None, :] + nslope[:, None, :] * stepmat

    # Approximate to nearest int
    bline_points = torch.round(bline).to(start.dtype)
    
    return bline_points

def bresenhamline(start, end, max_iter=5, device='cpu'):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter, device).reshape(-1, start.shape[-1])

# project distance to z axis
def dis_to_z(dist_images, intrinsic_matrix):
    # Extract parameters from the intrinsic matrix
    f_x = intrinsic_matrix[:, 0, 0]  # Focal length in x
    f_y = intrinsic_matrix[:, 1, 1]  # Focal length in y
    c_x = intrinsic_matrix[:, 0, 2]  # Principal point x
    c_y = intrinsic_matrix[:, 1, 2]  # Principal point y
    
    # Determine the number of images, height, and width
    n, img_height, img_width = dist_images.shape
    
    # Create arrays representing the x and y coordinates of each pixel
    y_indices, x_indices = torch.meshgrid(torch.arange(img_height), torch.arange(img_width), indexing='ij')
    y_indices, x_indices = y_indices.cuda(), x_indices.cuda()
    
    x_indices = x_indices[None, :, :] - c_x[:, None, None]
    y_indices = y_indices[None, :, :] - c_y[:, None, None]
    
    # Calculate the distance from each pixel to the principal point in the image plane
    d = torch.sqrt(x_indices**2 + y_indices**2 + f_x[:, None, None]**2)
    
    # Calculate Z-component using the cosine of the angle
    depth_images = dist_images * (f_x[:, None, None] / d)
    
    return depth_images

# TODO: implement unknown and free voxels
class OccupancyGrid:
    def __init__(self, env_size, grid_size, decrement=0.4, increment=0.84, max_log_odds=3.5, min_log_odds=-3.5, device='cpu'):
        """
        Initialize the occupancy grid on the specified device (CPU or GPU).
        """
        self.grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        self.grid_size = grid_size
        self.max_log_odds = max_log_odds
        self.min_log_odds = min_log_odds
        self.occupied_increment = increment
        self.free_decrement = decrement
        assert grid_size[1]==grid_size[2]
        assert grid_size[2]==grid_size[3]
        self.resolution = env_size/grid_size[1]
        self.device = device

    def prob_to_log_odds(self, probability):
        """
        Convert probability to log odds.
        """
        return torch.log(probability / (1 - probability))
    
    def log_odds_to_prob(self, l):
        """ Convert log-odds to probability. """
        return 1 / (1 + torch.exp(-l))

    def update_log_odds(self, i, indices, occupied=True):
        """
        Update log odds of the grid at specified indices.
        - indices: A 2D tensor of shape [N, 3] containing x, y, z indices to update.
        - occupied: Whether the points are occupied (True) or free (False).
        """
        indices = indices.long()
        if occupied:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] += self.occupied_increment
        else:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] -= self.free_decrement

        # Clamping the values
        self.grid.clamp_(min=self.min_log_odds, max=self.max_log_odds)
    
    def trace_path_and_update(self, i, camera_position, points):
        """
        Trace the path from the camera to each point using Bresenham's algorithm and update the grid.
        """
        camera_position = torch.tensor(camera_position).cuda()

        start_pts = (camera_position).unsqueeze(0).long()

        end_pts = (points).long()
        bresenham_path = bresenhamline((end_pts/self.resolution).long(), (start_pts/self.resolution).long(), 
                                        max_iter=-1, device=self.device)
        bresenham_path = bresenham_path.clamp(0, self.grid_size[1]-1)

        if bresenham_path is not None:
            self.update_log_odds(i, bresenham_path, occupied=False)

def get_all_mesh_prim_path(root):
    root_prim = get_prim_at_path(prim_path=root)
    stack = [root_prim]
    mesh_prim_path = []
    # While there are nodes in the stack
    while stack:
        # Pop the last node from the stack
        node = stack.pop()
        if node.GetTypeName() == "Mesh":
            mesh_prim_path.append(node.GetPath().pathString)
        # Get the children of the current node
        children = node.GetChildren()
    
        # Iterate over each child
        for child in children:
            # Add the child to the stack for further traversal
            stack.append(child)
    return mesh_prim_path

def get_minmax_mesh_coordinates(mesh_prim):
    # Access the mesh's point positions in local space
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    # Get the world transformation matrix for the mesh
    xformable = UsdGeom.Xformable(mesh_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Transform each point to world coordinates
    transformed_points = [world_transform.Transform(point) for point in points]

    # Calculate the maximum coordinates
    max_coords = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))
    min_coords = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
    for point in transformed_points:
        max_coords[0] = max(max_coords[0], point[0])
        max_coords[1] = max(max_coords[1], point[1])
        max_coords[2] = max(max_coords[2], point[2])

        min_coords[0] = min(min_coords[0], point[0])
        min_coords[1] = min(min_coords[1], point[1])
        min_coords[2] = min(min_coords[2], point[2])

    return max_coords, min_coords

def get_scale(mesh_prim_path, desired_len):
    
    max_x, max_y, max_z = -1e10, -1e10, -1e10
    min_x, min_y, min_z = 1e10, 1e10, 1e10

    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path=prim_path)
        max_coords, min_coords = get_minmax_mesh_coordinates(mesh_prim)

        max_x = max(max_x, max_coords[0])
        max_y = max(max_y, max_coords[1])
        max_z = max(max_z, max_coords[2])
        min_x = min(min_x, min_coords[0])
        min_y = min(min_y, min_coords[1])
        min_z = min(min_z, min_coords[2])
    extent = (max_x-min_x, max_y-min_y, max_z-min_z)
    max_side = max(extent)
    print(f"Max Side: {max_side} meters")
    return desired_len/max_side

def rescale_scene(scene_prim_root="/World/envs/env_0/Scene", max_len=15):

    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    scale_factor = get_scale(mesh_prim_path, max_len)
    print(f"Scaling factor: {scale_factor}")

    # Apply the scaling to the mesh
    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path=prim_path)
        xform = UsdGeom.Xformable(mesh_prim)
        scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
        xform.ClearXformOpOrder()  # Clear any existing transformations
        xform.AddTransformOp().Set(scale_transform)

def check_building_collision(occs, xyz, env_ids, org_x, org_y, org_z, cell_size, slice_height, env_origins):
    # remove offset
    xyz -= env_origins[env_ids]
    
    x, y, z = xyz.detach().cpu().numpy()

    x_, y_, z_ = x, y, z

    # smallest point to (0, 0, 0)
    x += org_x
    y += org_y
    z += org_z

    # to voxel id
    x = np.floor(x/cell_size).astype(np.int32)
    y = np.floor(y/cell_size).astype(np.int32)
    z = np.floor(z/slice_height).astype(np.int32)

    col = (z, x, y) in occs[env_ids]
    #print(f"Env: {env_ids} Map zxy: {z_} {x_} {y_} to voxel_zxy: {z} {x} {y}, Col: {col}")
    return col

def get_robot_scale(robot_prim_root, desired_max_size):
    crazyflie_prim = get_prim_at_path(prim_path=robot_prim_root)
    # Calculate the bounding box (extents) of the robot
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(crazyflie_prim)
    min_extent = bbox.GetRange().GetMin()
    max_extent = bbox.GetRange().GetMax()

    # Calculate the current dimensions
    current_size_x = max_extent[0] - min_extent[0]
    current_size_y = max_extent[1] - min_extent[1]
    current_size_z = max_extent[2] - min_extent[2]

    # Find the maximum dimension
    current_max_size = max(current_size_x, current_size_y, current_size_z)
    print(min_extent, max_extent)
    print(f"Robot Current max size: {current_max_size} meters")

    # Calculate the scaling factor
    scale_factor = desired_max_size / current_max_size
    print(f"Robot Scaling factor: {scale_factor}")
    return scale_factor

def rescale_robot(robot_prim_root, scale_factor):
    crazyflie_prim = get_prim_at_path(prim_path=robot_prim_root)
    # Apply the scaling to the geometry itself, keeping the transformation matrix intact
    for child in crazyflie_prim.GetChildren():
        geom_xform = UsdGeom.Xformable(child)
        geom_xform.ClearXformOpOrder()  # Clear any existing transformations
        scale_op = geom_xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(scale_factor, scale_factor, scale_factor))
        print(f"Applied scaling factor to child: {child.GetPath()}")

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_orientation(current_position, target_position=np.array([0, 0, 0])):
    # Compute the direction vector from current position to target (origin)
    direction_vector = target_position - current_position
    # Normalize the direction vector
    direction_vector = normalize_vector(direction_vector)
    
    forward_vector = normalize_vector(np.array([1, 1, 0]))
    rotation_axis = np.array([0, 0, 1])
    rotation_angle = np.arctan2(direction_vector[1],direction_vector[0])
     
    return rotation_angle

def create_blocks_from_occupancy(env_id, env_origin, occupancy_grid, cell_size, base_height, z):
    stage = omni.usd.get_context().get_stage()
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x, y] == 2:  # Occupied
                # Calculate position based on cell coordinates
                cube_pos = Gf.Vec3f((x*cell_size)+env_origin[0], (y*cell_size)+env_origin[1], base_height+60)

                # Define the cube's USD path
                cube_prim_path = f"/World/OccupancyBlocks/Block_{env_id}_{x}_{y}_{z}"

                # Create a cube primitive or get the existing one
                cube_prim = UsdGeom.Cube.Define(stage, Sdf.Path(cube_prim_path))
                cube_prim.GetPrim().GetAttribute("size").Set(cell_size)

                # Manage the transformation
                xform = UsdGeom.Xformable(cube_prim.GetPrim())
                xform_ops = xform.GetOrderedXformOps()
                if not xform_ops:
                    # If no transform ops exist, add a new translate op
                    xform_op = xform.AddTranslateOp()
                    xform_op.Set(cube_pos)
                else:
                    # If transform ops exist, modify the existing translate op
                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            op.Set(cube_pos)
                            break
                    else:
                        # If no translate op found, add it
                        xform.AddTranslateOp().Set(cube_pos)

def create_blocks_from_occ_set(env_id, env_origin, occ_set, cell_size, slice_height, env_size):
    stage = omni.usd.get_context().get_stage()
    for (z, x, y) in occ_set:
        # Calculate position based on cell coordinates
        cube_pos = Gf.Vec3f((x*cell_size)-env_size/2.+env_origin[0], (y*cell_size)-env_size/2.+env_origin[1], slice_height*z)

        #cube_pos = Gf.Vec3f((x*cell_size), (y*cell_size), slice_height*z)

        # Define the cube's USD path
        cube_prim_path = f"/World/OccupancyBlocks/BlockGt_{env_id}_{x}_{y}_{z}"

        # Create a cube primitive or get the existing one
        cube_prim = UsdGeom.Cube.Define(stage, Sdf.Path(cube_prim_path))
        cube_prim.GetPrim().GetAttribute("size").Set(cell_size)

        # Manage the transformation
        xform = UsdGeom.Xformable(cube_prim.GetPrim())
        xform_ops = xform.GetOrderedXformOps()
        if not xform_ops:
            # If no transform ops exist, add a new translate op
            xform_op = xform.AddTranslateOp()
            xform_op.Set(cube_pos)
        else:
            # If transform ops exist, modify the existing translate op
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(cube_pos)
                    break
            else:
                # If no translate op found, add it
                xform.AddTranslateOp().Set(cube_pos)
