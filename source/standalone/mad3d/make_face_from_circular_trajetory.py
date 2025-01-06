import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("input", type=str, help="The root to the input USD file.")
#parser.add_argument("output", type=str, help="The root to store the occupancy grid.")
parser.add_argument(
    "--rescale",
    action='store_true'
)
parser.add_argument(
    "--max_len",
    type=float,
    default=8,
    help="Normalize the longest side to max length.",
)
parser.add_argument(
    "--env_size",
    type=float,
    default=20,
    help="Env size",
)
parser.add_argument(
    "--floor_len",
    type=float,
    default=80,
    help="Floor length",
)
parser.add_argument(
    "--grid_size",
    type=float,
    default=20,
    help="Grid Size",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib
import os
import glob
import carb
import numpy as np
from PIL import Image
import torch
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.occupancy_map")
#from omni.isaac.occupancy_map.bindings import _occupancy_map
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from omni.isaac.lab.sim.spawners.sensors import spawn_camera, PinholeCameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth, convert_orientation_convention
from omni.isaac.lab.sensors import CameraCfg, Camera
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.math import transform_points, unproject_depth, quat_mul
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
import open3d as o3d

cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
cfg.markers["hit"].radius = 0.2
pc_markers = VisualizationMarkers(cfg)

camera_w=400
camera_h=400
grid_size=20
env_size=20
# sensor    
CameraCfg = CameraCfg(
    prim_path="/World/Camera",
    offset=CameraCfg.OffsetCfg(pos=(0,0,0), convention="world"),
    update_period=0, # update every physical step
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        #focal_length=1.38, # in cm
        #focus_distance=1.0, # in m 
        #horizontal_aperture=24., # in mm 
        #clipping_range=(0.1, 20.0) # near and far plane in meter
        clipping_range=(0.1, 40.0) # near and far plane in meter
    ),
    width=camera_w,
    height=camera_h,
    
)

def remove_occluded_face(grid_size, obv_grid, face_grid):

    faces = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32)


    face_grid_bool = face_grid.bool()
    # Mask out faces where the obv_grid is non-occupied (obv_grid == 0)
    non_occupied_mask = obv_grid == 1
    face_grid_bool[non_occupied_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 6)] = False

    # remove occluded face
    # Identify visible faces in the grid
    visible_indices = torch.nonzero(face_grid_bool, as_tuple=True)

    # Get indices and directions for occupied voxels
    env_idx, x, y, z, face_idx = visible_indices

    # Shift directions based on face index to find adjacent voxel positions
    shifts = faces[face_idx]  # Extract the shift for each visible face
    adj_x = x + shifts[:, 0].long()
    adj_y = y + shifts[:, 1].long()
    adj_z = z + shifts[:, 2].long()

    # Ensure adjacent positions are within bounds
    valid_mask = (
        (adj_x >= 0) & (adj_x < grid_size) &
        (adj_y >= 0) & (adj_y < grid_size) &
        (adj_z >= 0) & (adj_z < grid_size)
    )
    env_idx = env_idx[valid_mask]
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    adj_x = adj_x[valid_mask]
    adj_y = adj_y[valid_mask]
    adj_z = adj_z[valid_mask]
    face_idx = face_idx[valid_mask]

    # Check if the adjacent voxel is occupied, making the face occluded
    occluded_mask = obv_grid[env_idx, adj_x, adj_y, adj_z] == 2

    face_grid_bool = face_grid_bool.bool()

    # Update the face_grid to mark occluded faces as False
    face_grid_bool[env_idx, x, y, z, face_idx] = ~occluded_mask

    return face_grid_bool

def get_hit_face_with_grid(ray_origin, grid_resolution, env_origin, endpoint, offset, grid_shape=(20,20,20,6)):
    """
    Calculate the hit face of a ray on a cube and find the grid index of the endpoint.

    Args:
        ray_origin (torch.Tensor): Shape (N, 3), origin of the rays.
        grid_resolution (float or torch.Tensor): The resolution of the grid (dx, dy, dz).
        env_origin (torch.Tensor): Shape (3,), the origin of the grid.
        endpoint (torch.Tensor): Shape (N, 3), the coordinates of the endpoints.
        offset (torch.Tensor): Offset to adjust the grid computation.
        grid_shape (tuple): The shape of the output grid, default is (20, 20, 20, 6).

    Returns:
        tuple:
            - torch.Tensor: Shape (N, 3), grid indices for each endpoint.
            - torch.Tensor: Shape (grid_shape), indicating hit faces for each grid cell.
    """
    # Ensure grid_resolution is a tensor with the same shape as env_origin
    if not isinstance(grid_resolution, torch.Tensor):
        grid_resolution = torch.tensor(grid_resolution, dtype=torch.float32, device=ray_origin.device)

    # Add a small tolerance to avoid precision issues
    tolerance = 1e-6

    # Step 1: Compute the grid index for the endpoint
    grid_index = torch.floor((endpoint - env_origin + offset) / grid_resolution).long()

    # Step 2: Calculate ray direction based on endpoint and ray origin
    ray_direction = endpoint - ray_origin
    #ray_direction = ray_origin - endpoint

    # Step 3: Define cube face normals
    normals = torch.tensor([
        [1, 0, 0], [-1, 0, 0],  # +X, -X
        [0, 1, 0], [0, -1, 0],  # +Y, -Y
        [0, 0, 1], [0, 0, -1]   # +Z, -Z
    ], dtype=torch.float32, device=ray_origin.device)

    # Compute the center of the cube containing the endpoint
    cube_center = env_origin - offset + grid_index * grid_resolution + grid_resolution / 2

    # Compute face centers for the cube
    face_centers = cube_center.unsqueeze(1) + (grid_resolution / 2) * normals  # (N, 6, 3)

    # Compute t for each face
    denom = torch.sum(ray_direction.unsqueeze(1) * normals, dim=-1)  # (N, 6)
    valid = denom.abs() > 1e-6  # Avoid division by zero with increased threshold

    t = torch.where(
        valid,
        torch.sum(normals * (face_centers - ray_origin.unsqueeze(1)), dim=-1) / denom,
        torch.full_like(denom, float('inf'))  # Set invalid t to infinity
    )  # (N, 6)
    
    # Clamp t to be between 0 and 1
    t = torch.where((t >= 0) & (t <= 1), t, torch.full_like(t, float('inf')))

    # Compute intersection points
    intersection = ray_origin.unsqueeze(1) + t.unsqueeze(-1) * ray_direction.unsqueeze(1)  # (N, 6, 3)

    # Check if the intersection point is within cube bounds
    bounds = torch.abs(intersection - cube_center.unsqueeze(1)) <= (grid_resolution / 2 + tolerance)
    within_bounds = torch.all(bounds, dim=-1)  # (N, 6)

    # Initialize the output grid
    hit_face_grid = torch.zeros(grid_shape, dtype=torch.bool, device=ray_origin.device)

    # Assign hits to the grid
    for i, g_idx in enumerate(grid_index):
        if (g_idx >= 0).all() and (g_idx < torch.tensor(grid_shape[:3], device=g_idx.device)).all():
            hit_face_grid[g_idx[0], g_idx[1], g_idx[2]] |= within_bounds[i]

    return hit_face_grid

def vis_face_and_voxel(occ, face_vis):
    # Load the hollow_occ array
    #hollow_occ = np.load(scene_path)
    occ = np.where(occ == 2, 1, 0)

    X, Y, Z = occ.shape
    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

    voxel_list = []
    normal_list = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if occ[x, y, z] == 1:
                    voxel_list.append([x, y, z])
                    for n, (dx, dy, dz) in enumerate(faces):
                        if face_vis[x, y, z, n]:
                            normal_list.append([x + 0.5 * dx, y + 0.5 * dy, z + 0.5 * dz, dx, dy, dz])


    # Create Open3D voxel grid
    voxel_points = o3d.utility.Vector3dVector(voxel_list)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        o3d.geometry.PointCloud(points=voxel_points),
        voxel_size=1.0,
        min_bound=[0, 0, 0],
        max_bound=[X, Y, Z]
    )

    # Create Open3D lines for normals
    line_set = o3d.geometry.LineSet()
    line_points = []
    lines = []
    colors = []

    for i, (px, py, pz, nx, ny, nz) in enumerate(normal_list):
        line_points.append([px, py, pz])
        line_points.append([px + nx, py + ny, pz + nz])
        lines.append([2 * i, 2 * i + 1])
        colors.append([1, 0, 0])  # red color for normals

    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize using Open3D
    #o3d.visualization.draw_geometries([voxel_grid, line_set])
    o3d.visualization.draw([voxel_grid, line_set])

def save_face(scene_path, occ_set):
    hollow_occ = np.load(scene_path)
    # x,y,z -> z,x,y
    hollow_occ = np.where(hollow_occ == 2, 1, 0)

    X, Y, Z = hollow_occ.shape
    face_visibility = np.zeros((X, Y, Z, 6), dtype=bool)

    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

    def is_visible(x, y, z, dx, dy, dz):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nz < Z and 0 <= nx < X and 0 <= ny < Y:
            return (nx, ny, nz) not in occ_set
        return True

    for x, y, z in occ_set:
        if hollow_occ[x, y, z] == 1:
            for n, (dx, dy, dz) in enumerate(faces):
                face_visibility[x, y, z, n] = is_visible(x, y, z, dx, dy, dz)

    face_visibility = face_visibility#.transpose((1, 2, 0, 3))
    output = os.path.join(os.path.split(scene_path)[0], "faces.npy")
    np.save(output, face_visibility)
    return face_visibility


def setup_scene(world, scene_path, scene_prim_root="/World/Scene"):

    # ground plane
    world.scene.add_ground_plane(size=args_cli.floor_len/2., color=torch.tensor([52./255., 195./255., 235./255.]))

    # Add a dome light
    light_prim = UsdLux.DomeLight.Define(world.scene.stage, Sdf.Path("/Domeight"))
    light_prim.CreateIntensityAttr(500)

    # add scene
    scene = add_reference_to_stage(usd_path=scene_path, prim_path=scene_prim_root)
    #utils.setRigidBody(building, "convexDecomposition", False)
    scene_prim = XFormPrim(
        prim_path=scene_prim_root,
        translation=[0, 0, 0],
        #orientation=rot_utils.euler_angles_to_quats(np.array([90, 0, 0]), degrees=True)
    )
    #utils.setRigidBody(building, "convexDecomposition", False)
    world.scene.add(scene_prim)
    #camera = define_sensor()
    camera = Camera(CameraCfg)
    # Create a dictionary for the scene entities
    scene_entities = {}
    scene_entities["camera"] = camera
    
    return scene_entities

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
   
        #print(max_coords, min_coords)

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


def rescale_scene(scene_prim_root="/World/Scene"):

    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    print(mesh_prim_path)

    scale_factor = get_scale(mesh_prim_path, args_cli.max_len)
    print(scale_factor)
    print(f"Scaling factor: {scale_factor}")


    # Apply the scaling to the mesh
    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path=prim_path)
        xform = UsdGeom.Xformable(mesh_prim)
        scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
        xform.ClearXformOpOrder()  # Clear any existing transformations
        xform.AddTransformOp().Set(scale_transform)


def remove_prim(prim, level=0):
    """ Recursive function to traverse and print details of each prim """
    indent = "    " * level
    print(f"{indent}- {prim.GetName()} ({prim.GetTypeName()})")
    
    # Perform any specific checks or operations on the prim here
    # For example, you could check for specific properties or apply transformations

    # Traverse the children of the current prim
    for child in prim.GetChildren():
        remove_prim(child, level + 1)
        child.SetActive(False)  # Deactivates the current prim


def save_occupancy_grid_as_image(occupancy_grid, filename):
    # Define the color mapping for the occupancy grid
    colors = {
        1: (0, 0, 0),       # Black for occupied
        0: (255, 255, 255), # White for unoccupied
        2: (255, 255, 255)  # Assume it is free
    }

    # Create an RGB image from the occupancy grid
    image_data = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        image_data[occupancy_grid == value] = color

    # Create and save the image
    image = Image.fromarray(image_data, 'RGB')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(np.array(image).shape)
    image.save(filename)
    print(f"Saved occupancy map as image: {filename}")


def generate_occupancy_maps(world, output):
    # Acquire interfaces
    physx = omni.physx.acquire_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()

    # Create generator object
    generator = _occupancy_map.Generator(physx, stage_id)

    # Settings for the occupancy grid
    env_size_x, env_size_y, env_size_z = args_cli.env_size, args_cli.env_size, args_cli.env_size
    grid_size_x, grid_size_y, grid_size_z = args_cli.grid_size, args_cli.grid_size, args_cli.grid_size
    org_x, org_y = env_size_x/2., env_size_y/2.
    cell_size = min(env_size_x/grid_size_x, env_size_y/grid_size_y)  # meters per cell
    occupied_value = 0
    unoccupied_value = 1
    unseen_value = 2 # assume it is free space

    generator.update_settings(cell_size, occupied_value, unoccupied_value, unseen_value)

    # Dimensions of the environment
    slice_height = env_size_z / grid_size_z  # height of each slice in meters


    print(f"cell size: {cell_size}, cell height: {slice_height}")

    # this is z x y
    occ_maps = []

    # Iterate over each slice
    for i in range(grid_size_z):

        # Set transformation for this slice
        generator.set_transform((org_x, org_y, i*slice_height), (-env_size_x, -env_size_y, 0), (0, 0, slice_height))
        print(f"Org: ({org_x}, {org_y}, {i*slice_height}), Min: ({-env_size_x}, {-env_size_y}, 0), Max: (0, 0, {slice_height})")
        generator.generate2d()

        # Retrieve and process data
        buffer = generator.get_buffer()
        dims = generator.get_dimensions()
        # Convert buffer to numpy array for easier manipulation
        occupancy_grid = np.array(buffer).reshape(dims[1], dims[0])  # Adjust shape if needed
        # 1 for occupied, remaining for unoccupied 
        occupancy_grid = np.where(occupancy_grid<=0, 1, 0)

        # correct the direction for 3D grid
        occupancy_grid = np.transpose(occupancy_grid)[::-1, :]

        occ_maps.append(occupancy_grid)

    # only keep the occupied pixel whose neighbors are unoccupied
    for i in range(len(occ_maps)):
        if i == 0:
            occupancy_grid = occ_maps[i]
            occupancy_grid_next = occ_maps[i+1]
            nn_map = occupancy_grid[0:-2, 1:-1] & occupancy_grid[2:, 1:-1] & occupancy_grid[1:-1, 0:-2] & occupancy_grid[1:-1, 2:] & occupancy_grid_next[1:-1, 1:-1]
        elif i == len(occ_maps)-1:
            occupancy_grid = occ_maps[i]
            occupancy_grid_prev = occ_maps[i-1]
            nn_map = occupancy_grid[0:-2, 1:-1] & occupancy_grid[2:, 1:-1] & occupancy_grid[1:-1, 0:-2] & occupancy_grid[1:-1, 2:] & occupancy_grid_prev[1:-1, 1:-1] 
        else:
            occupancy_grid = occ_maps[i]
            occupancy_grid_prev = occ_maps[i-1]
            occupancy_grid_next = occ_maps[i+1]
            nn_map = occupancy_grid[0:-2, 1:-1] & occupancy_grid[2:, 1:-1] & occupancy_grid[1:-1, 0:-2] & occupancy_grid[1:-1, 2:] & occupancy_grid_prev[1:-1, 1:-1] & occupancy_grid_next[1:-1, 1:-1]
            occupancy_grid[1:-1, 1:-1] = occupancy_grid[1:-1, 1:-1] & (nn_map==0)   

        # Save or process the occupancy grid as an image
        image_filename = f"occupancy_map_slice_{i}.png"
        save_occupancy_grid_as_image(occupancy_grid, os.path.join(output, image_filename))

        # Save as npy file
        np.save(os.path.join(output, "occ.npy"), np.array(occ_maps))

        # Create blocks based on the occupancy grid
        create_blocks_from_occupancy(world, occupancy_grid, cell_size, i*slice_height, i, 22)
        print(f"Created blocks for occupancy map slice {i}")

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

        #print(points.min(), points.max())
        end_pts = (points).long()
        start_pts = start_pts.repeat(end_pts.shape[0],1)
        #get_hit_face_with_grid(camera_position, 1.0, 0.0, end_pts)
        
        bresenham_path = bresenhamline(end_pts, start_pts, max_iter=-1, device=self.device)
        #print(bresenham_path.min(), bresenham_path.max())
        #bresenham_path = bresenham_path.clamp(, self.grid_size[1]-1)
        mask = (bresenham_path[:,0]>=0) & (bresenham_path[:,1]>=0) & (bresenham_path[:,2]>=0) &\
            (bresenham_path[:,0]<self.grid_size[1]) & (bresenham_path[:,1]<self.grid_size[1]) & (bresenham_path[:,2]<self.grid_size[1])
        if bresenham_path[mask] is not None:
            self.update_log_odds(i, bresenham_path[mask], occupied=False)

def create_blocks_from_occupancy(env_id, env_origin, occupancy_grid, cell_size, base_height, z,env_size, target=0, h_off=60):
    stage = omni.usd.get_context().get_stage()
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x, y] == target:  
                # Calculate position based on cell coordinates
                cube_pos = Gf.Vec3f((x*cell_size)+env_origin[0]-env_size/2, (y*cell_size)+env_origin[1]-env_size/2, base_height+h_off)

                # Define the cube's USD path
                cube_prim_path = f"/World/OccupancyBlocks/Block_{env_id}_{x}_{y}_{z}_{target}"

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

def compute_orientation(current_position, target_position=np.array([0, 0, 0])):
    # Compute the direction vector from current position to target (origin)
    direction_vector = target_position - current_position
    # Normalize the direction vector
    direction_vector = normalize_vector(direction_vector)
    
    forward_vector = normalize_vector(np.array([1, 1, 0]))
    rotation_axis = np.array([0, 0, 1])
    rotation_angle = np.arctan2(direction_vector[1],direction_vector[0])
     
    return rotation_angle

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_normals_unorganized(point_cloud, camera_position, k=30):
    """
    Computes normals for an unorganized point cloud and orients them based on the camera position.

    Args:
        point_cloud (torch.Tensor): Shape (N, 3), the unorganized point cloud.
        camera_position (torch.Tensor): Shape (3,), the position of the camera.
        k (int): Number of nearest neighbors to use for normal estimation.

    Returns:
        torch.Tensor: Shape (N, 3), the oriented normal vectors for each point.
    """
    N, _ = point_cloud.shape
    normals = torch.zeros_like(point_cloud)  # Placeholder for normals

    # Compute pairwise distances
    dist = torch.cdist(point_cloud, point_cloud)  # Shape (N, N)

    # Find the k-nearest neighbors for each point
    knn_indices = dist.topk(k=k, largest=False).indices  # Shape (N, k)

    for i in range(N):
        # Get neighbors of the current point
        neighbors = point_cloud[knn_indices[i]]  # Shape (k, 3ids)

        # Compute the centroid of the neighbors
        centroid = neighbors.mean(dim=0)  # Shape (3,)

        # Compute the covariance matrix
        covariance = (neighbors - centroid).T @ (neighbors - centroid) / k

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        # The normal is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]  # Shape (3,)

        # Normalize the normal vector
        normal = normal / torch.norm(normal)

        # Orient the normal to face the camera
        to_camera = camera_position - point_cloud[i]  # Vector from point to camera
        if torch.dot(normal, to_camera) < 0:
            normal = -normal  # Flip the normal to face the camera

        # Assign the oriented normal
        normals[i] = normal

    return normals

def compute_visible_faces_grid_single_face(point_cloud, normals, voxel_face_normals, offset,camera_position, grid_resolution=1, grid_origin=0, grid_shape=(20,20,20)):

    """
    Computes a visibility grid for voxel faces ensuring only one visible face per voxel.

    Args:
        point_cloud (torch.Tensor): Shape (N, 3), the point cloud.
        normals (torch.Tensor): Shape (N, 3), the surface normals for each point.
        voxel_face_normals (torch.Tensor): Shape (6, 3), the six voxel face normals.
        grid_resolution (float): Size of each voxel.
        grid_origin (torch.Tensor): Shape (3,), origin of the voxel grid.
        grid_shape (tuple): Shape (X, Y, Z) of the voxel grid.

    Returns:
        torch.Tensor: Shape (X, Y, Z, 6), indicating the single visible face for each voxel.
    """
    # Initialize visibility grid and max dot product grid
    visibility_grid = torch.zeros((*grid_shape, 6), dtype=torch.bool)
    max_dot_product = torch.full((*grid_shape, 6), -float('inf'), dtype=torch.float32)

    # Map points to voxel grid indices
    voxel_indices = torch.floor((point_cloud +offset - grid_origin) / grid_resolution).long()  # Shape (N, 3)

    # Compute dot products between point normals and voxel face normals
    dot_products = torch.einsum('ij,kj->ik', normals, voxel_face_normals)  # Shape (N, 6)

    camera_position = torch.tensor(camera_position)
    
    for i in range(point_cloud.shape[0]):
        x, y, z = voxel_indices[i]
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            for face in range(6):
                #if torch.dot(point_cloud[i] - camera_position.squeeze(), voxel_face_normals[face]) > 0:
                #normals[i] = torch.tensor([0,0,1])
                if torch.dot(normals[i], voxel_face_normals[face]) > 0:
                #if True:
                #if dot_products[i, face] > 0: #and dot_products[i, face] > max_dot_product[x, y, z, face]:
                    #max_dot_product[x, y, z, face] = dot_products[i, face]
                    #visibility_grid[x, y, z] = False  # Clear all faces for this voxel
                    visibility_grid[x, y, z, face] = True  # Set the new visible face

    return visibility_grid


def compute_visible_faces_grid_from_obv(point_cloud, obv_grid, voxel_face_normals, offset, camera_position, grid_resolution=1, grid_origin=torch.tensor([0, 0, 0]), grid_shape=(20, 20, 20)):
    """
    Computes a visibility grid for voxel faces ensuring only one visible face per voxel.

    Args:
        point_cloud (torch.Tensor): Shape (N, 3), the point cloud.
        obv_grid (torch.Tensor): Shape (X, Y, Z), the occupancy grid, 1 is unoccupied, 2 is occupied.
        voxel_face_normals (torch.Tensor): Shape (6, 3), the six voxel face normals.
        offset (torch.Tensor): Shape (3,), offset for the voxel grid.
        camera_position (list or torch.Tensor): Shape (3,), the position of the camera.
        grid_resolution (float): Size of each voxel.
        grid_origin (torch.Tensor): Shape (3,), origin of the voxel grid.
        grid_shape (tuple): Shape (X, Y, Z) of the voxel grid.

    Returns:
        torch.Tensor: Shape (X, Y, Z, 6), indicating the single visible face for each voxel.
    """
    # Initialize visibility grid
    visibility_grid = torch.zeros((*grid_shape, 6), dtype=torch.bool)

    # Convert camera position to tensor
    camera_position = torch.tensor(camera_position, dtype=torch.float32)

    # Map points to voxel grid indices
    voxel_indices = torch.floor((point_cloud + offset - grid_origin) / grid_resolution).long()  # Shape (N, 3)

    # Iterate over each point in the point cloud
    for i in range(point_cloud.shape[0]):
        x, y, z = voxel_indices[i]

        # Check if the voxel index is within the grid bounds
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            point = point_cloud[i]
            voxel_center = grid_origin + torch.tensor([x + 0.5, y + 0.5, z + 0.5]) * grid_resolution

            for face in range(6):
                face_center = voxel_center + voxel_face_normals[face] * (grid_resolution / 2)

                # Check if the face is visible from the camera
                view_vector = camera_position[0] - face_center
                view_dot_normal = torch.dot(view_vector, voxel_face_normals[face])
                if view_dot_normal > 0:
                    visibility_grid[x, y, z, face] = True

    return visibility_grid

def run_simulator(sim, scene_entities):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    camera: Camera = scene_entities["camera"]

    # Create output directory to save images
    #output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    #os.makedirs(output_dir, exist_ok=True)

    env_size = 20
    radius = env_size/2.0 * 1.5 # Radius of the cylinder
    height = env_size/2.0 * 2. # Height of the cylinder
    #theta = [0, np.pi/2., np.pi, np.pi*3/2]
    theta = np.linspace(0,6*np.pi, 120)#200)
    num_points = len(theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(2, height, num_points)
    index = 0
    grid = OccupancyGrid(20, (1,20,20,20))
    # Simulate physics
    while simulation_app.is_running():
        
        # Update camera data
        target_position = np.array([x[index], y[index], z[index]])
        yaw = compute_orientation(target_position)
        target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
        target_position = torch.from_numpy(target_position).unsqueeze(0)
        target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
        #target_orientation = target_orientation.repeat(self.num_envs,1)
        #camera.set_world_poses(target_position, convert_orientation_convention(target_orientation.float(), origin="world", target="ros"))
        camera.set_world_poses_from_view(target_position.float(), torch.tensor([0,0,2]).unsqueeze(0).float())

        # Step simulation
        for i in range(10):
            sim.step()
            camera.update(dt=sim.get_physics_dt())

        index+=1
        if index>=num_points:
            break
        # Print camera info
        
        if "rgb" in camera.data.output.keys():
            print("Received shape of rgb image        : ", camera.data.output["rgb"].shape)
        if "distance_to_image_plane" in camera.data.output.keys():
            print("Received shape of depth image      : ", camera.data.output["distance_to_image_plane"].shape)
        camera_pos = camera.data.pos_w.clone()
        camera_quat = camera.data.quat_w_ros.clone()
        intrinsic_matrix = camera.data.intrinsic_matrices.clone()
        depth_image=camera.data.output["distance_to_image_plane"].clone()
        # prevent inf
        depth_image = torch.clamp(depth_image, 0, 20*2)
        points_3d_cam = unproject_depth(depth_image, intrinsic_matrix)
        points_3d_world = transform_points(points_3d_cam, camera_pos, camera_quat)
        
        i = 0
        mask_x = (points_3d_world[i,:, 0]).abs() < 20/2 - 1e-3
        mask_y = (points_3d_world[i,:, 1]).abs() < 20/2 - 1e-3
        mask_z = (points_3d_world[i,:, 2] < 20 - 1e-3) & (points_3d_world[i,:, 2] >=0) 

        # Combine masks to keep rows where both xyz are within the range
        mask = mask_x & mask_y & mask_z
        
        offset = torch.tensor([20/2, 20/2 ,0]).to()
        if points_3d_world[i][mask].shape[0] > 0:
            ratio = 20/20
            grid.update_log_odds(i,torch.floor((points_3d_world[i][mask]+offset)*ratio),
                                        occupied=True)
            #import pdb; pdb.set_trace()
            #face_vis = get_hit_face_with_grid(camera_pos, 1.0, 0.0, points_3d_world[i][mask], offset)
            #normal_map = compute_normals_unorganized(points_3d_world[i][mask], camera_pos.squeeze())
            voxel_face_normals = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32, device=points_3d_world.device)
            #face_vis = compute_visible_faces_grid_single_face(points_3d_world[i][mask], normal_map, voxel_face_normals, offset, camera_pos)
            #face_vis = compute_visible_faces_grid_from_obv((points_3d_world[i][mask], obv_occ, voxel_face_normals, offset, camera_pos)
        # Iterate over each slice
        org_x, org_y = 20./2., 20./2.
        org_z = 0
        cell_size = min(20./20., 20./20.)  # meters per cell
        slice_height = 20.0 / 20.0  # height of each slice in meters

        probability_grid = grid.log_odds_to_prob(grid.grid)
        # N, x_size, y_size, z_size
        # 0: free, 1: unknown, 2: occupied
        obv_occ = torch.ones(1, 20, 20, 20, 4)

        obv_occ[:, :, :, :, 0] = torch.where(probability_grid <= 0.3, 0, torch.where(probability_grid <= 0.6, 1, 2))
        
        face_vis = compute_visible_faces_grid_from_obv(points_3d_world[i][mask], obv_occ[0,:,:,:,0], voxel_face_normals, offset, camera_pos)
        face_vis = remove_occluded_face(grid_size, obv_occ[:,:,:,:,0], face_vis.unsqueeze(0))
        vis_face_and_voxel(obv_occ[0,:,:,:,0], face_vis.squeeze())
        #if self.cfg.vis_occ:
        for j in range(obv_occ.shape[0]):
            for i in range(20):
                # vis occ
                image_filename = f"occupancy_map_slice_{i}.png"
                #save_occupancy_grid_as_image(obv_occ[j, :, :, i, 0].cpu().numpy(), os.path.join(output, image_filename))
                # Save as npy file
                #np.save(os.path.join(output, "occ.npy"), obv_occ[0, :, :, :, 0].cpu().numpy())
                create_blocks_from_occupancy(j, np.array([0.,0.,0.]), 
                                                obv_occ[j, :, :, i, 0].cpu().numpy(), cell_size, i*slice_height, i, 20, 2, 25)
        #vp_api = get_active_viewport()
        #capture_viewport_to_file(vp_api, os.path.join(output, "vis.png"))
        #print(camera_pos)
        #if points_3d_world.squeeze().size()[0] > 0:
        #    pc_markers.visualize(translations=points_3d_world.squeeze())

            
def main():
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', '*[!_non_metric].usd'), recursive=True))
    #scenes_path = scenes_path[:1]

    world = World(stage_units_in_meters=1.0, backend='torch', device='cpu')
    stage = world.scene.stage

    set_camera_view(eye=np.array([40, 40, 60]), target=np.array([-15, 15, 8]))

    for i, scene_path in enumerate(scenes_path):
        scene_entities = setup_scene(world, scene_path)
        if args_cli.rescale:
            rescale_scene()
        relative_path = os.path.relpath(scene_path, args_cli.input)
        #dest_path = os.path.join(args_cli.output, relative_path)
        #output = os.path.split(dest_path)[0]
        
        world.reset()
        #generate_occupancy_maps(world, output)
        run_simulator(world, scene_entities)
        world.clear()
        #break
        #counter = 0
        # while simulation_app.is_running():
        #     world.step(render=True)
        #     vp_api = get_active_viewport()
        #     capture_viewport_to_file(vp_api, os.path.join(output, "vis.png"))
        #     if counter > 10:
        #         world.clear()
        #         break
        #     counter += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
