import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("input", type=str, help="The root to the input USD file.")
parser.add_argument("output", type=str, help="The root to store the occupancy grid.")
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

camera_w=2000
camera_h=2000
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

def initialize_occupancy_grid(x_size, y_size, z_size):
    # Define the grid size and resolution (cell size)
    grid = np.zeros((x_size, y_size, z_size))
    return grid

def populate_occupancy_grid(points, grid, origin, resolution):
    for point in points:
        # Convert point coordinates to grid indices
        x_index = int((point[0] - origin[0]) / resolution)
        y_index = int((point[1] - origin[1]) / resolution)
        z_index = int((point[2] - origin[2]) / resolution)
        # Check grid bounds
        if 0 <= x_index < grid.shape[0] and 0 <= y_index < grid.shape[1] and 0 <= z_index < grid.shape[2]:
            grid[x_index, y_index, z_index] = 1  # Mark the cell as occupied

    return grid


def setup_scene(world, scene_path, stage, scene_prim_root="/World/Scene"):

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

def clear_translation_and_scale(prim):
    """
    Clears all translation and scale transformations on the given prim,
    but keeps the rotation transformations intact.
    """
    # Wrap the prim as an Xformable
    xform = UsdGeom.Xformable(prim)

    # Get all transformation operations on the prim
    xform_ops = xform.GetOrderedXformOps()

    # Iterate through each operation and clear translation and scale transformations
    for op in xform_ops:
        op_type = op.GetOpType()
        
        # Clear only translation and scale types
        if op_type in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale):
            op.GetAttr().Clear()  # Clear the specific transformation operation
        
    # Update xformOpOrder to remove cleared operations, keeping only rotations
    new_xform_op_order = [op for op in xform.GetOrderedXformOps() if op.GetOpType() not in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale)]
    xform.SetXformOpOrder(new_xform_op_order)

    print(f"Cleared translation and scale for prim: {prim.GetPath()}")

def clear_transforms_for_parents(prim):
    """
    Clears translation and scale transformations for the given prim and all its parent nodes.
    """
    # Clear transformations for the current prim
    clear_translation_and_scale(prim)

    # Traverse up the hierarchy to clear transformations for all parent nodes
    parent = prim
    
    while parent.GetParent():
        clear_translation_and_scale(parent)
        parent = parent.GetParent()

# Helper function to find the last child prim
def find_last_child_prim(prim):
    last_child = None
    for child in prim.GetAllChildren():
        last_child = child
    return last_child if last_child else prim

def rescale_scene(scene_prim_root="/World/Scene"):
    all_points = []
    meshes = []
    scene_prim_root="/World/Scene"
    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    
    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        
        points = mesh.GetPointsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        meshes.append((points, face_vertex_counts, face_vertex_indices))
        
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get()
        xformable = UsdGeom.Xformable(mesh_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
  
        # Transform each point to world coordinates
        transformed_points = [world_transform.Transform(point) for point in points]
        if transformed_points:
            all_points.extend(transformed_points)  # Aggregate points
    

    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    # for prim_path in mesh_prim_path:
    #     mesh_prim = get_prim_at_path(prim_path=prim_path)
    #     clear_transforms_for_parents(mesh_prim)

    # Get the root prim of the scene
    root_prim = get_prim_at_path("/World/Scene")

    # Initialize cumulative rotation as an identity quaternion
    cumulative_rotation = Gf.Quatf(1, 0, 0, 0)  # Identity quaternion
    original_up_vector = Gf.Quatd(0,0, 0, 1)
    # Traverse from root prim to the last child and accumulate rotations

    current_prim = get_prim_at_path(prim_path=mesh_prim_path[0])
    while current_prim != root_prim:
        # Check if the current prim has a transform
        if current_prim.IsA(UsdGeom.Xform):
            xform = UsdGeom.Xform(current_prim)
            xform_ops = xform.GetOrderedXformOps()

            # Extract and accumulate rotation quaternions
            for op in xform_ops:
                print(op.GetOpType())
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    rotation_quaternion = op.Get()  # Get quaternion as Gf.Quatf or Gf.Quatd
                    print(rotation_quaternion)
                    cumulative_rotation = rotation_quaternion * cumulative_rotation
        # Move to the parent prim
        current_prim = current_prim.GetParent()
        
    # Output the final cumulative rotation
    print("Cumulative Rotation Quaternion from Root to Last Child:", cumulative_rotation)
    # Convert the up vector into a quaternion with zero scalar part
    #v_quat = Gf.Quatd(0, original_up_vector)

    # Apply the quaternion rotation: rotated_vector = q * v * q.inverse()
    rotated_quaternion = cumulative_rotation * original_up_vector * cumulative_rotation.GetInverse()

    # Extract the vector part of the resulting quaternion
    up_axis = rotated_quaternion.GetImaginary()
    print("Transformed Up Axis:", up_axis)
    abs_up_axis = Gf.Vec3d(abs(up_axis[0]), abs(up_axis[1]), abs(up_axis[2]))
    if abs_up_axis[0] >= abs_up_axis[1] and abs_up_axis[0] >= abs_up_axis[2]:
        # X-axis is dominant
        if up_axis[0] > 0:
            mapped_axis = "x"
        else:
            mapped_axis = "-x"
    elif abs_up_axis[1] >= abs_up_axis[0] and abs_up_axis[1] >= abs_up_axis[2]:
        # Y-axis is dominant
        if up_axis[1] > 0:
            mapped_axis = "y"
        else:
            mapped_axis = "-y"
    else:
        # Z-axis is dominant
        if up_axis[2] > 0:
            mapped_axis = "z"
        else:
            mapped_axis = "-z"

    print("Mapped Up Axis:", mapped_axis)
    
        


    centers=[]
    y_mins=[]
    x_mins=[]
    z_mins=[]
    voxel_grids = []
    
    for mesh_data in meshes:
        o3d_mesh = convert_to_open3d_mesh(mesh_data)

        # Compute minimal y-coordinate
        y_min = o3d_mesh.get_min_bound()[1]
        x_min = o3d_mesh.get_min_bound()[0]
        z_min = o3d_mesh.get_min_bound()[2]
        center = o3d_mesh.get_center()
        o3d_mesh.translate(-center)  # Center the mesh by translating it to the origin based on the adjusted center
        print("Adjusted Center:", center)

        # scale_factor = args_cli.max_len / max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound())
        # o3d_mesh.scale(scale_factor, center=(0, 0, 0))
        

        # Compute the center and adjust the y-coordinate to the minimum y

        # Store the adjusted center and y_min for later calculations
        centers.append(center)
        y_mins.append(y_min)
        x_mins.append(x_min)
        z_mins.append(z_min)
    # mesh_prim = get_prim_at_path('/World/Scene')
    # #mesh_prim = get_prim_at_path(mesh_prim_path[0])
    # #max_coords, min_coords = get_minmax_mesh_coordinates(mesh_prim)
    
    # mesh = UsdGeom.Mesh(mesh_prim)
    # points_attr = mesh.GetPointsAttr()
    # points = points_attr.Get()
    # import pdb; pdb.set_trace()
    # # Get the world transformation matrix for the mesh
    # xformable = UsdGeom.Xformable(mesh_prim)
    # world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # #import pdb; pdb.set_trace()
    # # Transform each point to world coordinates
    # transformed_points = [world_transform.Transform(point) for point in points]

    # # Initialize min/max coordinates and centroid
    # max_coords = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))
    # min_coords = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
    

    # # Calculate min/max coordinates and accumulate for centroid
    # for point in transformed_points:
    #     max_coords[0] = max(max_coords[0], point[0])
    #     max_coords[1] = max(max_coords[1], point[1])
    #     max_coords[2] = max(max_coords[2], point[2])

    #     min_coords[0] = min(min_coords[0], point[0])
    #     min_coords[1] = min(min_coords[1], point[1])
    #     min_coords[2] = min(min_coords[2], point[2])

    #import pdb; pdb.set_trace()
    # Calculate the mean center
    mean_center = np.mean(centers, axis=0)
    global_y_min = min(y_mins)
    global_x_min = min(x_mins)
    global_z_min = min(z_mins)
    if mapped_axis == '-y':
        mean_center[1] = global_y_min
    if mapped_axis == '-x':
        mean_center[0] = global_x_min
    if mapped_axis == '-z':
        mean_center[2] = -global_z_min
    if mapped_axis == 'y':
        mean_center[1] = -global_y_min
    if mapped_axis == 'x':
        mean_center[0] = -global_x_min
    if mapped_axis == 'z':
        mean_center[2] = global_z_min
    centroid = mean_center

    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    # for prim_path in mesh_prim_path:
    #     mesh_prim = get_prim_at_path(prim_path=prim_path)
    #     clear_transforms_for_parents(mesh_prim)
    root_prim = get_prim_at_path("/World/Scene")

    # Traverse all descendants of the root prim
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Xform):  # Check if the prim has transform attributes
            xform = UsdGeom.Xform(prim)
            #Get all transformation operations on the prim
            xform_ops = xform.GetOrderedXformOps()

            # Iterate through each operation and clear translation and scale transformations
            for op in xform_ops:
                op_type = op.GetOpType()
                
                # Clear only translation and scale types
                if op_type in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale):
                    op.GetAttr().Clear()  # Clear the specific transformation operation

            # Update xformOpOrder to remove cleared operations, keeping only rotations
            new_xform_op_order = [op for op in xform.GetOrderedXformOps() if op.GetOpType() not in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale)]
            xform.SetXformOpOrder(new_xform_op_order)

            # Clear the transformation by setting identity transforms
            #xform.ClearXformOpOrder()  # Clears all transform operations
            #xform.AddTransformOp().Set(Gf.Matrix4d(1.0))  # Sets identity matrix as the transform


    print(mesh_prim_path)

    scale_factor = get_scale(mesh_prim_path, args_cli.max_len)
    print(scale_factor)
    print(f"Scaling factor: {scale_factor}")

    # # Apply the scaling to the mesh
    # for prim_path in mesh_prim_path:
    #     mesh_prim = get_prim_at_path(prim_path=prim_path)
    #     xform = UsdGeom.Xformable(mesh_prim)
    #     scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
    #     xform.ClearXformOpOrder()  # Clear any existing transformations
    #     xform.AddTransformOp().Set(scale_transform)

    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path=prim_path)
        
        #clear_transforms_for_parents(mesh_prim)
        xform = UsdGeom.Xformable(mesh_prim)
        #xform.ClearXformOpOrder()
        # Get all transformation operations on the prim
        #xform_ops = xform.GetOrderedXformOps()

        shift_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-centroid[0], -centroid[1], -centroid[2]))
        scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
        #xform.ClearXformOpOrder()  # Clear any existing transformations
        #xform.AddTransformOp().Set(scale_transform)
        
        combined_transform = shift_transform*scale_transform
        #combined_transform = scale_transform
        xform.AddTransformOp().Set(combined_transform)
    
    return mapped_axis


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
        #start_pts = torch.tensor([[5,0,5]]).cuda()
        #end_pts = torch.tensor([[15,15,5],[10,10,10]]).cuda()
        #import pdb; pdb.set_trace()
        #print(start_pts.shape)
        #print(end_pts.shape)
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
                #cube_pos = Gf.Vec3f((x*cell_size)+env_origin[0]+env_size/2, (y*cell_size)+env_origin[1]+env_size/2, base_height+h_off)
                cube_pos = Gf.Vec3f((x*cell_size), (y*cell_size), base_height+h_off)
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

def convert_to_open3d_mesh(mesh_data):
    vertices, face_vertex_counts, face_vertex_indices = mesh_data
    vertices = np.array(vertices)
    triangles = []

    # Convert face indices
    index = 0
    for fvc in face_vertex_counts:
        if fvc == 3:  # Assuming all faces are triangles
            triangles.append(face_vertex_indices[index:index+3])
        index += fvc

    triangles = np.array(triangles)

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()  # Optional: for better visualization
    return o3d_mesh

def voxelize_mesh(mesh, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    return voxel_grid
    
def run_simulator(sim, scene_entities, output, stage, mapped_axis):
    """Run the simulator."""
    # Define simulation stepping
    
   
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    camera: Camera = scene_entities["camera"]
    
    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    occ_grid = initialize_occupancy_grid(20,20,20)

    #occ_grid = populate_occupancy_grid(all_points, occ_grid, origin, 0.5)
    #import pdb; pdb.set_trace()  
    
    cell_size = min(10./10., 10./10.)/2  # meters per cell
    slice_height = 10.0 / 20.0  # height of each slice in meters
    
    #if self.cfg.vis_occ:
    #for j in range(occ_grid.shape[0]):
            
    #while simulation_app.is_running():
    
    
    voxel_size = 0.5
    
    meshes = []
    scene_prim_root="/World/Scene"
    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    merged_points = []
    merged_face_vertex_indices = []
    merged_face_vertex_counts = []
    vertex_offset = 0  # This will help to adjust indices as we merge meshes
    for prim_path in mesh_prim_path:
        # Get the mesh prim
        mesh_prim = get_prim_at_path(prim_path)
        mesh = UsdGeom.Mesh(mesh_prim)               
        
        # Get points (vertices), face vertex indices, and face vertex counts
        points = mesh.GetPointsAttr().Get()  # List of Gf.Vec3f or Gf.Vec3d
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()  # Indices list
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()  # List of face sizes (triangles/quads)

        # Add points to merged points list
        merged_points.extend(points)
        
        # Offset and add face vertex indices to merged list
        adjusted_indices = [i + vertex_offset for i in face_vertex_indices]
        merged_face_vertex_indices.extend(adjusted_indices)
        
        # Add face counts directly (no need for offset here)
        merged_face_vertex_counts.extend(face_vertex_counts)
        
        # Update vertex offset
        vertex_offset += len(points)

    # Create a single tuple with merged points, face counts, and face indices
    merged_mesh = (merged_points, merged_face_vertex_counts, merged_face_vertex_indices)

    o3d_mesh = convert_to_open3d_mesh(merged_mesh)
    # Rescale the mesh
    scale_factor = args_cli.max_len / max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound())
    o3d_mesh.scale(scale_factor, center=(0, 0, 0))
    
    voxel_grid = voxelize_mesh(o3d_mesh, voxel_size)
    voxel_centers = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    x_off = (20-(max(voxel_centers[:,0]) - min(voxel_centers[:,0])))//2
    z_off = (20-(max(voxel_centers[:,2]) - min(voxel_centers[:,2])))//2
    y_off = (20-(max(voxel_centers[:,1]) - min(voxel_centers[:,1])))//2
    #import pdb; pdb.set_trace()

    # Assume `points` is an Nx3 numpy array of points in the original coordinate system
    # Apply the transformation to map the coordinates


    for i in range(len(voxel_centers)):
        occ_grid[voxel_centers[i][0],voxel_centers[i][1],voxel_centers[i][2]]=1

    #o3d.visualization.draw([voxel_grid])

    for kk in range(2):
        # if len(all_points) > 0:
        #     pc_markers.visualize(translations=np.array(all_points))        
        sim.step()

    #import pdb; pdb.set_trace()
    #camera.update(dt=sim.get_physics_dt())
    #import pdb; pdb.set_trace()
                             
    #occ_grid = np.transpose(occ_grid,[0,2,1])
    #occ_grid = np.flip(occ_grid, axis=1)
    up_axis = mapped_axis
    if up_axis == 'x':
        # Rotate so X is up
        occ_grid = np.transpose(occ_grid, [2, 1, 0])  # Z becomes X, Y remains, X becomes Z
        occ_grid = np.flip(occ_grid, axis=1)           # Flip along the new Y-axis

    elif up_axis == '-x':
        # Rotate so -X is up
        occ_grid = np.transpose(occ_grid, [2, 1, 0])   # Z becomes X, Y remains, X becomes Z
        occ_grid = np.flip(occ_grid, axis=2)           # Flip along the new Z-axis
        
    elif up_axis == 'y':
        occ_grid = np.transpose(occ_grid, [0, 2, 1])   # X remains, Z becomes Y, Y becomes Z
        
    elif up_axis == '-y':
        # Rotate so -Y is up
        occ_grid = np.transpose(occ_grid, [0, 2, 1])   # X remains, Z becomes Y, Y becomes Z
        occ_grid = np.flip(occ_grid, axis=1)           # Flip along the new Y-axis
    elif up_axis == 'z':
        # Rotate so Z is up
        #occ_grid = np.transpose(occ_grid, [1, 0, 2])   # Swap X and Y axes
        occ_grid = np.flip(occ_grid, axis=0)           # Flip along the new X-axis
        
    elif up_axis == '-z':
        occ_grid = np.flip(occ_grid, axis=0)           # Flip along the new X-axis
        occ_grid = np.flip(occ_grid, axis=2)           # Flip along the new X-axis
        
        #occ_grid = np.flip(occ_grid, axis=0)           # Flip along the new X-axis
        # Rotate so -Z is up
        #occ_grid = np.transpose(occ_grid, [1, 0, 2])   # Y becomes X, X becomes Y, Z remains
        #occ_grid = np.flip(occ_grid, axis=0)           # Flip along the new X-axis
    for kk in range(2):
        # if len(all_points) > 0:
        #     pc_markers.visualize(translations=np.array(all_points))        
        sim.step()

    for i in range(occ_grid.shape[2]):
        # vis occ
        image_filename = f"occupancy_map_slice_{i}.png"
        save_occupancy_grid_as_image(occ_grid[:, :, i], os.path.join(output, image_filename))
        # Save as npy file
        np.save(os.path.join(output, "occ.npy"), occ_grid[:, :, :])
        create_blocks_from_occupancy(0, np.array([0.,0.,0.]), 
                                            occ_grid[:, :, i], cell_size, i*slice_height, i, 20, 1, 25)
    for kk in range(2):
        # if len(all_points) > 0:
        #     pc_markers.visualize(translations=np.array(all_points))        
        sim.step()
    vp_api = get_active_viewport()
    capture_viewport_to_file(vp_api, os.path.join(output, "vis.png"))
    
    #print(camera_pos)
    #if points_3d_world.squeeze().size()[0] > 0:
    #    pc_markers.visualize(translations=points_3d_world.squeeze())

            
def main():
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', '*[!_non_metric].usd'), recursive=True))
    #scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', '*[_non_metric].usd'), recursive=True))
    #scenes_path = scenes_path[:1]

    world = World(stage_units_in_meters=1.0, backend='torch', device='cpu')
    stage = world.scene.stage

    set_camera_view(eye=np.array([40, 40, 60]), target=np.array([-15, 15, 8]))

    for i, scene_path in enumerate(scenes_path):
        scene_entities = setup_scene(world, scene_path,stage)
        if args_cli.rescale:
            mapped_axis= rescale_scene()
        #import pdb; pdb.set_trace()
        relative_path = os.path.relpath(scene_path, args_cli.input)
        dest_path = os.path.join(args_cli.output, relative_path)
        output = os.path.split(dest_path)[0]
        
        world.reset()

        #generate_occupancy_maps(world, output)
        run_simulator(world, scene_entities, output, stage, mapped_axis)
        
        print(scene_path)
        #import pdb; pdb.set_trace()
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
