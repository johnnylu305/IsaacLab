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
from omni.isaac.occupancy_map.bindings import _occupancy_map
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd


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


def create_blocks_from_occupancy(world, occupancy_grid, cell_size, base_height, z, offset=0, occ_prim_path="/World/OccupancyBlocks"):
    # Ensure the root node for blocks exists and can be transformed
    occupancy_blocks_path = occ_prim_path
    occupancy_blocks_node = UsdGeom.Xform.Define(world.scene.stage, Sdf.Path(occupancy_blocks_path))

    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x, y] == 1:  # Occupied cell
                # Calculate position relative to the node
                cube_pos = Gf.Vec3f(x * cell_size - args_cli.grid_size/2.0 + cell_size/2.0 - offset, 
                                    y * cell_size - args_cli.grid_size/2.0 + cell_size/2.0 + offset, base_height)

                # Define the cube's USD path under the node
                cube_prim_path = f"{occupancy_blocks_path}/Block_{x}_{y}_{z}"
                
                # Create a cube primitive
                cube_prim = UsdGeom.Cube.Define(world.scene.stage, Sdf.Path(cube_prim_path))
                cube_prim.GetPrim().GetAttribute("size").Set(cell_size)
                
                # Manage the transformation
                xform = UsdGeom.Xformable(cube_prim.GetPrim())
                xform.AddTranslateOp().Set(cube_pos)


def main():
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', '*[!_non_metric].usd'), recursive=True))
    #scenes_path = scenes_path[:1]

    world = World(stage_units_in_meters=1.0, backend='torch', device='cpu')
    stage = world.scene.stage

    set_camera_view(eye=np.array([40, 40, 40]), target=np.array([-15, 15, 8]))

    for i, scene_path in enumerate(scenes_path):
        setup_scene(world, scene_path)
        if args_cli.rescale:
            rescale_scene()
        relative_path = os.path.relpath(scene_path, args_cli.input)
        dest_path = os.path.join(args_cli.output, relative_path)
        output = os.path.split(dest_path)[0]
        
        world.reset()

        generate_occupancy_maps(world, output)

        counter = 0
        while simulation_app.is_running():
            world.step(render=True)
            vp_api = get_active_viewport()
            capture_viewport_to_file(vp_api, os.path.join(output, "vis.png"))
            if counter > 10:
                world.clear()
                break
            counter += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
