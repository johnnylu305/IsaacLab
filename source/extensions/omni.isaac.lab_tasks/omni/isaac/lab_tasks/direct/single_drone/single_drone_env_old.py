# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.lab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd, spawn_from_multiple_usd
from omni.isaac.lab.sim.spawners.sensors import spawn_camera, PinholeCameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils.math import transform_points, unproject_depth, quat_mul
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file, CameraCfg, Camera, ContactSensorCfg,ContactSensor
#from multi_object import spawn_multi_object_randomly
import random
import glob
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
from pxr import Sdf, UsdLux, Gf, UsdGeom, Usd, PhysxSchema, PhysicsSchemaTools, Vt, UsdPhysics
from PIL import Image
import omni.isaac.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
import math
import omni
from omni.physx import get_physx_scene_query_interface
from omni.isaac.core.articulations import ArticulationView


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
cfg.markers["hit"].radius = 0.002
pc_markers = VisualizationMarkers(cfg)
        
class OverlapShapeDemo:
    def __init__(self, stage, collision_prim_path):
        self._stage = stage
        self._collision_prim_path = collision_prim_path

    def report_hit(self, hit):
        #print("hit", hit)
        print("Collision detected with object:", hit.rigid_body)
        hitColor = Vt.Vec3fArray([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red color for collision
        usdGeom = UsdGeom.Cube.Get(self._stage, hit.rigid_body)
        if usdGeom:
            usdGeom.GetDisplayColorAttr().Set(hitColor)
        return True

    def check_collision(self):
        path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self._collision_prim_path))
        numHits = get_physx_scene_query_interface().overlap_shape(path_tuple[0], path_tuple[1], self.report_hit, False)
        if numHits>0:
            print("numHits", numHits)
            print("numHits", numHits)
            print("numHits", numHits)
            print("numHits", numHits)
        #else:            
            #import pdb; pdb.set_trace()
        print("numHits", numHits)

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

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
    #print(np.array(image).shape)
    image.save(filename)
    #print(f"Saved occupancy map as image: {filename}")

def _bresenhamline_nslope(slope):
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

def _bresenhamlines(start, end, max_iter):
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
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = torch.arange(1, max_iter + 1).to(device)
    stepmat = stepseq.repeat(dim, 1) #np.tile(stepseq, (dim, 1)).T
    stepmat = stepmat.T

    # some hacks for broadcasting properly
    bline = start[:, None, :] + nslope[:, None, :] * stepmat

    # Approximate to nearest int
    bline_points = torch.round(bline).to(start.dtype)
    
    return bline_points

def bresenhamline(start, end, max_iter=5):
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
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])

class OccupancyGrid:
    def __init__(self, grid_size, device='cpu'):
        """
        Initialize the occupancy grid on the specified device (CPU or GPU).
        """
        self.grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        self.max_log_odds = 3.5
        self.min_log_odds = -2.0
        self.occupied_increment = 0.84
        self.free_decrement = 0.4
        self.resolution = 30/64
        self.x_max = 64
        self.y_max = 64
        self.z_max = 64
        self.device = device

    def prob_to_log_odds(self, probability):
        """
        Convert probability to log odds.
        """
        return torch.log(probability / (1 - probability))
    
    def log_odds_to_prob(self, l):
        """ Convert log-odds to probability. """
        return 1 / (1 + torch.exp(-l))

    def update_log_odds(self, indices, occupied=True):
        """
        Update log odds of the grid at specified indices.
        - indices: A 2D tensor of shape [N, 3] containing x, y, z indices to update.
        - occupied: Whether the points are occupied (True) or free (False).
        """
        indices = indices.long()
        if occupied:
            self.grid[indices[:, 0], indices[:, 1], indices[:, 2]] += self.occupied_increment
        else:
            self.grid[indices[:, 0], indices[:, 1], indices[:, 2]] -= self.free_decrement

        # Clamping the values
        self.grid.clamp_(min=self.min_log_odds, max=self.max_log_odds)
    
    def trace_path_and_update(self, camera_position, points):
        """
        Trace the path from the camera to each point using Bresenham's algorithm and update the grid.
        """
        camera_position = torch.tensor(camera_position).cuda()

        #for pt in points:
        start_pts = (camera_position).unsqueeze(0).long()
        #for i in range(points.shape[0]):
            #start_pts = camera_position[i].long()
        #    start_pts = camera_position.long()

        end_pts = (points).long()
        # Generate Bresenham path from the camera to the point
        #bresenham_path = self.bresenhamline(start_pt, end_pt, max_iter=-1)
        #import pdb; pdb.set_trace()
        bresenham_path = bresenhamline((end_pts/self.resolution).long(), (start_pts/self.resolution).long(), max_iter=-1)
        #bresenham_path += torch.tensor([32, 32, 0], dtype=torch.long).cuda()
        bresenham_path = bresenham_path.clamp(0, 63)

        if bresenham_path is not None:
            # Update the grid for free space
            self.update_log_odds(bresenham_path[:-1], occupied=False)
            # Update the grid for occupied space at the end point
            #self.update_log_odds(bresenham_path[-1:], occupied=True)
            

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        disable_contact_processing=True,
        physx=sim_utils.PhysxCfg(use_gpu=False),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")


#      sensors
#     camera = CameraCfg(
#         prim_path="/World/envs/env_.*/Robot/body/camera",
#         update_period=0.1,
#         height=480,
#         width=640,
#         data_types=["rgb", "distance_to_image_plane"],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
#         ),
#         offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
#     )
#     camera 
    '''
    tiled_camera:CameraCfg =CameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), convention="world"),
        #offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            #focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=80,
        height=80,
      
    )
    '''
    
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), convention="world"),
        #offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.0, 0.05), convention="world"),
        #offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        update_period=0.01,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=200,
        height=200,
    )
    
    tiled_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera2",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), convention="world"),
        #offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.0, 0.05), convention="world"),
        #offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        update_period=0.01,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )
    
    #from omni.isaac.lab.sim.schemas import activate_contact_sensors
    
    #activate_contact_sensors("/World/envs/env_.*/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=6, debug_vis=True
    )
    
    # scene
    #scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=5, replicate_physics=True)

    
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # env
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    num_observations = 12
    num_states = 0
    debug_vis = True

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    
    




class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "coverage_ratio",
                "collision",
            ]
            #for key in [
            #    "lin_vel",
            #    "ang_vel",
            #    "distance_to_goal",
            #]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self._index=0
        self.pose = torch.zeros((self.num_envs,7)).cuda()
        self.current_position = torch.tensor([[2.5, 2.5, 2.5]], device=self.device).repeat(self.num_envs, 1)
        self.current_orientation = torch.tensor([[1, 0, 0, 0]], device=self.device).repeat(self.num_envs, 1)
        self.robot_view = ArticulationView("/World/envs/env_.*/Robot")
        

    def get_all_mesh_prim_path(self, root):
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

    def get_minmax_mesh_coordinates(self, mesh_prim):
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

    def get_scale(self, mesh_prim_path, desired_len):
        
        max_x, max_y, max_z = -1e10, -1e10, -1e10
        min_x, min_y, min_z = 1e10, 1e10, 1e10

        for prim_path in mesh_prim_path:
            mesh_prim = get_prim_at_path(prim_path=prim_path)
            max_coords, min_coords = self.get_minmax_mesh_coordinates(mesh_prim)
    
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

    def rescale_scene(self, scene_prim_root="/World/envs/env_0/Scene"):

        mesh_prim_path = self.get_all_mesh_prim_path(scene_prim_root)
        print(mesh_prim_path)
        #max_len = 15
        max_len = 1.5
        scale_factor = self.get_scale(mesh_prim_path, max_len)
        print(scale_factor)
        print(f"Scaling factor: {scale_factor}")

        # Apply the scaling to the mesh
        for prim_path in mesh_prim_path:
            mesh_prim = get_prim_at_path(prim_path=prim_path)
            xform = UsdGeom.Xformable(mesh_prim)
            scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
            xform.ClearXformOpOrder()  # Clear any existing transformations
            xform.AddTransformOp().Set(scale_transform)

    def _setup_scene(self):
        
        #scene_path = r'./Set_C_converted/BAT1_SETC_HOUSE1/BAT1_SETC_HOUSE1.usd'
        scenes_path = sorted(glob.glob(os.path.join(r'/mnt/zhuzhuan/Documents/MAD3D/IsaacLab/Set_C_converted (copy)', '**', '*[!_non_metric].usd'), recursive=True))
        scene_path= random.choice(scenes_path)
        #print(scene_path)
        #scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', '*[!_non_metric].usd'), recursive=True))
        #import re
        #regex_pattern = re.compile(r'/mnt/zhuzhuan/Documents/MAD3D/IsaacLab/Set_C_converted/BAT1_SETC_HOUSE1/*[^_non_metric]\.usd$')

        #spawn_from_usd(prim_path="/World/envs/env_.*/Scene", cfg=UsdFileCfg(usd_path=scene_path))       
        print(scenes_path)
        cfg_list = []
        for scene_path in scenes_path:
            cfg_list.append(UsdFileCfg(usd_path=scene_path))
        spawn_from_multiple_usd(prim_path="/World/envs/env_.*/Scene", my_asset_list=cfg_list)
        #import pdb; pdb.set_trace()
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        
        #Define the Crazyflie path and collision prim path
        
        collision_prim_path = "/World/envs/env_0/Robot/body/body_collision/geometry"
        #collision node need to remove from body node to parent
        crazyflie_path = "/World/envs/env_0/Robot"
        
        stage = omni.usd.get_context().get_stage()
        # Ensure the robot and collision geometry exist
        if not stage.GetPrimAtPath(crazyflie_path).IsValid():
            print(f"Robot at path {crazyflie_path} does not exist.")
        else:
            # Access the Crazyflie robot and collision geometry
            crazyflie_prim = stage.GetPrimAtPath(crazyflie_path)
            collision_prim = stage.GetPrimAtPath(collision_prim_path)
        
            # Ensure the geometry is a UsdGeom.Xformable
            if not crazyflie_prim.IsA(UsdGeom.Xformable) or not collision_prim.IsA(UsdGeom.Xformable):
                print("The geometry type is not supported. Please check the robot type.")
            else:
                # Retrieve the scale factor for /World/Crazyflie/body
                #body_prim_path = "/World/envs/env_0/Scene/Robot/body"
                #body_prim_path = "/World/Crazyflie"
                #body_prim = stage.GetPrimAtPath(body_prim_path)
                # disable gravity to avoid falling
                # Access the RigidBodyAPI
                # apply rigid body API and schema
                # physicsAPI = UsdPhysics.RigidBodyAPI.Apply(body_prim)
        
                #/World/Crazyflie/body.physxRigidBody:disableGravity
                
                #disable_gravity_attr = body_prim.GetAttribute("physxRigidBody:disableGravity")     
                #body_prim.physxRigidBody:disableGravity=True
                #import pdb; pdb.set_trace()   
                #disable_gravity_attr.Set(True)
        
                # print(attr)
                #print(f"Gravity disabled for {body_prim.GetPath()}")
                #body_xform = UsdGeom.Xformable(body_prim)
                body_scale_factor = Gf.Vec3f(1.0, 1.0, 5.0)
                #for op in body_xform.GetOrderedXformOps():
                #    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                #        body_scale_factor = op.Get()
                #        break
        
                # Retrieve the scale factor for /World/Crazyflie/body/body_collision/geometry
                collision_xform = UsdGeom.Xformable(collision_prim)
        
                # Define the desired size in world space
                #desired_size = 5.0  # meters
                #desired_size = 3.0  # meters
                desired_size = 0.5  # meters
        
                # Calculate the scale to be applied to the collision box to make it 3x3x3 meters in world space
                final_scale_factor = Gf.Vec3f(
                    desired_size / body_scale_factor[0],
                    desired_size / body_scale_factor[1],
                    desired_size / body_scale_factor[2]
                )
                print(f"Final scale factor to be applied: {final_scale_factor}")
        
                # Apply the final scale factor to the collision geometry
                collision_xform.ClearXformOpOrder()
                collision_xform.AddScaleOp().Set(final_scale_factor)
        
        #self.rescale_scene()
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        #self.contact_sensor = ContactSensor(self.cfg.contact_forces)
        self._tiled_camera2 = TiledCamera(self.cfg.tiled_camera2)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=5000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        grid_size = (64, 64, 64)
        self.grid = OccupancyGrid(grid_size, device=self.device)
        
        #spawn_camera(prim_path="/World/envs/env_.*/Robot/body/camera", cfg=PinholeCameraCfg(data_types=["rgb", "distance_to_image_plane"]))

    def _pre_physics_step(self, actions: torch.Tensor):
        #-self._terrain.env_origins
        #self._tiled_camera.set_world_poses(self.current_position+self._terrain.env_origins, self.current_orientation, convention="world")
        camera_targets = self._terrain.env_origins
        self._tiled_camera.set_world_poses_from_view(self.current_position, camera_targets)
        #self._tiled_camera.update(dt=self.sim.get_physics_dt())
        self._tiled_camera._update_buffers_impl(self.env_ids)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._xyz=self._actions[:, :3]*1
        self._yaw=self._actions[:,-1]
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def normalize_vector(self,v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def compute_orientation(self,current_position, target_position=np.array([0, 0, 0])):
        # Compute the direction vector from current position to target (origin)
        direction_vector = target_position - current_position
        # Normalize the direction vector
        direction_vector = self.normalize_vector(direction_vector)
        
        # Assuming the forward direction of the drone is along the z-axis and it starts facing up.
        # Calculate the quaternion that rotates the z-axis to align with the direction vector.
        # This is a basic vector alignment calculation and could be more complex based on your coordinate system.
        forward_vector = self.normalize_vector(np.array([1, 1, 0]))  # This may need adjustment based on your drone's model orientation
        #rotation_axis = np.cross(forward_vector, direction_vector)
        rotation_axis = np.array([0, 0, 1])
        #rotation_angle = np.arccos(direction_vector[1])
        #rotation_angle = np.arccos(np.dot(forward_vector[:-1], direction_vector[:-1]))
        rotation_angle = np.arctan2(direction_vector[1],direction_vector[0])
 
        # Normalize the rotation axis
        #rotation_axis = self.normalize_vector(rotation_axis)
        # Create a quaternion based on the axis-angle representation
        #quaternion = np.array([
        #    np.cos(rotation_angle / 2),
        #    np.sin(rotation_angle / 2) * rotation_axis[0],
        #    np.sin(rotation_angle / 2) * rotation_axis[1],
        #    np.sin(rotation_angle / 2) * rotation_axis[2]
        #])
        
        return rotation_angle

    def _apply_action(self):
        #init_root_pos_w, init_root_quat_w = self.robot_view.get_world_poses()
        #init_root_pos_w-=0.1
        #self.robot_view.set_world_poses(init_root_pos_w, init_root_quat_w)
        #self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        radius = 0.5        # Radius of the cylinder
        
        height = 0.2       # Height of the cylinder
        num_points = 100  # Number of points in the trajectory

        # Angular coordinates
        theta = np.linspace(0, 2 * np.pi, num_points)
        
        radius+=0.01*self._index
        # X and Y coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)


        #x = np.linspace(0, height, num_points)
        #y = np.linspace(0, height, num_points)

        # Z coordinates (extending along the height of the cylinder)
        z = np.linspace(0, height, num_points)
        
        self._index+=1

        if self._index >= num_points:
            self._index = 0  # Reset index to loop the trajectory
        current_position = np.array([x[self._index], y[self._index], z[self._index]])
        
        yaw = self.compute_orientation(current_position)
        #print('yaw')
        #print(yaw)
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, yaw+np.pi]), degrees=False)
        #orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=False)
        # Assuming x, y, z are PyTorch tensors with the position data and _index is an integer or tensor of indices
        current_position = torch.from_numpy(current_position)
        orientation = torch.from_numpy(orientation)
        
        #orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, self._yaw*180]), degrees=True)
        
        #import pdb; pdb.set_trace()
        #env_ids = self.env_ids
        env_ids = torch.arange(self.num_envs).cuda()

        #default_root_state = self._robot.data.default_root_state[env_ids]
        #print(self.default_root_state.shape)
        #import pdb; pdb.set_trace()
        root_state = torch.zeros((self.num_envs,13)).cuda()
        #self.default_root_state[:,:3]=current_position.unsqueeze(0)
        #self.default_root_state[:,3:7]=orientation.unsqueeze(0)
        root_state[:,:3]=current_position.unsqueeze(0)
        #print(self._xyz.shape)
        #print(self._terrain.env_origins[self.env_ids].shape)
        #root_state[:,:3]=self._xyz + self._terrain.env_origins
        #self.current_position=root_state[:,:3]
        
        root_state[:,3:7]=orientation.unsqueeze(0)
        #default_root_state = self.default_root_state
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.current_position=root_state[:,:3]
        self.current_orientation=root_state[:,3:7]
        #print(root_state[0, :7])
        #print(env_ids)
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_data_to_sim()
        #self._robot.write_joint_state_to_sim(root_state[:, :7], root_state[:, 7:], None, env_ids)
        #print(self._index)
        
        #self.robot_view.set_world_poses(root_state[:, :3], root_state[:, 3:7])
        #print(default_root_state)
        #import pdb; pdb.set_trace()
        #joint_pos = self._robot.data.default_joint_pos[env_ids]
        #joint_vel = self._robot.data.default_joint_vel[env_ids]
        #self._robot.write_joint_state_to_sim(root_state[:, 3:7], joint_vel, None, env_ids)

    def pointcloud_from_and_depth(self, depth:torch.Tensor, intrinsic_matrix:torch.Tensor, camera_matrix:torch.Tensor=None):
        print(depth.shape)
        print(intrinsic_matrix.shape)
        print(camera_matrix.shape)
        depth = depth.squeeze(-1)
    
        # Create idx array
        v_idxs, u_idxs = torch.meshgrid(torch.arange(depth.shape[0]), torch.arange(depth.shape[1]), indexing='ij')
        # Get only non-masked depth and idxs
        z = depth
        # Calculate local position of each point
        # Apply vectorized math to depth using compressed arrays
        cx = intrinsic_matrix[0,2]
        fx = intrinsic_matrix[0,0]
        z = torch.where(z == float('inf'), torch.tensor(1e10), z)
        x = (u_idxs - cx) * z / fx
        cy = intrinsic_matrix[1,2]
        fy = intrinsic_matrix[1,1]
        # Flip y as we want +y pointing up not down
        
        y = -((v_idxs - cy) * z / fy)
        
        # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
        if camera_matrix is not None:
            w = torch.ones_like(z)
            # Calculate camera pose from extrinsic matrix
            # Create homogenous array of vectors by adding 4th entry of 1
            # At the same time flip z as for eye space the camera is looking down the -z axis
            x_y_z_eye_hom = torch.stack((z.flatten(), -x.flatten(), y.flatten(), w.flatten()), dim=0)
            x_y_z_world = camera_matrix @ x_y_z_eye_hom
            x_y_z_world = x_y_z_world[:3]
            print("world", x_y_z_world)
            return x_y_z_world.T
        else:
            x_y_z_local = torch.vstack((x, y, z),dim=0)
            return x_y_z_local.T
    


    def get_meters_per_unit(self):
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        return UsdGeom.GetStageMetersPerUnit(stage)

    def create_blocks_from_occupancy(self, occupancy_grid, cell_size, base_height, z):
        """
        Create blocks in the simulation based on the occupancy grid.
        :param occupancy_grid: Numpy array of the occupancy grid
        :param cell_size: Size of each cell in meters
        :param base_height: The base height for this slice of the occupancy grid
        :param slice_height: The height of each slice
        """
        world = World(stage_units_in_meters=1.0, backend='torch', device='cpu')
        #print(occupancy_grid)
        for x in range(occupancy_grid.shape[0]):
            for y in range(occupancy_grid.shape[1]):
                if occupancy_grid[x, y] == 0:  # Occupied
                    # Calculate position based on cell coordinates
                    #print('creating occupancy grid')
                    #cube_pos = Gf.Vec3f((x * cell_size) - 15, (y * cell_size) - 15+50, base_height-15)
                    cube_pos = Gf.Vec3f((x * cell_size) - 15, (y * cell_size) - 15, base_height)
    
                    # Define the cube's USD path
                    cube_prim_path = f"/World/OccupancyBlocks/Block_{x}_{y}_{z}"
                    #print(cube_prim_path) 
                    # Create a cube primitive or get the existing one
                    cube_prim = UsdGeom.Cube.Define(world.scene.stage, Sdf.Path(cube_prim_path))
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

    def _get_observations(self) -> dict:

        #data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        #observations = {"policy": self._tiled_camera.data.output[data_type].clone()}
        # print information from the sensors
        
        
        print("-------------------------------")
        #print(self.contact_sensor)
        print("Received shape of rgb   image: ", self._tiled_camera.data.output["rgb"].shape)
        print("Received shape of depth image: ", self._tiled_camera.data.output["depth"].shape)

        print("Received shape of rgb   image of camera 2: ", self._tiled_camera2.data.output["rgb"].shape)
        print("Received shape of depth image of camera 2: ", self._tiled_camera2.data.output["depth"].shape)
        #print("Received shape of depth image: ", self._tiled_camera.data.output["distance_to_image_plane"].shape)
        #print(self._tiled_camera.data.intrinsic_matrices)
        print(self._tiled_camera.data.pos_w[0])
        
        print("-------------------------------")
        
        camera_targets = self._terrain.env_origins
        #self._tiled_camera.update(dt=self.sim.get_physics_dt())
        print(self._tiled_camera.data.pos_w)
        
        #self._tiled_camera.set_world_poses_from_view(self.current_position, camera_targets)
        #self._tiled_camera.set_world_poses(self.current_position, self.current_orientation, convention="world")
        #self._tiled_camera._update_buffers_impl(self.env_ids)
        print(self._index)
        
        #import matplotlib.pyplot as plt
        #plt.imshow(self._tiled_camera.data.output["depth"][0,:,:,0].cpu())
        #plt.imsave('camera_image/depth_{}.png'.format(self._index),self._tiled_camera.data.output["distance_to_image_plane"][0,:,:].clone().cpu(), cmap='gray')
        #import pdb; pdb.set_trace()
        #plt.imsave('camera_image/rgb_{}.png'.format(self._index),(self._tiled_camera.data.output["rgb"][0,:,:].cpu().numpy()).astype(np.uint8))
        depth_image = self._tiled_camera.data.output["depth"].clone().clamp(-32,32)
        
        #depth_image = self._tiled_camera.data.output["distance_to_image_plane"].clone()
        #import pdb; pdb.set_trace()
        rgb_image = self._tiled_camera.data.output["rgb"].clone()
        save_images_to_file(rgb_image,f"camera_image/depth_{self._index}.png")

        points_3d_cam = unproject_depth(depth_image, self._tiled_camera.data.intrinsic_matrices)
        #points_3d_world = transform_points(points_3d_cam, self._tiled_camera.data.pos_w, self._tiled_camera.data.quat_w_ros)
        
        points_3d_world = transform_points(points_3d_cam, self.current_position, self.current_orientation)
        
        #points_3d_world = points_3d_world.transpose(1,3)
        #import pdb; pdb.set_trace()
        pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=self._tiled_camera.data.intrinsic_matrices[0],
                depth=depth_image[0],
                position=self._tiled_camera.data.pos_w[0],
                orientation=self._tiled_camera.data.quat_w_ros[0],
                device=self.device,
            )
        

        if points_3d_world.size()[0] > 0:
            pc_markers.visualize(translations=points_3d_world[0])
                
        #print(points_3d_world)
        # # Add pointcloud to scene
        # points = (1.0/self.get_meters_per_unit()) * points

        # invalid_indices = np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1)

        # invalid_indices = invalid_indices | (points[:, 0] > 15) | (points[:, 0] < -15) | (points[:, 1] > 15) | (points[:, 1] < -15) | (points[:, 2] > 30) | (points[:, 2] < 0)
        # # Remove points with NaN or Inf values
        # points = points[~invalid_indices]
        # print("P", points)

        # points = (points + np.array([[15, 15, 0]]))*64./30.

        # cliped_depth = np.clip(depth_camera,0,100)
                    
        # plt.imsave('extension_examples/user_examples/camera_image/depth_{}.png'.format(self._index),cliped_depth, cmap='gray')
        
        offset=(self._terrain.env_origins).unsqueeze(1)
        
        
        #offset=torch.tensor([0,0,0]).cuda()
        points_3d_world-=offset
        points_3d_world = points_3d_world[~torch.isnan(points_3d_world).any(dim=2)]

        import time
        current_time = time.perf_counter()
        # resolution = 1
        # zero = 0
        # points = torch.from_numpy(points).cuda()
        #print(self._terrain.env_origins[self.env_ids])
        #print(points_3d_world[:,0].max())
        #print(points_3d_world[:,0].min())
        #print(points_3d_world[:,1].max())
        #print(points_3d_world[:,1].min())
        #print(points_3d_world[:,2].max())
        #print(points_3d_world[:,2].min())
      
        #import pdb; pdb.set_trace()
        #points_3d_world += offset
        points_3d_world = points_3d_world.clamp(-0.99,0.99)
        points_3d_world = (points_3d_world*32)
        offset=torch.tensor([32,32,0]).cuda()
        points_3d_world+=offset
        
        self.grid.update_log_odds(points_3d_world, occupied=True)
        
        #import pdb; pdb.set_trace()
        self.grid.trace_path_and_update(self._tiled_camera.data.pos_w[0]-self._terrain.env_origins[0], points_3d_world)

        # print(new_pcd.points)
        self.probability_grid = self.grid.log_odds_to_prob(self.grid.grid)
        
        self.occupancy_grid = torch.where(self.probability_grid<=0.5, 1, 0)
        print('occupancy grid time:')
        self.probability_grid = self.probability_grid.cpu().numpy()

        print(time.perf_counter()-current_time)

        self.probability_grid = np.where(self.probability_grid<=0.5, 1, 0)
        #import pdb; pdb.set_trace()
        
        print(np.unique(self.probability_grid), np.sum(self.probability_grid))
        # Iterate over each slice
        env_size_x, env_size_y, env_size_z = 30, 30, 30
        grid_size_x, grid_size_y, grid_size_z = 64, 64, 64
        #grid_size_x, grid_size_y, grid_size_z = 100, 100, 100
        org_x, org_y = env_size_x/2., env_size_y/2.
        cell_size = min(env_size_x/grid_size_x, env_size_y/grid_size_y)  # meters per cell
        slice_height = env_size_z / grid_size_z  # height of each slice in meters

        output='occ_test'
        for i in range(grid_size_z):                
            #self.create_blocks_from_occupancy(self.probability_grid[:,:,i], cell_size, i*slice_height, i)
            image_filename = f"occupancy_map_slice_{i}.png"
            #save_occupancy_grid_as_image(self.probability_grid[:,:,i], os.path.join(output, image_filename))
        #import pdb; pdb.set_trace()

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        stage = omni.usd.get_context().get_stage()
        collision_prim_path = "/World/envs/env_0/Robot/body/body_collision/geometry"
        overlap_demo = OverlapShapeDemo(stage, collision_prim_path)
        overlap_demo.check_collision()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        #rewards = {
        #    "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
        #    "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
        #    "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        #}
        
        rewards = {
            "coverage_ratio": (torch.sum(1-self.occupancy_grid[10:54,10:54,2:30])/(44*44*28)).repeat(self.num_envs), #to do
            #"collision": 0, #to do
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        #reward = rewards["coverage_ratio"]
        print('coverage_ratio', reward)
    
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self.env_ids = env_ids
        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.default_root_state = default_root_state
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first tome
    #     if debug_vis:
    #         if not hasattr(self, "goal_pos_visualizer"):
    #             marker_cfg = CUBOID_MARKER_CFG.copy()
    #             marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    #             # -- goal pose
    #             marker_cfg.prim_path = "/Visuals/Command/goal_position"
    #             self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.goal_pos_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_pos_visualizer"):
    #             self.goal_pos_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     self.goal_pos_visualizer.visualize(self._desired_pos_w)