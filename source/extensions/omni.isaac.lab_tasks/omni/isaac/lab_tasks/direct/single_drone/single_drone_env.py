# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import pickle
import gymnasium as gym


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
#from omni.isaac.lab.envs.ui import BaseEnvWindow
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
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim

from omni.isaac.core.prims import XFormPrim
from omni.isaac.lab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd, spawn_from_multiple_usd, spawn_from_multiple_usd_env_id
from omni.isaac.lab.sim.spawners.sensors import spawn_camera, PinholeCameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth, convert_orientation_convention

from omni.isaac.lab.utils.math import transform_points, unproject_depth, quat_mul
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file, CameraCfg, Camera, ContactSensorCfg,ContactSensor
from .single_drone_env_cfg import QuadcopterEnvCfg
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
import omni.isaac.core.utils.prims as prim_utils

from .utils import bresenhamline, check_building_collision, rescale_scene, rescale_robot, get_robot_scale, compute_orientation, create_blocks_from_occupancy, create_blocks_from_occ_set, create_blocks_from_occ_list, OccupancyGrid, dis_to_z, extract_foreground, get_seen_face, compute_distance_to_center_distance, remove_occluded_face, check_free, shift_gt_occs, shift_gt_faces, shift_occs, check_height
import time
from omni.isaac.lab.markers import VisualizationMarkers
import open3d as o3d 
from .utils import merge_point_clouds
import re

cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
cfg.markers["hit"].radius = 0.002
pc_markers = VisualizationMarkers(cfg)

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.set_debug_vis(self.cfg.debug_vis)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "coverage_ratio",
                "collision",
                "goal",
                "penalty",
                "free",
                "hlimit", 
                "face_ratio",
                "all_penalty",
                "status coverage_ratio",
                "status obv_face",
                "status fg",
                "status fgc",
                "status ssim_icr",
                "status ssim_icr_v2",
                "status ssim_fg",
                "status ssim_fgc",
                "status gi_ssim"
            ]
        }

        self.episode_rec = dict()
        self.episode_rec["x"] = [[] for i in range(self.num_envs)]
        self.episode_rec["y"] = [[] for i in range(self.num_envs)]
        self.episode_rec["z"] = [[] for i in range(self.num_envs)]
        self.episode_rec["pitch"] = [[] for i in range(self.num_envs)]
        self.episode_rec["yaw"] = [[] for i in range(self.num_envs)]

    def _setup_scene(self):


        self.cfg.num_envs = self.num_envs
        
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # for pre-planned trajectory 
        self._index = -1
        radius = self.cfg.env_size/2.0 * 0.8 # Radius of the cylinder
        height = self.cfg.env_size/2.0 * 1.8 # Height of the cylinder
        #theta = [0, np.pi/2., np.pi, np.pi*3/2]
        #theta = [0, np.pi/4., np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4]
        theta = np.linspace(0,np.pi*6,20)
        self.num_points = len(theta)
        self.x = radius * np.cos(theta)
        self.y = radius * np.sin(theta)
        self.z = np.linspace(2, height, self.num_points)
        # line trajectory
        #self.x = np.zeros(self.num_points)
        #self.y = np.linspace(-self.cfg.env_size/2.0*0.6, self.cfg.env_size/2.0*0.6, self.num_points)
        #self.z = np.ones(self.num_points) * 4

        # occ set for collision detection
        self.occs = [set() for i in range(self.cfg.num_envs)]
        # ground truth occ grid for coverage ratio
        self.gt_occs = torch.zeros((self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size), device=self.device)
        # ground truth for face coverage ratio
        # [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.gt_faces = torch.zeros((self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 6), device=self.device)
        # collision status
        self.col = [False for i in range(self.cfg.num_envs)]
        self.not_free = [False for i in range(self.cfg.num_envs)]
        self.not_height = [False for i in range(self.cfg.num_envs)]
        self.stuck = torch.tensor([0 for i in range(self.cfg.num_envs)], device=self.device)

        # x, y, z
        self.robot_pos = torch.zeros((self.cfg.num_envs, 3), device=self.device)
        self.real_xyz = torch.zeros((self.cfg.num_envs, 3), device=self.device)
        # pitch, yaw
        self.robot_ori = torch.zeros((self.cfg.num_envs, 2), device=self.device)

        # obv
        # it should start from 0 instead of -1 because we need one obv for initial action
        self.env_step = torch.ones(self.cfg.num_envs,) * 0
        # N,T,H,W,3
        self.obv_imgs = torch.zeros((self.cfg.num_envs, self.cfg.img_t, self.cfg.camera_h, self.cfg.camera_w, 3), device=self.device)
        # N,T,xyz + pitch + yaw
        self.obv_pose_history = torch.zeros(self.cfg.num_envs, self.cfg.total_img, 5, device=self.device)
        # N, x_size, y_size, z_size, label+xyz
        self.obv_occ = torch.ones(self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 4, device=self.device)*0.5
        # N, x_size, y_size, z_size, 6 
        self.obv_face = torch.zeros(self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 6, device=self.device)
        # Generate the linearly spaced values for each dimension
        x_coords = torch.linspace(-self.cfg.env_size/2.0, self.cfg.env_size/2.0, self.cfg.grid_size, device=self.device)
        y_coords = torch.linspace(-self.cfg.env_size/2.0, self.cfg.env_size/2.0, self.cfg.grid_size, device=self.device)
        z_coords = torch.linspace(0.0, self.cfg.env_size, self.cfg.grid_size, device=self.device)
        # Create a meshgrid of the coordinates
        x_mesh, y_mesh, z_mesh = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.obv_occ[:, :, :, :, 1:] = torch.stack((x_mesh, y_mesh, z_mesh), dim=-1) / self.cfg.env_size

        self.env_episode = np.ones(self.cfg.num_envs).astype(np.int32) * -1

        self.last_coverage_ratio = torch.zeros(self.cfg.num_envs, device=self.device).reshape(-1, 1)
        self.coverage_ratio_reward = torch.zeros(self.cfg.num_envs,  device=self.device).reshape(-1, 1)
        self.sub_goal = torch.zeros(self.cfg.num_envs,  device=self.device).reshape(-1, 1)

        self.last_face_ratio = torch.zeros(self.cfg.num_envs, device=self.device).reshape(-1, 1)
        self.face_ratio = torch.zeros(self.cfg.num_envs,  device=self.device).reshape(-1, 1)
 
        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        rescale_scene(scene_prim_root="/World/ground/Environment", max_len=13e4)
 

        

        # prevent mirror
        self.scene.clone_environments(copy_from_source=True)#True)

        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # sensor
        #self._camera = Camera(self.cfg.camera)
        self._camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["camera"] = self._camera
        
        #self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # occupancy grid
        grid_size = (self.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)
        self.grid = OccupancyGrid(self.cfg.env_size, grid_size, self.cfg.decrement, self.cfg.increment, 
                                  self.cfg.max_log_odds, self.cfg.min_log_odds, self.device)
        
        # rescale robot
        #robot_scale = get_robot_scale("/World/envs/env_0/Robot", 1)
        #for i in range(self.num_envs):
        #    rescale_robot(f"/World/envs/env_{i}/Robot", robot_scale)

        # grid for random initialization
        self.goal_grid = np.ones((self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size))
        self.init_vox_pos = np.zeros((self.num_envs, 3)).astype(np.int32)

        # scene
        if not self.cfg.preplan:
            scenes_path = []
            # Loop over each batch number
            for batch_num in range(1, 7):  # Range goes from 1 to 6 inclusive
                # Generate the path pattern for the glob function
                path_pattern = os.path.join(f'../Dataset/Raw_Rescale_USD/BATCH_{batch_num}', '**', '*[!_non_metric].usd')
                # Use glob to find all .usd files (excluding those ending with _non_metric.usd) and add to the list
                scenes_path.extend(sorted(glob.glob(path_pattern, recursive=True)))
            # only use one building
            scenes_path = scenes_path[0:256]
            #scenes_path = scenes_path[0:1]
            self.cfg_list = []
            for scene_path in scenes_path:
                self.cfg_list.append(UsdFileCfg(usd_path=scene_path))
            
            # random translation
            tx = torch.randint(-2, 2 + 1, (256, 1))
            ty = torch.randint(-2, 2 + 1, (256, 1))
            tz = torch.randint(0, 0 + 1, (256, 1))*0
            self.txyz = torch.cat((tx, ty, tz), dim=1).to(self.device)
            # create scenes
            _, scene_lists = spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
                                                            env_ids=torch.arange(self.cfg.num_envs), my_asset_list=self.cfg_list, translation=self.txyz)
            
            self.first_asset_list = scene_lists

            env_ids = torch.arange(self.cfg.num_envs)
            for i, scene in enumerate(scene_lists):
                #TODO: chnage Occ to Occ_new_2000
                path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ_new_2000")) #"Occ"))
                occ_path = os.path.join(path, "fill_occ_set.pkl")
                # To load the occupied voxels from the file
                # TODO NOTED THAT OCCS MAY HAVE BEEN SWAPPED
                with open(occ_path, 'rb') as file:
                    self.occs[env_ids[i]] = pickle.load(file)    
                path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ_new_2000"))
                #print(path)
                occ_path = os.path.join(path, "occ.npy")
                self.gt_occs[env_ids[i]] = torch.tensor(np.where(np.load(occ_path)==2, 1, 0)).to(self.device)
                #self.gt_occs[env_ids[i]] = torch.tensor(np.load(occ_path)).permute(1, 2, 0).to(self.device)
                occ_path = os.path.join(path, "faces.npy")
                self.gt_faces[env_ids[i]] = torch.tensor(np.load(occ_path)).to(self.device)
                #print(torch.sum(self.gt_faces[env_ids[i]]), torch.sum(self.gt_occs[env_ids[i]]), torch.sum(self.gt_faces[env_ids[i]][self.gt_occs[env_ids[i]].bool()]))

                # shift
                #print(self.occs)
                #print(occ_path)
                self.occs[env_ids[i]] = shift_occs(self.occs[env_ids[i]], self.txyz[env_ids[i]], self.cfg.grid_size, self.cfg.env_size)
                self.gt_occs[env_ids[i]] = shift_gt_occs(self.gt_occs[env_ids[i]], self.txyz[env_ids[i]], self.cfg.grid_size, self.cfg.env_size)
                self.gt_faces[env_ids[i]] = shift_gt_faces(self.gt_faces[env_ids[i]], self.txyz[env_ids[i]], self.cfg.grid_size, self.cfg.env_size)

            """
            for i, scene in enumerate(scene_lists):
                #TODO: change Occ to Occ_new_2000
                path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ"))
                occ_path = os.path.join(path, "fill_occ_set.pkl")
                # To load the occupied voxels from the file
                with open(occ_path, 'rb') as file:
                    self.occs[env_ids[i]] = pickle.load(file)      

            for i, scene in enumerate(scene_lists):
                path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ_new_2000"))           
                occ_path = os.path.join(path, "occ.npy")
                self.gt_occs[env_ids[i]] = torch.tensor(np.where(np.load(occ_path)==2, 1, 0)).to(self.device)
            """
        self.fg_masks = torch.zeros((self.cfg.num_envs, self.cfg.camera_h, self.cfg.camera_w)).to(self.device)
        self.point_cloud = o3d.geometry.PointCloud()
        self.scene_id=0
        self.h_limit = (torch.ones(self.cfg.num_envs).to(self.device)*10*self.cfg.grid_size/self.cfg.env_size).int()
        # for visualization
        #temp = set()
        #for x in range(self.cfg.grid_size):
        #    for y in range(self.cfg.grid_size):
        #        for z in range(self.cfg.grid_size):
        #            if self.gt_occs[0][x, y, z]==1:
        #                temp.add((z, x, y))
        #cell_size = self.cfg.env_size/self.cfg.grid_size  # meters per cell
        #slice_height = self.cfg.env_size / self.cfg.grid_size  # height of each slice in meters 
        #create_blocks_from_occ_set(0, env_origin, temp, cell_size, slice_height, self.cfg.env_size)
        self.used_nearest = True 
        #self.out_limit = torch.zeros(self.cfg.num_envs)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._index >= self.num_points-1:
            self._index = -1  # Reset index to loop the trajectory
        self._index += 1

        self.env_step += 1

        # action
        # the default action space in wrapper is -100 to 100
        #self._actions = actions.clone().clamp(-1.0, 1.0)
        self._actions = actions.clone()
        self._xyz = self._actions[:, :3]
        # TODO tune z
        # x: -9.5~9.5, y: -9.5~9.5, z: 0~10
        # using 9.5 instead of 10 to avoid boundary case for collision detection
        #self._xyz = (self._xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-5e-1, self.cfg.env_size/2.0-5e-1, 1.0*self.cfg.env_size/4.0]).to(self.device)
        #self._xyz = (self._xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0, self.cfg.env_size/2.0, 1.0*self.cfg.env_size/4.0]).to(self.device)
        #self._xyz = (self._xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([0, self.cfg.env_size/2.0, 0.6*self.cfg.env_size/4.0]).to(self.device)
        self._xyz = (self._xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3]).to(self.device)

        used_yp = False #False
        #self.used_nearest = True
        if self.used_nearest:
            #print(self._actions.shape)
            self.real_xyz = self._actions[:, 6:9]
            #self.real_xyz = (self.real_xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-5e-1, self.cfg.env_size/2.0-5e-1, 1.0*self.cfg.env_size/4.0]).to(self.device)
            self.real_xyz = (self.real_xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3]).to(self.device)

        if used_yp:
            self._yaw = self._actions[:,3]*torch.pi
            #self._pitch = (self._actions[:,4]+1)*torch.pi/4.
            self._pitch = (self._actions[:,4]+1/5.)/2.*torch.pi*5/6.
        else:
            # 0~1
            lookatxyz = self._actions[:, 3:6]
            # -1~1
            lookatxyz = lookatxyz*2-1
            # -9.5~9.5, y: -9.5~9.5, z: 0~10
            #lookatxyz = (lookatxyz+torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-5e-1, self.cfg.env_size/2.0-5e-1, 1.0*self.cfg.env_size/4.0]).to(self.device)
            lookatxyz = (lookatxyz+torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3]).to(self.device)
            # compute yaw and pitch
            dxyz = lookatxyz - self._xyz + 1e-6
            # Calculate yaw using torch functions
            self._yaw = torch.atan2(dxyz[:, 1], dxyz[:, 0])
            # Calculate pitch using torch functions
            self._pitch = torch.atan2(dxyz[:, 2], torch.sqrt(dxyz[:, 0]**2 + dxyz[:, 1]**2))
            # Normalize pitch as specified
            self._pitch = ((self._pitch / torch.pi + 1/5.0) / 2.0 * torch.pi * 5/6.0)

        hard_occ = torch.where(self.obv_occ[:, :, :, 1:, 0] >= 0.6, 1, 0)
        num_match_occ = torch.sum(torch.logical_and(hard_occ, self.gt_occs[:, :, :, 1:]), dim=(1, 2, 3))
        total_occ = torch.sum(self.gt_occs[:, :, :, 1:], dim=(1, 2, 3))
        self.last_coverage_ratio = (num_match_occ/total_occ).reshape(-1, 1)
        self.last_face_ratio = (torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]),(1,2,3,4))/torch.sum(self.gt_faces[:, :, :, 1:, :], (1,2,3,4))).reshape(-1, 1)

    def _apply_action(self):
        if self.cfg.preplan:
            target_position = np.array([self.x[self._index], self.y[self._index], self.z[self._index]]).astype(np.float32)
            yaw, pitch = compute_orientation(target_position)
            target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
            target_position = torch.from_numpy(target_position).unsqueeze(0).to(self.device)
            target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
            target_orientation = target_orientation.repeat(self.num_envs,1)
            yaw = yaw * torch.ones(self.num_envs,).to(self.device)
            pitch_radians = 0.1 * torch.ones(self.num_envs,).to(self.device)
            
        else:
            target_position = self._xyz
            pitch_radians = self._pitch
            yaw = self._yaw
            # roll, pitch, yaw
            target_orientation = rot_utils.euler_angles_to_quats(torch.cat([torch.zeros(yaw.shape[0],1), torch.zeros(yaw.shape[0],1), yaw.cpu().unsqueeze(1)],dim=1).numpy(), degrees=False)
            target_orientation = torch.from_numpy(target_orientation)
        
        rotation_matrix = rot_utils.quats_to_rot_matrices(target_orientation)
        env_ids = torch.arange(self.num_envs).to(self.device)

        # record robot pos and pitch, yaw
        self.robot_pos = target_position + self._terrain.env_origins
        self.robot_ori[:, 0] = self._pitch
        self.robot_ori[:, 1] = self._yaw

        # setup robot position
        root_state = torch.ones((self.num_envs, 13)).to(self.device) * 0
        root_state[:, :3] = target_position
        root_state[:,3:7] = target_orientation
        self.target_orientation = target_orientation
        self.target_position = target_position
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        
        drone_euler = torch.cat([torch.zeros(yaw.shape[0],1).to(self.device), pitch_radians.unsqueeze(1), yaw.unsqueeze(1)], dim=1).cpu()
        pitch_quat = torch.from_numpy(rot_utils.euler_angles_to_quats(drone_euler, degrees=False)).float()
        orientation_camera = convert_orientation_convention(pitch_quat, origin="world", target="ros")
        #self._camera.set_world_poses(root_state[:, :3]+torch.tensor(np.dot(self.cfg.camera_offset,rotation_matrix)).to(self.device), orientation_camera)

        x_new = root_state[:, 0] + self.cfg.camera_offset[0] * torch.cos(self._yaw) - self.cfg.camera_offset[1] * torch.sin(self._yaw)
        y_new = root_state[:, 1] + self.cfg.camera_offset[0] * torch.sin(self._yaw) + self.cfg.camera_offset[1] * torch.cos(self._yaw)
        z_new = root_state[:, 2] + self.cfg.camera_offset[2]
 
        new_positions = torch.stack([x_new, y_new, z_new], dim=1)

        if self.cfg.preplan:
            view_target = self._terrain.env_origins.repeat(self.num_envs,1)
            #view_target[:,2]+=5
            target_position[:,2]-=2
            self._camera.set_world_poses_from_view(target_position,view_target)
        else:
            self._camera.set_world_poses(new_positions, orientation_camera)
        # for tiled camera
        #self.sim.step()
        #self.scene.update(dt=0)
        #self._camera.update(dt=0)
 
        # do not need this if decimation is large enough?
        # temporary solution for unsync bug between camera position and image
        #for i in range(3):
        #    self.sim.step()
        #    self.scene.update(dt=0)
       
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # img: N, T', H, W, 3
        # occ: N, grid_size, grid_size, grid_size, 4
        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Dict()
        # normalize input
        #self.single_observation_space["policy"]["pose_step"] = gym.spaces.Box(low=-1., high=1., shape=(self.num_observations+1,))
        self.single_observation_space["policy"]["pose_step"] = gym.spaces.Box(low=-20., high=20., shape=(6,))
        #self.single_observation_space["policy"]["img"] = gym.spaces.Box(low=0., high=1., shape=(self.cfg.img_t*3, self.cfg.camera_h, self.cfg.camera_w))
        self.single_observation_space["policy"]["img"] = gym.spaces.Box(low=0., high=1., shape=(1, self.cfg.camera_h, self.cfg.camera_w))
        self.single_observation_space["policy"]["occ"] = gym.spaces.Box(low=-1., high=1., shape=(10, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)) # 4
        self.single_observation_space["policy"]["aux_cov"] = gym.spaces.Box(low=0., high=1., shape=(1,))
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

    def update_observations(self, env_ids=None):
        env_ids = [i for i in range(self.cfg.num_envs)] if env_ids is None else env_ids

        # get images
        depth_image = self._camera.data.output["distance_to_image_plane"].clone()[:, :, :, 0]
        #print(depth_image)
        #depth_image = self._camera.data.output["depth"][:,:,:,0].clone()
        rgb_image = self._camera.data.output["rgb"].clone()
       
 
        for i in env_ids:
            self.obv_imgs[i][1] = self.obv_imgs[i][0]
            self.obv_imgs[i][0] = rgb_image[i][:, :, :3] / 255.

        # Settings for the occupancy grid
        org_x, org_y = self.cfg.env_size/2., self.cfg.env_size/2.
        org_z = 0
        cell_size = self.cfg.env_size/self.cfg.grid_size # meters per cell
        slice_height = self.cfg.env_size/self.cfg.grid_size  # height of each slice in meters
        
        #self.out_limit = self.robot_pos[:, 2]>self.h_limit

        # collision detection
        for i in env_ids:
            self.col[i] = check_building_collision(self.occs, self.robot_pos[i].clone(), i, org_x, org_y, org_z, 
                                                   cell_size, slice_height, self._terrain.env_origins)
            if self.used_nearest:
                self.not_free[i] = check_free(self.obv_occ[i, :, :, :, 0], self.real_xyz[i].clone(), i, org_x, org_y, org_z, 
                                              cell_size, slice_height, self._terrain.env_origins, False)
            else:
                self.not_free[i] = check_free(self.obv_occ[i, :, :, :, 0], self.robot_pos[i].clone(), i, org_x, org_y, org_z, 
                                              cell_size, slice_height, self._terrain.env_origins)
            self.not_height[i] = check_height(self.h_limit[i], self.real_xyz[i].clone(), i, org_z, 
                                              cell_size, slice_height, self._terrain.env_origins, False)

        # robot pose
        for i in env_ids:
            self.obv_pose_history[i, self.env_step[i].int(), :3] = (self.robot_pos[i] - self._terrain.env_origins[i])/self.cfg.env_size
            self.obv_pose_history[i, self.env_step[i].int(), 3:] = self.robot_ori[i]/3.15
        #print(self.obv_pose_history)       

        # get camera intrinsic and extrinsic matrix
        
        intrinsic_matrix = self._camera.data.intrinsic_matrices.clone()
        camera_pos = self._camera.data.pos_w.clone()
        camera_quat = self._camera.data.quat_w_ros.clone()
        
                
        #depth_image = dis_to_z(depth_image, intrinsic_matrix)
        # prevent inf
        depth_image = torch.clamp(depth_image, 0, self.cfg.env_size*2)
        # make log odd occupancy grid
        points_3d_cam = unproject_depth(depth_image, intrinsic_matrix)
        points_3d_world = transform_points(points_3d_cam, camera_pos, camera_quat)
        
         
        if self.cfg.vis_pointcloud:
            #import pdb; pdb.set_trace()
            #colors = rgb_image[0,:,:,:-1].transpose(0,1).detach().cpu().numpy().reshape(-1, 3) / 255.0  # Normalize color values
            colors = rgb_image[0,:,:,:].transpose(0,1).detach().cpu().numpy().reshape(-1, 3) / 255.0  # Normalize color values
                            
            # Create a point cloud object from the points
            point_cloud = o3d.geometry.PointCloud()
            points = points_3d_world[0].detach().cpu().numpy()
            mask_x = (points_3d_world[0,:, 0]-self._terrain.env_origins[0][0]).abs() < self.cfg.env_size/2 - 1e-3
            mask_y = (points_3d_world[0,:, 1]-self._terrain.env_origins[0][1]).abs() < self.cfg.env_size/2 - 1e-3
            mask_z = (points_3d_world[0,:, 2] < self.cfg.env_size - 1e-3) & (points_3d_world[0,:, 2] >=0) 

            # Combine masks to keep rows where both xyz are within the range
            mask = mask_x & mask_y & mask_z
            filtered_points = points[mask.cpu().numpy()]
            filtered_colors = colors[mask.cpu().numpy()]

            point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
            point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
            # Visualize the point cloud
            self.point_cloud = merge_point_clouds(self.point_cloud, point_cloud)
            o3d.visualization.draw_geometries([self.point_cloud])
        # vis end points
        #create_blocks_from_occ_list(0, self._terrain.env_origins[0].detach().cpu().numpy(), points_3d_world[0].detach().cpu().numpy(), cell_size, slice_height, self.cfg.env_size)
       
        #for i in range(self._camera.data.pos_w.shape[0]):
        for i in env_ids:
            mask_x = (points_3d_world[i,:, 0]-self._terrain.env_origins[i][0]).abs() < self.cfg.env_size/2 - 1e-3
            mask_y = (points_3d_world[i,:, 1]-self._terrain.env_origins[i][1]).abs() < self.cfg.env_size/2 - 1e-3
            mask_z = (points_3d_world[i,:, 2] < self.cfg.env_size - 1e-3) & (points_3d_world[i,:, 2] >=0) 

            # Combine masks to keep rows where both xyz are within the range
            mask = mask_x & mask_y & mask_z
            
            offset = torch.tensor([self.cfg.env_size/2, self.cfg.env_size/2,0]).to(self.device)
            if not self.col[i] and points_3d_world[i][mask].shape[0] > 0:
            # TODO: consider all?
            #if (not self.col[i] and not self.not_free[i] and not self.not_height[i] or self.env_step[i].int()==0) and points_3d_world[i][mask].shape[0] > 0:
                ratio = self.cfg.grid_size/self.cfg.env_size
                self.grid.trace_path_and_update(i, torch.floor(self._camera.data.pos_w[i]-self._terrain.env_origins[i]+offset)*ratio, 
                                                torch.floor((points_3d_world[i]-self._terrain.env_origins[i]+offset))*ratio)
                self.grid.update_log_odds(i,
                                          torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio),
                                          occupied=True)
                self.fg_masks[i] = extract_foreground(points_3d_world[i], 0, self.cfg.camera_h, self.cfg.camera_w, mask)
                self.obv_face[i] = torch.logical_or(self.obv_face[i], 
                                                    get_seen_face(torch.unique(torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio).int(), dim=0), 
                                                    torch.floor((self._camera.data.pos_w[i]-self._terrain.env_origins[i]+offset)*ratio), self.cfg.grid_size, self.device))
            else:
                self.fg_masks[i] = 0

        # save images
        if self.cfg.save_img:
            for i in self.cfg.save_env_ids:
                if i >= self.num_envs:
                    break
                if self.env_episode[i]%self.cfg.save_img_freq != 0:
                    continue
                root_path = os.path.join('camera_image_newset_h10limit_auxoff05', f'{self.env_episode[i]}')
                os.makedirs(root_path, exist_ok=True)
                #plt.imsave(os.path.join(root_path, f'{i}_mask_{self.env_step[i].long()}.png'),
                #           (self.fg_masks[i].detach().cpu().numpy()*255).astype(np.uint8),
                #           cmap='gray')

        # Iterate over each slice
        org_x, org_y = self.cfg.env_size/2., self.cfg.env_size/2.
        org_z = 0
        cell_size = min(self.cfg.env_size/self.cfg.grid_size, self.cfg.env_size/self.cfg.grid_size)  # meters per cell
        slice_height = self.cfg.env_size / self.cfg.grid_size  # height of each slice in meters
 
        if points_3d_world.size()[0] > 0:
            self.probability_grid = self.grid.log_odds_to_prob(self.grid.grid)
            # N, x_size, y_size, z_size
            # 0: free, 1: unknown, 2: occupied
            #self.obv_occ[:, :, :, :, 0] = torch.where(self.probability_grid <= 0.3, 0, torch.where(self.probability_grid <= 0.6, 1, 2))
            self.obv_occ[:, :, :, :, 0] = self.probability_grid.clone()

            if self.cfg.vis_occ:
                vis_occ = torch.where(self.probability_grid <= 0.3, 0, torch.where(self.probability_grid <= 0.6, 1, 2))
                vis_occ = vis_occ.cpu().numpy()
                for j in range(self.obv_occ.shape[0]):
                    for i in range(self.cfg.grid_size):
                        # vis occ
                        create_blocks_from_occupancy(j, self._terrain.env_origins[j].cpu().numpy(), 
                                                     vis_occ[j, :, :, i], cell_size, i*slice_height, i, self.cfg.env_size, 2, 60)
                        # vis free
                        #create_blocks_from_occupancy(j, self._terrain.env_origins[j].cpu().numpy(), 
                        #                             vis_occ[j, :, :, i], cell_size, i*slice_height, i, self.cfg.env_size, 0, 30)

        #for i in range(self.cfg.num_envs):
            # free to unknown
            #self.obv_occ[i, :, :, self.h_limit[i]+1:][self.obv_occ[i, :, :, self.h_limit[i]+1:] < 0.5] = 0.5

        hard_occ = torch.where(self.obv_occ[:, :, :, :, 0] >= 0.6, 1, 0)



        #print(torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :])), torch.sum(self.gt_faces[:, :, :, 1:, :]))
        # remove the occluded face by checking current occ grid
        # the face amount may still larger than gt because current occ grid is incomplete
        #print(torch.sum(self.obv_face[:, :, :, 1:]))
        self.obv_face = remove_occluded_face(self.cfg.grid_size, hard_occ, self.obv_face, self.device)
        #print(torch.sum(self.obv_face[:, :, :, 1:]))
        #print(torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :])), torch.sum(self.gt_faces[:, :, :, 1:, :]))
        

        hard_occ = hard_occ[:, :, :, 1:]

        num_match_occ = torch.sum(torch.logical_and(hard_occ, self.gt_occs[:, :, :, 1:]), dim=(1, 2, 3))
        total_occ = torch.sum(self.gt_occs[:, :, :, 1:], dim=(1, 2, 3))
        self.coverage_ratio_reward = (num_match_occ/total_occ).reshape(-1, 1)
        self.face_ratio = (torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]),(1,2,3,4))/torch.sum(self.gt_faces[:, :, :, 1:, :], (1,2,3,4))).reshape(-1, 1)


        # save images
        if self.cfg.save_img:
            for i in self.cfg.save_env_ids:
                if i >= self.num_envs:
                    break
                if self.env_episode[i]%self.cfg.save_img_freq != 0:
                    continue
                root_path = os.path.join('camera_image_newset_h10limit_auxoff05', f'{self.env_episode[i]}')
                
                os.makedirs(root_path, exist_ok=True)
                #print(f"save {i}_rgb_{self.env_step[i].long()}.png")
                x, y, z = self.obv_pose_history[i, self.env_step[i].int(), :3] * self.cfg.env_size
                rew = self.coverage_ratio_reward[i, 0] 
                plt.imsave(os.path.join(root_path, f'{i}_depth_{self.env_step[i].long()}_{x:.1f}_{y:.1f}_{z:.1f}_{rew:.3f}_{self.face_ratio[i][0]:.3f}_{self.not_free[i]}_{self.not_height[i]}_{self.real_xyz[i][0]:.3f}_{self.real_xyz[i][1]:.3f}_{self.real_xyz[i][2]:.3f}_{self.h_limit[i]}.png'),
                           np.clip(depth_image[i].detach().cpu().numpy(),0,20).astype(np.uint8),
                           cmap='gray',
                           vmin=0,
                           vmax=20)
                plt.imsave(os.path.join(root_path, f'{i}_rgb_{self.env_step[i].long()}_{x:.1f}_{y:.1f}_{z:.1f}_{rew:.3f}_{self.face_ratio[i][0]:.3f}_{self.not_free[i]}_{self.not_height[i]}_{self.real_xyz[i][0]:.3f}_{self.real_xyz[i][1]:.3f}_{self.real_xyz[i][2]:.3f}_{self.h_limit[i]}.png'),
                           rgb_image[i].detach().cpu().numpy().astype(np.uint8))
 

    def _get_observations(self) -> dic:       
        # pose: N, T, 7
        # img: N, T', H, W, 3
        # occ: N, grid_size, grid_size, grid_size, 4
        #if points_3d_world.size()[0] > 0:
        #    pc_markers.visualize(translations=points_3d_world[0])
        
        pose_step = torch.cat([self.obv_pose_history.reshape(self.cfg.num_envs, -1), 
                               self.env_step.to(self.device).reshape(self.cfg.num_envs, -1)/self.cfg.total_img], dim=1)

        occ_face = torch.cat((self.obv_occ, self.obv_face), dim=-1)

        gray_scale_img = torch.mean(self.obv_imgs[:, 0, :, :, :], (3))

        rew_mask = (torch.tensor(self.col)==False).float().to(self.device).reshape(-1, 1)

        # Reshape h_limit to have the same batch dimension as pose_step
        h_limit_expanded = self.h_limit.unsqueeze(1)  # Shape becomes [num_envs, 1]
        # Concatenate along the last dimension
        pose_step = torch.cat(
            [torch.stack([self.obv_pose_history[i, self.env_step[i].int()] for i in range(self.cfg.num_envs)]).reshape(-1, 5),
            h_limit_expanded],
            dim=1
        )

        # Collect the last two obv_pose_history entries per environment, padding with zeros if necessary
        """
        pose_step_list = []
        for i in range(self.cfg.num_envs):
            env_step = int(self.env_step[i].item())  # Current step for the environment
            
            if env_step == 0:  # Only one step available
                history = torch.cat([
                    torch.zeros(1, self.obv_pose_history.shape[2], device=self.obv_pose_history.device),  # One zero row
                    self.obv_pose_history[i, :1]  # First available step
                ], dim=0)
            else:  # Two or more steps available
                history = self.obv_pose_history[i, max(0, env_step - 1):env_step + 1]  # Take last two steps

            pose_step_list.append(history)
        # Stack pose histories and reshape
        pose_history_combined = torch.stack(pose_step_list).reshape(-1, 10)  # Shape [num_envs, 10]
        # Concatenate pose_history_combined with h_limit_expanded
        pose_step = torch.cat([pose_history_combined, h_limit_expanded], dim=1)  # Final shape [num_envs, 11]
        """
        obs = {#"pose_step": pose_step,
               "pose_step": pose_step,
               #"img": self.obv_imgs.permute(0, 1, 4, 2, 3).reshape(-1, 3 * self.cfg.img_t, self.cfg.camera_h, self.cfg.camera_w),
               "img": gray_scale_img.reshape(-1, 1 * 1, self.cfg.camera_h, self.cfg.camera_w),
               #"occ": self.obv_occ.permute(0, 4, 1, 2, 3),
               "occ": occ_face.permute(0, 4, 1, 2, 3),
               "aux_cov": (self.face_ratio-self.last_face_ratio) * rew_mask
              }

        #print(obs["pose"][0].reshape(50, 5))
        #print("A", obs["img"][0].reshape(2, 3, 300, 300)[0])
        #print("B", obs["img"][0].reshape(2, 3, 300, 300)[1])
        #print(obs["occ"][0, 0, :, :, :])
        observations = {"policy": obs}
       

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # update observation first because _get_rewards is after _get_observations
        #self.update_observations()

        hit_face = torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]))
        total_face = torch.sum(self.gt_faces[:, :, :, 1:, :])

        #hard_occ = torch.where(self.obv_occ[:, :, :, 1:, 0] >= 0.6, 1, 0)

        #num_match_occ = torch.sum(torch.logical_and(hard_occ, self.gt_occs[:, :, :, 1:]), dim=(1, 2, 3))
        #total_occ = torch.sum(self.gt_occs[:, :, :, 1:], dim=(1, 2, 3))
        #print("Cov", num_match_occ/total_occ, self.last_coverage_ratio)
        #print(torch.sum(torch.where(self.obv_occ[:, :, :, 1:, 0]==2, 1, 0), dim=(1, 2, 3))/total_occ)
        #self.coverage_ratio_reward = (num_match_occ/total_occ).reshape(-1, 1)
        
        fg_ratio = torch.sum(self.fg_masks, dim=(1, 2)).reshape(-1, 1)/(self.cfg.camera_w*self.cfg.camera_h) 

        # centroid distance
        distances = compute_distance_to_center_distance(self.fg_masks, self.cfg.camera_w, self.cfg.camera_h).reshape(-1, 1)
        prox = 1.0 - distances
        sub_goal_reward = torch.logical_and(self.coverage_ratio_reward>=0.9, self.sub_goal==0)
        self.sub_goal[sub_goal_reward] = 1.
 
        ssim_icr = (2*self.coverage_ratio_reward/(self.coverage_ratio_reward**2+1)).reshape(-1, 1)
        ssim_fg = (2*fg_ratio/(fg_ratio**2+1)).reshape(-1, 1)
        ssim_fgc = (2*prox/(prox**2+1)).reshape(-1, 1)
        gi_ssim = ssim_icr * ssim_fg * ssim_fgc

        factor = torch.ones(1).to(self.device) * 2
        factor_small = torch.ones(1).to(self.device) * 1
        rew_mask = (torch.tensor(self.col)==False).float().to(self.device).reshape(-1, 1)
        free_mask = (torch.tensor(self.not_free)==False).float().to(self.device).reshape(-1, 1)
        h_mask = (torch.tensor(self.not_height)==False).float().to(self.device).reshape(-1, 1)
        all_mask = ((torch.tensor(self.col).bool() | (self.face_ratio==self.last_face_ratio).reshape(-1).bool().cpu() | torch.tensor(self.not_free).bool() | torch.tensor(self.not_height).bool())==False).float().to(self.device).reshape(-1, 1)
        #face_ratio = (torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]),(1,2,3,4))/torch.sum(self.gt_faces[:, :, :, 1:, :], (1,2,3,4))).reshape(-1, 1)

        """
        rewards = {
            #"coverage_ratio": (self.coverage_ratio_reward - self.last_coverage_ratio) * self.cfg.occ_reward_scale * rew_mask,
            "face_ratio": (self.face_ratio-self.last_face_ratio) * rew_mask * self.cfg.occ_reward_scale * free_mask * h_mask,
            #"coverage_ratio": (ssim_icr*1+0) * self.cfg.occ_reward_scale * rew_mask * (self.coverage_ratio_reward - self.last_coverage_ratio),
            "collision": torch.tensor(self.col).float().to(self.device).reshape(-1, 1)  * self.cfg.col_reward_scale,
            #"goal": (self.coverage_ratio_reward >= self.cfg.goal).int().reshape(-1, 1) * 120. * rew_mask,
            #"goal": (self.face_ratio >= self.cfg.goal).int().reshape(-1, 1) * 120. * rew_mask,
            "penalty": (self.face_ratio==self.last_face_ratio).int() * -10., #* rew_mask,
            "free": torch.tensor(self.not_free).to(self.device).int().reshape(-1, 1) * -10., #* rew_mask,
            "hlimit": torch.tensor(self.not_height).to(self.device).int().reshape(-1, 1) * -10., #* rew_mask,
        }
        """

        for i in range(self.cfg.num_envs):
            if (self.face_ratio-self.last_face_ratio)[i]==0:
                self.stuck[i] += 1
            else:
                self.stuck[i] = 0               
        #print(torch.unique(self.stuck))
        rewards = {
            "face_ratio": (self.face_ratio-self.last_face_ratio) * self.cfg.occ_reward_scale * rew_mask,
            "all_penalty": (1.0-all_mask)*-1, #-20.
        }

        for k in rewards.keys():
            rewards[k] /= 100. #1 #100.

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0).reshape(-1)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value.squeeze(1)
        n = self.cfg.total_img 
        self._episode_sums["status coverage_ratio"] = self.coverage_ratio_reward.squeeze()
        self._episode_sums["status obv_face"] = torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]),(1,2,3,4))/torch.sum(self.gt_faces[:, :, :, 1:, :], (1,2,3,4))
        self._episode_sums["status fg"] = (self._episode_sums["status fg"]*(n-1)+fg_ratio.squeeze())/n
        self._episode_sums["status fgc"] = (self._episode_sums["status fgc"]*(n-1)+prox.squeeze())/n
        self._episode_sums["status ssim_icr"] = (self._episode_sums["status ssim_icr"]*(n-1)+ssim_icr.squeeze())/n
        self._episode_sums["status ssim_icr_v2"] = (self._episode_sums["status ssim_icr_v2"]*(n-1)+ssim_icr.squeeze()*(self.coverage_ratio_reward - self.last_coverage_ratio).squeeze())/n
        self._episode_sums["status ssim_fg"] = (self._episode_sums["status ssim_fg"]*(n-1)+ssim_fg.squeeze())/n
        self._episode_sums["status ssim_fgc"] = (self._episode_sums["status ssim_fgc"]*(n-1)+ssim_fgc.squeeze())/n
        self._episode_sums["status gi_ssim"] = (self._episode_sums["status gi_ssim"]*(n-1)+gi_ssim.squeeze())/n
        #print(self._episode_sums["obv_face"])
        #import pdb; pdb.set_trace()
        for i in range(self.cfg.num_envs):
            self.episode_rec["x"][i].append(self.robot_pos[i, 0].clone()-self._terrain.env_origins[i][0])
            self.episode_rec["y"][i].append(self.robot_pos[i, 1].clone()-self._terrain.env_origins[i][1])
            self.episode_rec["z"][i].append(self.robot_pos[i, 2].clone()-self._terrain.env_origins[i][2])
            self.episode_rec["pitch"][i].append(self.robot_ori[i, 0].clone())
            self.episode_rec["yaw"][i].append(self.robot_ori[i, 1].clone())
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # update observation here because _get_dones is the first call after _apply_action
        self.update_observations()
        time_out = self.episode_length_buf >= self.max_episode_length-1
       
        # TODO enable this when using goal reward
        #done = torch.logical_or(self.env_step.to(self.device) >= self.cfg.total_img - 1, self.coverage_ratio_reward.squeeze() >= 0.99)
        #done = torch.logical_or(self.env_step.to(self.device) >= self.cfg.total_img - 1, self.coverage_ratio_reward.squeeze() >= self.cfg.goal)
        done = torch.logical_or(self.env_step.to(self.device) >= self.cfg.total_img - 1, self.face_ratio.squeeze() >= self.cfg.goal)
        # stuck
        #done = torch.logical_or(done, self.stuck>=10)
        #done = self.env_step.to(self.device) >= self.cfg.total_img - 1
        #done = self.face_ratio.squeeze() >= self.cfg.goal
        

        if self.cfg.preplan:
            done = self.env_step.to(self.device) >= self.cfg.total_img - 1
            time_out = time_out*0
        else:
            for i in range(self.num_envs):
                if self.coverage_ratio_reward.squeeze()[i] >= self.cfg.goal:
                    x, y, z = self.init_vox_pos[i]
                    self.goal_grid[x, y, z] += 0.5 #1
                elif (self.env_step.to(self.device) >= self.cfg.total_img - 1)[i]:
                    x, y, z = self.init_vox_pos[i]
                    self.goal_grid[x, y, z] -= 0.1 #1                   
        died = done
        #died = torch.logical_or(torch.tensor(self.col).to(self.device), done.to(self.device))
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self.point_cloud = o3d.geometry.PointCloud()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)


        #if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            #self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # if self.cfg.preplan and self.scene_id>0:
        #     print(self._episode_sums["status coverage_ratio"])
        #     file_path = 'test.npy'
        #     if os.path.exists(file_path):
        #         # Load the existing data
        #         existing_data = np.load(file_path)
        #         # Append the new data
        #         updated_data = np.concatenate((existing_data, self._episode_sums["status coverage_ratio"].cpu().numpy()))
        #     else:
        #         # If the file doesn't exist, the new data becomes the updated data
        #         updated_data = self._episode_sums["status coverage_ratio"].cpu().numpy()
            
        #     np.save('test.npy', updated_data)

        # Logging

        if not self.cfg.preplan:
            extras = dict()
            for key in self._episode_sums.keys():
                #print(self._episode_sums[key])
                #print(self.env_step.to(self.device))
                #print("ccccccccccccccccccccc")
                if "status" not in key:
                    extras["Episode Reward/" + key] = self._episode_sums[key] / self.env_step.to(self.device)
                    self._episode_sums[key][env_ids] = 0.0
            extras["Episode Reward/status coverage_ratio"] = self._episode_sums["status coverage_ratio"].clone()
            self._episode_sums["status coverage_ratio"][env_ids] = 0.0

            extras["Episode Reward/status obv face"] = self._episode_sums["status obv_face"].clone()
            self._episode_sums["status obv_face"][env_ids] = 0.0

            extras["Episode Reward/status fg"] = self._episode_sums["status fg"].clone()
            self._episode_sums["status fg"][env_ids] = 0.0

            extras["Episode Reward/status fgc"] = self._episode_sums["status fgc"].clone()
            self._episode_sums["status fgc"][env_ids] = 0.0

            extras["Episode Reward/status ssim_icr"] = self._episode_sums["status ssim_icr"].clone()
            self._episode_sums["status ssim_icr"][env_ids] = 0.0

            extras["Episode Reward/status ssim_icr_v2"] = self._episode_sums["status ssim_icr_v2"].clone()
            self._episode_sums["status ssim_icr_v2"][env_ids] = 0.0

            extras["Episode Reward/status ssim_fg"] = self._episode_sums["status ssim_fg"].clone()
            self._episode_sums["status ssim_fg"][env_ids] = 0.0

            extras["Episode Reward/status ssim_fgc"] = self._episode_sums["status ssim_fgc"].clone()
            self._episode_sums["status ssim_fgc"][env_ids] = 0.0

            extras["Episode Reward/status gi_ssim"] = self._episode_sums["status gi_ssim"].clone()
            self._episode_sums["status gi_ssim"][env_ids] = 0.0

            extras["train_cus/x"] = self.episode_rec["x"].copy()
            extras["train_cus/y"] = self.episode_rec["y"].copy()
            extras["train_cus/z"] = self.episode_rec["z"].copy()
            extras["train_cus/pitch"] = self.episode_rec["pitch"].copy()
            extras["train_cus/yaw"] = self.episode_rec["yaw"].copy()
            for i in env_ids:
                self.episode_rec["x"][i] = []
                self.episode_rec["y"][i] = []
                self.episode_rec["z"][i] = []
                self.episode_rec["pitch"][i] = []
                self.episode_rec["yaw"][i] = []

            self.extras["log"] = dict()
            self.extras["log"].update(extras)

        # reset step
        self.env_step[env_ids] = 0

        # reset imgs obv
        self.obv_imgs[env_ids] = torch.zeros((self.cfg.img_t, self.cfg.camera_h, self.cfg.camera_w, 3)).to(self.device)
        # reset pose obv
        self.obv_pose_history[env_ids] = torch.zeros(self.cfg.total_img, 5).to(self.device)
        # reset occ obv
        self.obv_occ[env_ids, :, :, :, 0] = 0.5
        # reset ray obv
        self.obv_face[env_ids] = 0

        self.last_coverage_ratio[env_ids] = 0
        self.last_face_ratio[env_ids] = 0
        self.sub_goal[env_ids] = 0
        self.fg_masks[env_ids] = 0
        
        self._actions[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        

        #default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        #self.default_root_state = default_root_state
        #self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        #self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        #self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # clear building
        # TODO NOT USE 256 building
        for env_id in env_ids:
            delete_prim(f'/World/envs/env_{env_id}/Scene')

        # reset occupancy grid
        grid_size = (self.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)
        # only reset corresponding env
        self.grid.grid[env_ids] = 0 
                                
        for env_id in env_ids:        
            self.stuck[env_id] = 0

        if self.cfg.preplan:
            scenes_path = []
            self.scene_id+=1

            for batch_num in range(1, 13):  # Range goes from 1 to 6 inclusive
                # Generate the path pattern for the glob function
                path_pattern = os.path.join(f'../Dataset/Raw_Rescale_USD/BATCH_{batch_num}', '**', '*[!_non_metric].usd')
                # Use glob to find all .usd files (excluding those ending with _non_metric.usd) and add to the list
                scenes_path.extend((glob.glob(path_pattern, recursive=True)))
            
            # Function to convert text to integers if possible
            def atoi(text):
                return int(text) if text.isdigit() else text

            # Function to split the filename into parts that are either digits or non-digits
            def natural_keys(text):
                return [atoi(c) for c in re.split(r'(\d+)', text)]

            scenes_path_sorted = sorted(scenes_path, key=natural_keys)
            #import pdb; pdb.set_trace()
            #print(self._episode_sums["status coverage_ratio"])
            
            scenes_path = scenes_path_sorted[self.scene_id:self.scene_id+1]
            #scenes_path = scenes_path_sorted[6:7]
            self.cfg_list = []
            for scene_path in scenes_path:
                self.cfg_list.append(UsdFileCfg(usd_path=scene_path))
            _, scene_lists = spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
                                                            env_ids=env_ids, my_asset_list=self.cfg_list)
            #print(self.cfg_list)
            #print(scene_lists)
        else:
            # add scene
            # TODO NOT DELETE SCENE
            #_, scene_lists = spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
            #                                             env_ids=env_ids, my_asset_list=self.cfg_list)
            #pass
            
            #scenes_path = []
            # Loop over each batch number
            #for batch_num in range(1, 7):  # Range goes from 1 to 6 inclusive
                # Generate the path pattern for the glob function
            #    path_pattern = os.path.join(f'../Dataset/Raw_Rescale_USD/BATCH_{batch_num}', '**', '*[!_non_metric].usd')
                # Use glob to find all .usd files (excluding those ending with _non_metric.usd) and add to the list
            #    scenes_path.extend(sorted(glob.glob(path_pattern, recursive=True)))
            # only use one building
            #scenes_path = scenes_path[0:256]
            #scenes_path = scenes_path[0:1]
            #temp_list = []
            #for scene_path in scenes_path:
            #    temp_list.append(UsdFileCfg(usd_path=scene_path))
            #for i in env_ids:
            #    self.cfg_list[i] = temp_list[i]

            # random translation
            # should be integer
            # otherwise
            # ex: 0.1 + 0.5 => 0.6 (0)
            #     0.6 + 0.5 => 1.1 (1)
            tx = torch.randint(-2, 2 + 1, (256, 1))
            ty = torch.randint(-2, 2 + 1, (256, 1))
            tz = torch.randint(0, 0 + 1, (256, 1))*0
            temp = torch.cat((tx, ty, tz), dim=1)
            for i in env_ids:
                self.txyz[i] = temp[i]
            # create scenes
            _, scene_lists = spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
                                                            env_ids=env_ids, my_asset_list=[UsdFileCfg(usd_path=self.first_asset_list[i]) for i in env_ids], 
                                                            translation=self.txyz[env_ids], is_random=False)
            

        
        # load occ set and gt
        # TODO DONT NEED THIS IF WE DO NOT CHANGE BUILDING
        for i, scene in enumerate(scene_lists):
            #TODO: chnage Occ to Occ_new_2000
            path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ_new_2000")) #"Occ"))
            occ_path = os.path.join(path, "fill_occ_set.pkl")
            # To load the occupied voxels from the file
            # TODO NOTED THAT OCCS MAY HAVE BEEN SWAPPED
            with open(occ_path, 'rb') as file:
                self.occs[env_ids[i]] = pickle.load(file)    
            path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ_new_2000"))
            #print(path)
            occ_path = os.path.join(path, "occ.npy")
            self.gt_occs[env_ids[i]] = torch.tensor(np.where(np.load(occ_path)==2, 1, 0)).to(self.device)
            #self.gt_occs[env_ids[i]] = torch.tensor(np.load(occ_path)).permute(1, 2, 0).to(self.device)
            occ_path = os.path.join(path, "faces.npy")
            self.gt_faces[env_ids[i]] = torch.tensor(np.load(occ_path)).to(self.device)
        
            # shift
            #print(self.cfg.grid_size, self.cfg.env_size)
            #print(self.occs[env_ids[i]])
            self.occs[env_ids[i]] = shift_occs(self.occs[env_ids[i]], self.txyz[env_ids[i]], self.cfg.grid_size, self.cfg.env_size)
            self.gt_occs[env_ids[i]] = shift_gt_occs(self.gt_occs[env_ids[i]], self.txyz[env_ids[i]], self.cfg.grid_size, self.cfg.env_size)
            self.gt_faces[env_ids[i]] = shift_gt_faces(self.gt_faces[env_ids[i]], self.txyz[env_ids[i]], self.cfg.grid_size, self.cfg.env_size)
        
        if not self.cfg.random_initial:
            target_position = np.array([self.cfg.env_size//2-1, self.cfg.env_size//2-1, self.cfg.env_size//4-1])
            yaw, pitch = compute_orientation(target_position)
            # TODO empty scene
            #yaw *= -1
            target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
            target_position = torch.from_numpy(target_position).unsqueeze(0).to(self.device)
            target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
            
            # TODO assume initial pitch is 0 for current version, yaw, xyz is the same
            orientation_camera = convert_orientation_convention(target_orientation.float(), origin="world", target="ros")
            
            default_root_state[:,:3] = target_position
            default_root_state[:,3:7] = target_orientation        
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self.default_root_state = default_root_state

            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
        
            x_new = default_root_state[:, 0] + self.cfg.camera_offset[0] * np.cos(yaw) - self.cfg.camera_offset[1] * np.sin(yaw)
            y_new = default_root_state[:, 1] + self.cfg.camera_offset[0] * np.sin(yaw) + self.cfg.camera_offset[1] * np.cos(yaw)
            z_new = default_root_state[:, 2] + self.cfg.camera_offset[2]

            new_positions = torch.stack([x_new, y_new, z_new], dim=1)
        
            orientation_camera = orientation_camera.repeat(len(env_ids), 1)

            self._camera.set_world_poses(new_positions, orientation_camera, env_ids)

            # record robot pos and pitch, yaw
            self.robot_pos[env_ids] = target_position + self._terrain.env_origins[env_ids]
            self.real_xyz[env_ids] = target_position
            self.robot_ori[env_ids, 0] = 0
            self.robot_ori[env_ids, 1] = yaw
        else:
            # Function to generate a list of all possible positions excluding occupied positions
            #def generate_possible_positions(x_range, y_range, z_range, occupied_set):
            # TODO: may need to move possible_positions = [] to for loop
            #possible_positions = []
            random_position_tensor = torch.empty((0, 3), dtype=torch.int)
            neighbor_offsets = [ (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                 (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                                 (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                                 (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                                 (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                                 (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

            # Settings for the occupancy grid
            org_x, org_y = self.cfg.env_size/2., self.cfg.env_size/2.
            org_z = 0
            cell_size = self.cfg.env_size/self.cfg.grid_size # meters per cell
            slice_height = self.cfg.env_size/self.cfg.grid_size  # height of each slice in meters

            for k, env_id in enumerate(env_ids):
                possible_positions = []
                complete_count = []
                """
                for x in range(-self.cfg.env_size//2, self.cfg.env_size//2):
                    for y in range(-self.cfg.env_size//2, self.cfg.env_size//2):
                        flag = True
                        for z in range(0, self.cfg.env_size):
                            pos = (x, y, z)
                            if pos not in self.occs[env_id]:
                                for dx, dy, dz in neighbor_offsets:
                                    neighbor_pos = (x + dx, y + dy, z + dz)
                                    if neighbor_pos in self.occs[env_id]:
                                        flag = False
                                        break
                                if flag:
                                    possible_positions.append(pos)
                """
                
                flag = True
                while flag:
                    # Randomly sample a point within the range
                    x = random.randint(-self.cfg.env_size//2+1, self.cfg.env_size//2-1)
                    y = random.randint(-self.cfg.env_size//2+1, self.cfg.env_size//2-1)
                    z = random.randint(1, int(self.cfg.env_size/2*0.6))
                    pos = (x, y, z)

                    target_x, target_y, target_z = self.txyz[env_id].cpu().numpy()
                    # Calculate the distance from the sampled point to the target point
                    distance = np.sqrt((x - target_x)**2 + (y - target_y)**2 + (z - target_z)**2)

                    # Check if the distance is within the allowed range
                    if distance < 5.5:
                        continue  # If not, sample again

                    x += org_x
                    y += org_y
                    z += org_z

                    # to voxel id
                    x = np.floor(x/cell_size).astype(np.int32)
                    y = np.floor(y/cell_size).astype(np.int32)
                    z = np.floor(z/slice_height).astype(np.int32)


                    vox_pos = (x, y, z)
                
                    # Check if the point and its neighbors are all free voxels
                    if vox_pos not in self.occs[env_id]:
                        possible_positions.append(pos)
                        flag = False 
                        #print(random.choice(possible_positions))
                random_position = possible_positions[0]
                #random_position = random.choice(possible_positions)
                
                # TODO: we do not need this for new net?
                """
                for _x in range(-self.cfg.env_size//2+1, self.cfg.env_size//2-1):
                    for _y in range(-self.cfg.env_size//2+1, self.cfg.env_size//2-1):
                        for _z in range(1, int(self.cfg.env_size/2*0.6)):
                            pos = (_x, _y, _z)
                            x = _x+org_x
                            y = _y+org_y
                            z = _z+org_z

                            # to voxel id
                            x = np.floor(x/cell_size).astype(np.int32)
                            y = np.floor(y/cell_size).astype(np.int32)
                            z = np.floor(z/slice_height).astype(np.int32)

                            vox_pos = (x, y, z)
                            if vox_pos not in self.occs[env_id]:
                                possible_positions.append(pos)
                                complete_count.append(self.goal_grid[x, y, z])
                # adaptive sampler
                inverse_count = 1. / (np.array(complete_count)+np.abs(np.min(complete_count))+1)
                weights = inverse_count / np.sum(inverse_count)
                random_position = random.choices(possible_positions, weights=weights, k=1)[0]
                """
                x = random_position[0]+org_x
                y = random_position[1]+org_y
                z = random_position[2]+org_z
                # to voxel id
                x = np.floor(x/cell_size).astype(np.int32)
                y = np.floor(y/cell_size).astype(np.int32)
                z = np.floor(z/slice_height).astype(np.int32)
                self.init_vox_pos[env_id] = [x, y, z]
                #random_position_tensor = torch.cat((torch.tensor(random_position).unsqueeze(0),random_position_tensor), dim=0)
                random_position_tensor = torch.tensor(random_position).unsqueeze(0)
                #yaw, pitch = compute_orientation(random_position)
                yaw, pitch = compute_orientation(random_position, self.txyz[env_id].cpu().numpy())
                # from initial h to max h
                self.h_limit[env_id] = random.randint(int(random_position[2] * self.cfg.grid_size/self.cfg.env_size), 10)
                if random.random() <= 0.9: 
                    self.h_limit[env_id] = random.randint(9, 10)
                # TODO: DO NOT NEED THIS FOR CONSTRAINT
                # 50% chance to multiply yaw by -1
                #if random.random() < 0.5:
                #    yaw *= -1
                #print('pitch', pitch)
                target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
                target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)

                default_root_state[k:k+1, :3] =  random_position_tensor
                default_root_state[k:k+1, 3:7] = target_orientation        
                default_root_state[k:k+1, :3] += self._terrain.env_origins[env_id]
                self.default_root_state = default_root_state
                # TODO assume initial pitch is 0 for current version, yaw, xyz is the same
                target_orientation = rot_utils.euler_angles_to_quats(np.array([0, -pitch, yaw]), degrees=False)
                target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
                orientation_camera = convert_orientation_convention(target_orientation.float(), origin="world", target="ros")

                x_new = default_root_state[k:k+1, 0] + self.cfg.camera_offset[0] * np.cos(yaw) - self.cfg.camera_offset[1] * np.sin(yaw)
                y_new = default_root_state[k:k+1, 1] + self.cfg.camera_offset[0] * np.sin(yaw) + self.cfg.camera_offset[1] * np.cos(yaw)
                z_new = default_root_state[k:k+1, 2] + self.cfg.camera_offset[2]
                new_positions = torch.stack([x_new, y_new, z_new], dim=1)

                #import pdb; pdb.set_trace()
                #orientation_camera = orientation_camera.repeat(len(env_ids), 1)
                # TODO May need to set robot positions
                self._camera.set_world_poses(new_positions, orientation_camera, [env_id])
                self.robot_pos[env_id] = random_position_tensor.to(self.device) + self._terrain.env_origins[env_id]
                self.real_xyz[env_id] = random_position_tensor.to(self.device) 
                self.robot_ori[env_id, 0] = -pitch
                self.robot_ori[env_id, 1] = yaw
            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        #import pdb; pdb.set_trace()1
        #self._camera.set_world_poses_from_view(random_position_tensor.cuda()+self._terrain.env_origins[env_ids], self._terrain.env_origins[env_ids], env_ids)
        #cell_size = self.cfg.env_size/self.cfg.grid_size  # meters per cell
        #slice_height = self.cfg.env_size / self.cfg.grid_size  # height of each slice in meters        
        #for i in range(len(env_ids)):
        #    env_origin = self._terrain.env_origins[env_ids[i]].detach().cpu().numpy()
        #    create_blocks_from_occ_set(env_ids[i], env_origin, self.occs[env_ids[i]], cell_size, slice_height, self.cfg.env_size)
        #print(env_ids)
        #print(sorted(list(self.occs[0])))

        # offline rescale: do not need this
        #for i in env_ids:
        #    rescale_scene(f"/World/envs/env_{i}/Scene")
        self._index = -1

        self.env_episode[env_ids.cpu().numpy()] += 1
        # still need this to stablize the initial viewpoint
        for i in range(3):
            self.sim.step()
            self.scene.update(dt=0)

        self.update_observations(env_ids)
