# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import pickle



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

from .utils import bresenhamline, check_building_collision, rescale_scene, rescale_robot, get_robot_scale, compute_orientation, create_blocks_from_occupancy, create_blocks_from_occ_set, create_blocks_from_occ_list, OccupancyGrid


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.num_envs = self.num_envs
        
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "coverage_ratio",
                "collision",
            ]
        }

        self.set_debug_vis(self.cfg.debug_vis)
        
        # for pre-planned trajectory 
        self._index=-1
        radius = self.cfg.env_size/2.0 * 0.8 # Radius of the cylinder
        height = self.cfg.env_size/2.0 * 0.5 # Height of the cylinder
        theta = [0, np.pi/2., np.pi, np.pi*3/2]
        self.num_points = len(theta)
        self.x = radius * np.cos(theta)
        self.y = radius * np.sin(theta)
        self.z = np.linspace(2, height, self.num_points)

        # line trajectory
        #self.x = np.zeros(self.num_points)
        #self.y = np.linspace(-self.cfg.env_size/2.0*0.6, self.cfg.env_size/2.0*0.6, self.num_points)
        #self.z = np.ones(self.num_points) * 4

        self.occs = [set() for i in range(self.cfg.num_envs)]
        self.gt_occs = torch.zeros((self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)).to(self.device)
        self.col = [False for i in range(self.cfg.num_envs)]

        self.robot_pos = torch.tensor([[0., 0., 0.] for i in range(self.cfg.num_envs)]).to(self.device)
        self.robot_ori = torch.tensor([[0., 0., 0., 0.] for i in range(self.cfg.num_envs)]).to(self.device)

        # obv
        # it should start from 0 instead of -1 because we need one obv for initial action
        # TODO add initial value 
        self.env_step = torch.ones(self.cfg.num_envs,) * 0
        # N,T,H,W,3
        self.obv_imgs = []
        for i in range(self.cfg.num_envs):
            self.obv_imgs.append([np.zeros((self.cfg.camera_h, self.cfg.camera_w, 3)) for i in range(self.cfg.img_t)])
        self.obv_imgs = torch.tensor(self.obv_imgs).to(self.device)
        # N,T,xyzwxyz
        self.obv_pose_history = torch.zeros(self.cfg.num_envs, self.cfg.total_img, 7).to(self.device)
        # N, x_size, y_size, z_size, label+xyz
        self.obv_occ = torch.ones(self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 4, device=self.device)
        # Generate the linearly spaced values for each dimension
        x_coords = torch.linspace(-self.cfg.env_size/2.0, self.cfg.env_size/2.0, self.cfg.grid_size, device=self.device)
        y_coords = torch.linspace(-self.cfg.env_size/2.0, self.cfg.env_size/2.0, self.cfg.grid_size, device=self.device)
        z_coords = torch.linspace(0.0, self.cfg.env_size, self.cfg.grid_size, device=self.device)
        # Create a meshgrid of the coordinates
        x_mesh, y_mesh, z_mesh = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.obv_occ[:, :, :, :, 1:] = torch.stack((x_mesh, y_mesh, z_mesh), dim=-1)

    def _setup_scene(self):
        # prevent mirror
        self.scene.clone_environments(copy_from_source=True)

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # sensor
        self._camera = Camera(self.cfg.camera)
        self.scene.sensors["camera"] = self._camera
        
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # occupancy grid
        grid_size = (self.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)
        self.grid = OccupancyGrid(self.cfg.env_size, grid_size, self.cfg.decrement, self.cfg.increment, 
                                  self.cfg.max_log_odds, self.cfg.min_log_odds, self.device)
        
        # rescale robot
        robot_scale = get_robot_scale("/World/envs/env_0/Robot", 1)
        for i in range(self.num_envs):
            rescale_robot(f"/World/envs/env_{i}/Robot", robot_scale)

        # scene

        scenes_path = []
        # Loop over each batch number
        for batch_num in range(1, 7):  # Range goes from 1 to 6 inclusive
            # Generate the path pattern for the glob function
            path_pattern = os.path.join(f'../Dataset/Raw_Rescale_USD/BATCH_{batch_num}', '**', '*[!_non_metric].usd')

            # Use glob to find all .usd files (excluding those ending with _non_metric.usd) and add to the list
            scenes_path.extend(sorted(glob.glob(path_pattern, recursive=True)))

        self.cfg_list = []
        for scene_path in scenes_path:
            self.cfg_list.append(UsdFileCfg(usd_path=scene_path))
       

    def _pre_physics_step(self, actions: torch.Tensor):
        # TODO  replace _index with env_step
        if self._index >= self.num_points-1:
            self._index = -1  # Reset index to loop the trajectory
        self._index += 1

        self.env_step +=1

        # TODO clamp each action
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._xyz=self._actions[:, :3]*self.cfg.env_size/2
        self._yaw=self._actions[:,3]*torch.pi
        self._pitch=(self._actions[:,4]+1)*torch.pi/4.

    def _apply_action(self):
        if self.cfg.preplan:
            target_position = np.array([self.x[self._index], self.y[self._index], self.z[self._index]])
            yaw = compute_orientation(target_position)
            target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
            target_position = torch.from_numpy(target_position).unsqueeze(0).to(self.device)
            target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
            target_orientation = target_orientation.repeat(self.num_envs,1)
            yaw = yaw * torch.ones(self.num_envs,).to(self.device)
            pitch_radians = 0.2 * torch.ones(self.num_envs,).to(self.device)
            
        else:
            target_position = self._xyz
            pitch_radians = self._pitch
            yaw = self._yaw
            target_orientation = rot_utils.euler_angles_to_quats(torch.cat([torch.zeros(yaw.shape[0],1), torch.zeros(yaw.shape[0],1), yaw.cpu().unsqueeze(1)],dim=1).numpy(), degrees=False)
            target_orientation = torch.from_numpy(target_orientation)
        
        #pitch_quat = torch.from_numpy(rot_utils.euler_angles_to_quats(np.array([0, pitch_radians, yaw]), degrees=False)).unsqueeze(0).float()
        env_ids = torch.arange(self.num_envs).to(self.device)

        # apply action
        #for i in range(self.num_envs):
        #    self.robot_pos[i] = target_position + self._terrain.env_origins[env_ids[i]].detach().cpu()
        #    self.robot_ori[i] = target_orientation
        self.robot_pos = target_position + self._terrain.env_origins
        self.robot_ori = target_orientation
        root_state = torch.ones((self.num_envs, 13)).to(self.device) * 0
        root_state[:, :3] = target_position
        root_state[:,3:7] = target_orientation
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        #drone_euler = rot_utils.quats_to_euler_angles(root_state[:,3:7].cpu().numpy())
        #import pdb; pdb.set_trace()
        drone_euler = torch.cat([torch.zeros(yaw.shape[0],1).to(self.device), pitch_radians.unsqueeze(1), yaw.unsqueeze(1)], dim=1).cpu()
        #drone_euler[:,1] += pitch_radians
        pitch_quat = torch.from_numpy(rot_utils.euler_angles_to_quats(drone_euler, degrees=False)).float()
        #import pdb; pdb.set_trace()
        orientation = convert_orientation_convention(root_state[:,3:7], origin="world", target="ros")
        
        orientation_camera = convert_orientation_convention(pitch_quat, origin="world", target="ros")
        self._camera.set_world_poses(root_state[:, :3]+torch.tensor(self.cfg.camera_offset).to(self.device),  orientation_camera)
        #root_state[:,:3]=self._xyz + self._terrain.env_origins
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        
        # do not need this if decimation is large enough?
        # temporary solution for unsync bug between camera position and image
        #for i in range(3):
        #    self.sim.step()
        #    self.scene.update(dt=0)
       

    def _get_observations(self) -> dict:

        # get images
        depth_image = self._camera.data.output["distance_to_image_plane"].clone()
        rgb_image = self._camera.data.output["rgb"].clone()
        for i in range(self.cfg.num_envs):
            self.obv_imgs[i][self.env_step[i].int()%self.cfg.img_t] = rgb_image[i][:, :, :3]


        # Settings for the occupancy grid
        org_x, org_y = self.cfg.env_size/2., self.cfg.env_size/2.
        org_z = 0
        cell_size = self.cfg.env_size/self.cfg.grid_size # meters per cell
        slice_height = self.cfg.env_size/self.cfg.grid_size  # height of each slice in meters
        
        # collision detection
        for i in range(self.cfg.num_envs):
            #robot_body_idx = self._robot.find_bodies("body")[0][0]
            #robot_pos = self._robot.data.body_pos_w[i, robot_body_idx 
            self.col[i] = check_building_collision(self.occs, self.robot_pos[i], i, org_x, org_y, org_z, 
                                                   cell_size, slice_height, self._terrain.env_origins)
        #print(self.col)

        # robot pose
        for i in range(self.cfg.num_envs):
            self.obv_pose_history[i, self.env_step[i].int(), :3] = self.robot_pos[i] - self._terrain.env_origins[i]
            self.obv_pose_history[i, self.env_step[i].int(), 3:] = self.robot_ori[i]
        #print(self.env_step[i], self.obv_pose_history)

        # get camera intrinsic and extrinsic matrix
        intrinsic_matrix = self._camera.data.intrinsic_matrices.clone()
        camera_pos = self._camera.data.pos_w.clone()
        camera_quat = self._camera.data.quat_w_ros.clone()
        
        # save images
        if self.cfg.save_img:
            for i in self.cfg.save_env_ids:
                if i >= self.num_envs:
                    break
                plt.imsave(f'camera_image/{i}_depth_{self._index}.png',
                           np.clip(depth_image[i].detach().cpu().numpy(),0,20).astype(np.uint8),
                           cmap='gray')
                plt.imsave(f'camera_image/{i}_rgb_{self._index}.png',
                           rgb_image[i].detach().cpu().numpy().astype(np.uint8))
        

        depth_image = torch.clamp(depth_image, 0, self.cfg.env_size*2)
        # make log odd occupancy grid
        points_3d_cam = unproject_depth(depth_image, intrinsic_matrix)
        points_3d_world = transform_points(points_3d_cam, camera_pos, camera_quat)
        
        #create_blocks_from_occ_list(0, self._terrain.env_origins[0].detach().cpu().numpy(), points_3d_world[0].detach().cpu().numpy(), cell_size, slice_height, self.cfg.env_size)

        for i in range(self._camera.data.pos_w.shape[0]):
            mask_x = (points_3d_world[i,:, 0]-self._terrain.env_origins[i][0]).abs() < self.cfg.env_size/2 - 1e-3
            mask_y = (points_3d_world[i,:, 1]-self._terrain.env_origins[i][1]).abs() < self.cfg.env_size/2 - 1e-3
            mask_z = (points_3d_world[i,:, 2] < self.cfg.env_size - 1e-3) & (points_3d_world[i,:, 2] >=0) 

            # Combine masks to keep rows where both xyz are within the range
            mask = mask_x & mask_y & mask_z
            
            offset = torch.tensor([self.cfg.env_size/2, self.cfg.env_size/2,0]).to(self.device)
            if points_3d_world[i][mask].shape[0] > 0:
                ratio = self.cfg.grid_size/self.cfg.env_size
                self.grid.trace_path_and_update(i, torch.floor(self._camera.data.pos_w[i]-self._terrain.env_origins[i]+offset)*ratio, 
                                                torch.floor((points_3d_world[i]-self._terrain.env_origins[i]+offset))*ratio)
                print(torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio).min(), torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio).max())
                self.grid.update_log_odds(i,
                                          torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio),
                                          occupied=True)
       
        # Iterate over each slice
        org_x, org_y = self.cfg.env_size/2., self.cfg.env_size/2.
        org_z = 0
        cell_size = min(self.cfg.env_size/self.cfg.grid_size, self.cfg.env_size/self.cfg.grid_size)  # meters per cell
        slice_height = self.cfg.env_size / self.cfg.grid_size  # height of each slice in meters
 
        if points_3d_world.size()[0] > 0:
            self.probability_grid = self.grid.log_odds_to_prob(self.grid.grid)
            # N, x_size, y_size, z_size
            # 0: free, 1: unknown, 2: occupied
            self.obv_occ[:, :, :, :, 0] = torch.where(self.probability_grid <= 0.3, 0, torch.where(self.probability_grid <= 0.7, 1, 2))

            if self.cfg.vis_occ:
                for j in range(self.obv_occ.shape[0]):
                    for i in range(self.cfg.grid_size):
                        create_blocks_from_occupancy(j, self._terrain.env_origins[j].cpu().numpy(), 
                                                     self.obv_occ[j, :, :, i, 0].cpu().numpy(), cell_size, i*slice_height, i, self.cfg.env_size, 2, 60)
                        #create_blocks_from_occupancy(j, self._terrain.env_origins[j].cpu().numpy(), 
                        #                             self.obv_occ[j, :, :, i, 0].cpu().numpy(), cell_size, i*slice_height, i, self.cfg.env_size, 0, 30)
        #print("grid shape", self.occupancy_grid.shape)

        # pose: N, T, 7
        # img: N, T', H, W, 3
        # occ: N, grid_size, grid_size, grid_size, 7

        # TODO update these to rgb, occ, and drone pose
        obs = torch.cat(
            [
                self.obv_pose_history.reshape(self.cfg.num_envs, -1),
            ],
            dim=-1,
        )
        
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        num_match_occ = torch.sum(torch.logical_and(torch.where(self.obv_occ[:, :, :, :, 0]==2, 1, 0), self.gt_occs), dim=(1, 2, 3))
        total_occ = torch.sum(self.gt_occs, dim=(1, 2, 3))
       
        # TODO implement rewards
        rewards = {
            "coverage_ratio": (num_match_occ/total_occ).reshape(-1, 1),
            "collision": torch.tensor(self.col).float().reshape(-1, 1).to(self.device),
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0).reshape(-1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length-1
        # TODO setup died when collision happens
        done = self.env_step >= self.cfg.total_img
        died = done 
        #died = self._index > 2

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # reset step
        self.env_step[env_ids] = 0

        # reset imgs obv
        for i in env_ids:
            self.obv_imgs[i] = torch.zeros((self.cfg.img_t, self.cfg.camera_h, self.cfg.camera_w, 3)).to(self.device)
        
        # reset pose obv
        for i in env_ids:
            self.obv_pose_history[i] = torch.zeros(self.cfg.total_img, 7).to(self.device)

        # reset occ obv
        for i in env_ids:
            self.obv_occ[i, :, :, :, 0] = 1

        self._actions[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.default_root_state = default_root_state
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # clear building
        for env_id in env_ids:
            delete_prim(f'/World/envs/env_{env_id}/Scene')

        # reset occupancy grid
        grid_size = (self.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)
        self.grid = OccupancyGrid(self.cfg.env_size, grid_size, self.cfg.decrement, self.cfg.increment, 
                                  self.cfg.max_log_odds, self.cfg.min_log_odds, self.device)
                                  
        # add scene
        _, scene_lists = spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
                                                        env_ids=env_ids, my_asset_list=self.cfg_list)

        
        # load occ set and gt
        for i, scene in enumerate(scene_lists):
            path, file = os.path.split(scene.replace("Raw_Rescale_USD", "Occ"))
            occ_path = os.path.join(path, "fill_occ_set.pkl")
            # To load the occupied voxels from the file
            with open(occ_path, 'rb') as file:
                self.occs[env_ids[i]] = pickle.load(file)      
            
            occ_path = os.path.join(path, "occ.npy")
            self.gt_occs[env_ids[i]] = torch.tensor(np.load(occ_path)).permute(1, 2, 0).to(self.device)
           
       
        cell_size = self.cfg.env_size/self.cfg.grid_size  # meters per cell
        slice_height = self.cfg.env_size / self.cfg.grid_size  # height of each slice in meters
        
        #for i in range(len(env_ids)):
        #    env_origin = self._terrain.env_origins[env_ids[i]].detach().cpu().numpy()
        #    create_blocks_from_occ_set(env_ids[i], env_origin, self.occs[env_ids[i]], cell_size, slice_height, self.cfg.env_size)
        #print(env_ids)
        #print(sorted(list(self.occs[0])))

        # offline rescale: do not need this
        #for i in env_ids:
        #    rescale_scene(f"/World/envs/env_{i}/Scene")

        self._index = -1
