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
from omni.isaac.core.utils.prims import get_prim_at_path
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

from .utils import bresenhamline, rescale_scene, rescale_robot, get_robot_scale, compute_orientation, create_blocks_from_occupancy, OccupancyGrid


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        
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
        robot_scale = get_robot_scale("/World/envs/env_0/Robot", 2)
        for i in range(self.num_envs):
            rescale_robot(f"/World/envs/env_{i}/Robot", robot_scale)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._index >= self.num_points-1:
            self._index = -1  # Reset index to loop the trajectory
        self._index += 1
        # TODO clamp each action
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        if self.cfg.preplan:
            target_position = np.array([self.x[self._index], self.y[self._index], self.z[self._index]])
            yaw = compute_orientation(target_position)
            target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
            target_position = torch.from_numpy(target_position).unsqueeze(0)
            target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)

        env_ids = torch.arange(self.num_envs).to(self.device)

        # apply action
        root_state = torch.ones((self.num_envs, 13)).to(self.device)
        root_state[:, :3] = target_position
        root_state[:,3:7] = target_orientation
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # temporary solution for unsync bug between camera position and image
        for i in range(3):
            self.scene.update(dt=0)
            self.sim.step()

        # TODO do this after loading occupancy grid
        # Settings for the occupancy grid
        #org_x, org_y = self.cfg.env_size/2., self.cfg.env_size/2.
        #org_z = 0
        #cell_size = min(self.cfg.env_size/self.cfg.grid_size, self.cfg.env_size/self.cfg.grid_size)  # meters per cell
        #slice_height = self.cfg.env_size / self.cfg.grid_size  # height of each slice in meters
        
        #for i in range(self.num_envs):
        #    self.check_building_collision(root_state[i, :3], i, org_x, org_y, org_z, cell_size, slice_height)

    def _get_observations(self) -> dict:

        # get images
        depth_image = self._camera.data.output["distance_to_image_plane"].clone()
        rgb_image = self._camera.data.output["rgb"].clone()

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

        # make log odd occupancy grid
        points_3d_cam = unproject_depth(depth_image, intrinsic_matrix)
        points_3d_world = transform_points(points_3d_cam, camera_pos, camera_quat)

        for i in range(self._camera.data.pos_w.shape[0]):
            mask_x = (points_3d_world[i,:, 0]-self._terrain.env_origins[i][0]).abs() < self.cfg.env_size/2
            mask_y = (points_3d_world[i,:, 1]-self._terrain.env_origins[i][1]).abs() < self.cfg.env_size/2
            mask_z = (points_3d_world[i,:, 2] < self.cfg.env_size) & (points_3d_world[i,:, 2] >=0) 

            # Combine masks to keep rows where both xyz are within the range
            mask = mask_x & mask_y & mask_z

            offset = torch.tensor([self.cfg.env_size/2, self.cfg.env_size/2,0]).to(self.device)
            if points_3d_world[i][mask].shape[0] > 0:
                ratio = self.cfg.grid_size/self.cfg.env_size
                self.grid.trace_path_and_update(i, self._camera.data.pos_w[i], 
                                                torch.floor((points_3d_world[i][mask]+offset)*ratio))
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
            self.probability_grid = self.probability_grid.cpu().numpy()
            self.probability_grid = np.where(self.probability_grid<=0.5, 1, 0)
            for j in range(self.probability_grid.shape[0]):
                for i in range(self.cfg.grid_size):
                    if self.cfg.vis_occ:
                        create_blocks_from_occupancy(j, self._terrain.env_origins[j].cpu().numpy(), 
                                                     self.probability_grid[j,:,:,i], cell_size, i*slice_height, i)

        # TODO update these to rgb, occ, and drone pose
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # TODO implement rewards
        rewards = {
            "coverage_ratio": torch.tensor([0]).repeat(self.num_envs),
            "collision": torch.tensor([0]).repeat(self.num_envs),
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length-1
        # TODO setup died when collision happens
        died = False #self._index > 2
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

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

        # TODO load all usd path
        # setup building
        scenes_path = sorted(glob.glob(os.path.join(r'/home/dsr/Documents/Dataset/Raw_USD/BATCH_1/Set_A', '**', '*[!_non_metric].usd'), recursive=True))  
        stage = omni.usd.get_context().get_stage()
        #for env_id in env_ids:
        #    stage.RemovePrim(f'/World/envs/env_{env_id}/Scene')
        cfg_list = []
        for scene_path in scenes_path:
            cfg_list.append(UsdFileCfg(usd_path=scene_path))
        spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", env_ids=env_ids, my_asset_list=cfg_list)
        # TODO need to wait for rescaling
        for i in env_ids:
            rescale_scene(f"/World/envs/env_{i}/Scene")

        self._index = -1
