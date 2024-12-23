# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
#parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
import yaml
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import sys
sys.path.append('/projects/MAD3D/Zhuoli/IsaacLab/source/standalone/mad3d/neural_rendering')
from neu_nbv_utils import util
from neural_rendering.evaluation.pretrained_model import PretrainedModel
from dotmap import DotMap

def get_nbv_ref_index(
    model, images, poses, focal, c, z_near, z_far, candidate_list, budget, ref_index
):
    _, _, H, W = images.shape

    for i in range(budget):
        remain_candidate_list = list(set(candidate_list) - set(ref_index))
        reward_list = []

        model.network.encode(
            images[ref_index].unsqueeze(0),
            poses[ref_index].unsqueeze(0),
            focal.unsqueeze(0),
            c.unsqueeze(0),
        )

        for target_view in remain_candidate_list:
            novel_pose = poses[target_view]
            import pdb; pdb.set_trace()
            target_rays = util.gen_rays(
                novel_pose.unsqueeze(0), W, H, focal, z_near, z_far, c
            )
            target_rays = target_rays.reshape(1, H * W, -1)
            predict = DotMap(model.renderer_par(target_rays))
            uncertainty = predict["uncertainty"][0]
            reward = torch.sum(uncertainty**2).cpu().numpy()
            reward_list.append(reward)

        nbv_index = np.argmax(reward_list)
        new_ref_index = remain_candidate_list[nbv_index]
        ref_index.append(new_ref_index)

    return ref_index

def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
     
    agent = PPO.load(checkpoint_path, print_system_info=True)
    # reset environment
    obs = env.reset()
    
    log_path = os.path.join("source","standalone","mad3d","neural_rendering", "logs", 'dtu_training')
    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)
    import pdb; pdb.set_trace()
    neu_nbv_model = PretrainedModel(cfg["model"], ckpt_file, 'cuda', [0])

    batch_size = 5
    channels = 3
    H, W = 64, 64  # Image height and width

    # Fake Images (Random Tensor)
    images = torch.randn((batch_size, channels, H, W)).cuda()

    # Camera poses: N poses represented as 4x4 matrices
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()  # Identity poses as example

    # Focal length and camera center
    focal = torch.tensor([50.0,50.0]).cuda()  # Example focal length
    c = torch.tensor([0.0, 0.0]).cuda()  # Example camera center

    # Near and far plane distances
    z_near = 0.1
    z_far = 3.5

    # Candidate view list and initial reference index
    candidate_list = list(range(batch_size))
    initial_ref_index = [0]  # Start with view 0 as the reference

    # Budget for iterations
    budget = 0
    ref_index = get_nbv_ref_index(
                                neu_nbv_model,
                                images,
                                poses,
                                focal,
                                c,
                                z_near,
                                z_far,
                                candidate_list,
                                budget,
                                copy.deepcopy(initial_ref_index),
                            )
    print(ref_index)
    import pdb; pdb.set_trace()
    
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            print(obs.keys())
            import pdb; pdb.set_trace()
            actions, _ = agent.predict(obs, deterministic=True)
            actions = np.concatenate([actions, actions], axis=1)
            #actions = actions.reshape((-1, *self.action_space.shape))
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
