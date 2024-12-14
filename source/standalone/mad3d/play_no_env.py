# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

import argparse

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
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import pickle

def main():
    """Play with stable-baselines agent."""
    # parse configuration

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
    #agent = PPO.load(checkpoint_path, env, print_system_info=True)
    agent = PPO.load(checkpoint_path, print_system_info=True)
    # reset environment
    #obs = env.reset()
    file_path = 'obs.pkl'
    with open(file_path, 'rb') as file:
        obs = pickle.load(file)
    # simulate environment
    while True:
    # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            #print(obs.keys())
            actions, _ = agent.predict(obs, deterministic=True)
            actions = np.concatenate([actions, actions], axis=1)
            print(actions)
            #actions = actions.reshape((-1, *self.action_space.shape

if __name__ == "__main__":
    # run the main function
    main()

