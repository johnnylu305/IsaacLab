# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from sb3_encoder import CustomCombinedExtractor
from stable_baselines3.common.utils import safe_mean


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir)   
        self.freq = 200 

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        current_step = self.num_timesteps
        for i, info in enumerate(infos):
            episode_info = info['episode']
            if episode_info:
                for key, value in episode_info.items():
                    if "Episode Reward" in key:
                        #print(i, key, value)
                        #self.logger.record(f"Episode Reward Env {i}/{key.split('/')[1]}", value.detach().cpu().item())
                        self.writer.add_scalar(f"Episode Reward Env {i}/{key.split('/')[1]}", value.detach().cpu().item(), current_step)
                    elif "train_cus" in key:
                        for j, v in enumerate(value):
                            self.writer.add_scalar(f"Episode Reward Env {i}/{key.split('/')[1]}", v, current_step-(len(value)-j-1)*len(infos))                        
        if self.num_timesteps % self.freq == 0:
            x_s, y_s, z_s, pitch_s, yaw_s = torch.exp(self.model.policy.log_std).detach().cpu().numpy()
            self.writer.add_scalar("train_cus/x_std", x_s, current_step)
            self.writer.add_scalar("train_cus/y_std", y_s, current_step)
            self.writer.add_scalar("train_cus/z_std", z_s, current_step)
            self.writer.add_scalar("train_cus/pitch_std", pitch_s, current_step)
            self.writer.add_scalar("train_cus/yaw_std", yaw_s, current_step)
        return True
        
    def _on_training_end(self) -> None:
        self.writer.close()                


# Define a learning rate schedule function
def custom_lr_schedule(initial_value, end_value, decrease_end_iteration, total_timesteps):
    """
    Custom learning rate schedule that decreases to end_value by decrease_end_iteration
    and stays constant at end_value after that.
    
    :param initial_value: (float) Initial learning rate.
    :param end_value: (float) The final learning rate after decreasing.
    :param decrease_end_iteration: (int) The iteration where the learning rate stops decreasing.
    :param total_timesteps: (int) The total number of timesteps in training.
    :return: (function) Learning rate schedule function.
    """
    def func(progress_remaining):
        # Calculate the current timestep based on progress_remaining
        current_step = (1 - progress_remaining) * total_timesteps

        if current_step < decrease_end_iteration:
            # Linearly decrease the learning rate
            lr = initial_value - (initial_value - end_value) * (current_step / decrease_end_iteration)
        else:
            # After the decrease_end_iteration, keep the learning rate constant at end_value
            lr = end_value
        
        return lr
    
    return func


# Define a learning rate schedule function with step decay
def custom_step_decay_lr_schedule(initial_value, decay_factor, decay_interval, min_lr, total_timesteps):
    """
    Custom learning rate schedule that decreases by a factor of 'decay_factor' every 'decay_interval' steps,
    but the learning rate cannot go below 'min_lr'.
    
    :param initial_value: (float) Initial learning rate.
    :param decay_factor: (float) Factor by which to decay the learning rate every 'decay_interval' steps.
    :param decay_interval: (int) The number of steps after which the learning rate is decayed.
    :param min_lr: (float) The minimum learning rate (lower bound).
    :param total_timesteps: (int) The total number of timesteps in training.
    :return: (function) Learning rate schedule function.
    """
    def func(progress_remaining):
        # Calculate the current timestep based on progress_remaining
        current_step = (1 - progress_remaining) * total_timesteps

        # Calculate how many decay intervals have passed
        decay_steps = current_step // decay_interval

        # Calculate the decayed learning rate
        lr = initial_value / (decay_factor ** decay_steps)

        # Ensure the learning rate doesn't go below the threshold min_lr
        if lr < min_lr:
            lr = min_lr

        return lr

    return func


class EpisodeMeanLengthCallback(BaseCallback):
    """
    Custom callback for tracking the mean episode length during training.
    """
    def __init__(self):
        super().__init__()
        self.ep_mean_length = 49.

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        current_step = self.num_timesteps

        temp = []
        for i, info in enumerate(infos):
            if info["episode"] is not None:
                temp.append(info["episode"]["l"])

        if temp != []:
            self.ep_mean_length = min(np.mean(temp), self.ep_mean_length)

        return True


class MeanEpisodeLengthCallback(BaseCallback):
    """
    Custom callback for tracking the mean episode length using the built-in 'rollout/ep_len_mean' key.
    """
    def __init__(self, ref):
        super().__init__()
        self.ref = ref
        #self.ep_mean_length = 50.

    def _on_step(self) -> bool:
        # Extract the episode lengths from the buffer
        episode_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
        if not np.isnan(safe_mean(episode_lengths)):
            self.ref[0] = min(safe_mean(episode_lengths), self.ref[0])
            #self.ep_mean_length = min(safe_mean(episode_lengths), self.ep_mean_length)
        #print(f"Mean Episode Length: {self.ep_mean_length}")
        
        return True


def custom_ep_mean_length_decay_lr_schedule(callback, initial_value, min_lr, ep_mean_thresholds, ep_mean_values):
    """
    Custom learning rate schedule that decays according to episode mean length.

    :param callback: (EpisodeMeanLengthCallback) Callback to track episode mean length.
    :param initial_value: (float) Initial learning rate.
    :param decay_factor: (float) Factor by which to decay the learning rate when conditions are met.
    :param min_lr: (float) The minimum learning rate (lower bound).
    :param ep_mean_thresholds: (list) List of episode mean length thresholds to trigger decay.
    :param ep_mean_values: (list) List of learning rate divisors corresponding to each threshold.
    :return: (function) Learning rate schedule function.
    """
    
    def func(progress_remaining):
        # Get episode mean length from the callback
        ep_mean_length = callback[0]#.ep_mean_length
        
        # Default learning rate if no episode mean length available
        if ep_mean_length is None:
            return initial_value

        # Initialize learning rate
        lr = initial_value

        # Apply different decays based on episode mean length thresholds
        for i, threshold in enumerate(ep_mean_thresholds):
            if ep_mean_length < threshold:
                # Divide learning rate by corresponding value
                lr /= ep_mean_values[i]

        # Ensure the learning rate doesn't go below the threshold min_lr
        if lr < min_lr:
            lr = min_lr

        return lr
    
    return func


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
   
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)
 
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # create agent from stable baselines
    #agent_cfg["learning_rate"] = custom_lr_schedule(3e-4, 1e-5, decrease_end_iteration=3000000, total_timesteps=n_timesteps)
    #agent_cfg["learning_rate"] = custom_step_decay_lr_schedule(3e-4, 2, 250000, 1e-5, total_timesteps=n_timesteps)
    # using reference is needed, otherwise it will trigger unpickle error during checkpoint saving
    ref = [50]
    ep_mean_length_callback = MeanEpisodeLengthCallback(ref)
    agent_cfg["learning_rate"] = custom_ep_mean_length_decay_lr_schedule(ref, 1.5e-4, 1e-5, [15, 10, 5], [2, 2, 2])
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    #print(agent.policy)
    #exit()
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    # save: save_freq * num_envs 10000
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2)
    
    # Instantiate the callback
    reward_logging_callback = RewardLoggingCallback(log_dir, verbose=2)

    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=[checkpoint_callback, ep_mean_length_callback, reward_logging_callback])

    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    
    # close the simulator
    env.close()
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
