# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
#parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from sb3_encoder import CustomCombinedExtractor

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
            if i>2:
                break
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

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
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
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    #agent = DDPG(policy_arch, env, verbose=1, **agent_cfg)
    #print(agent.policy)
    #exit()
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    # save: save_freq * num_envs
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2)
    
    # Instantiate the callback
    reward_logging_callback = RewardLoggingCallback(log_dir, verbose=2)

    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=[checkpoint_callback, reward_logging_callback])

    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    
    # close the simulator
    env.close()
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
