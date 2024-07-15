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
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from sb3_encoder import CustomCombinedExtractor

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        # this is for step reward
        #info = self.locals.get("infos")
        return True
       
    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos")
        for i, info in enumerate(infos):
            episode_info = info['episode']
            if episode_info:
                for key, value in episode_info.items():
                    if "Episode Reward" in key:
                        #print(i, key, value)
                        self.logger.record(f"Episode Reward Env {i}/{key.split('/')[1]}", value.detach().cpu().item())
        
    def _on_training_end(self) -> None:
        infos = self.locals.get("infos")
        for i, info in enumerate(infos):
            episode_info = info['episode']
            if episode_info:
                for key, value in episode_info.items():
                    if "Episode Reward" in key:
                        #print(i, key, value)
                        self.logger.record(f"Episode Reward Env {i}/{key.split('/')[1]}", value.detach().cpu().item())
               

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
 
    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)

    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    
    # Instantiate the callback
    reward_logging_callback = RewardLoggingCallback(verbose=1)

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
