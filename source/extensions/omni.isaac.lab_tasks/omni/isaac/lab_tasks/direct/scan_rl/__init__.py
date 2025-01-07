# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .single_drone_env import SCANRLEnv, SCANRLEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Scan-RL",
    entry_point="omni.isaac.lab_tasks.direct.scan_rl:SCANRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SCANRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ddpg_cfg.yaml",
    },
)
