# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Ocean-BDX-Stand-Direct-v0",
    entry_point=f"{__name__}.oceanisaaclab_env:OceanisaaclabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.oceanisaaclab_env_cfg:OceanisaaclabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Ocean-BDX-Walk-Direct-v0",
    entry_point=f"{__name__}.oceanisaaclab_walk_env:OceanisaaclabWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.oceanisaaclab_walk_env_cfg:OceanisaaclabWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkPPORunnerCfg",
    },
)

gym.register(
    id="Ocean-BDX-WalkRough-Direct-v0",
    entry_point=f"{__name__}.oceanisaaclab_walk_env:OceanisaaclabWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.oceanisaaclab_walk_env_cfg:OceanisaaclabWalkRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkRoughPPORunnerCfg",
    },
)

gym.register(
    id="Ocean-BDX-StandPaper-Direct-v0",
    entry_point=f"{__name__}.oceanisaaclab_stand_env:OceanisaaclabStandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.oceanisaaclab_stand_env_cfg:OceanisaaclabStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandPPORunnerCfg",
    },
)

gym.register(
    id="Template-Oceanisaaclab-Direct-v0",
    entry_point=f"{__name__}.oceanisaaclab_env:OceanisaaclabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.oceanisaaclab_env_cfg:OceanisaaclabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
