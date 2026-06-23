# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


OCEANISAACLAB_EXT_DIR = Path(__file__).resolve().parents[4]
OCEAN_ASSET_DIR = OCEANISAACLAB_EXT_DIR / "assets"
OCEAN_URDF_PATH = OCEAN_ASSET_DIR / "urdf" / "ocean.urdf"


@configclass
class OceanisaaclabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 8.0
    # - spaces definition
    action_space = 10
    observation_space = 39
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(OCEAN_URDF_PATH),
            fix_base=False,
            merge_fixed_joints=False,
            activate_contact_sensors=True,
            self_collision=False,
            collision_type="Convex Hull",
            ros_package_paths=[{"name": "ocean_description", "path": str(OCEAN_ASSET_DIR)}],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["leg_[lr][1-5]_joint"],
                effort_limit_sim=23.0,
                velocity_limit_sim=6.0,
                stiffness=50.0,
                damping=2.5,
            ),
            "neck": ImplicitActuatorCfg(
                joint_names_expr=["neck_n[1-4]_joint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=1.0,
                stiffness=8.0,
                damping=0.5,
            ),
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joints
    leg_joint_names = [
        "leg_r1_joint",
        "leg_r2_joint",
        "leg_r3_joint",
        "leg_r4_joint",
        "leg_r5_joint",
        "leg_l1_joint",
        "leg_l2_joint",
        "leg_l3_joint",
        "leg_l4_joint",
        "leg_l5_joint",
    ]
    neck_joint_names = ["neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint"]
    # - action scale
    action_scale = 0.25  # [rad]
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    command_scale = (2.0, 2.0, 0.25)
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    rew_scale_upright = 2.0
    rew_scale_height = 1.0
    rew_scale_ang_vel = -0.05
    rew_scale_joint_pos = -0.4
    rew_scale_joint_vel = -0.01
    rew_scale_action_rate = -0.02
    # - reset states/conditions
    reset_joint_pos_noise = 0.02  # [rad]
    target_base_height = 0.42  # [m]
    min_base_height = 0.25  # [m]
    min_upright_projection = 0.65
    # - random lateral push disturbance
    enable_random_push = True
    push_force_range = (5.0, 15.0)  # [N]
    push_duration_s = 0.10
    push_interval_s = (2.0, 4.0)