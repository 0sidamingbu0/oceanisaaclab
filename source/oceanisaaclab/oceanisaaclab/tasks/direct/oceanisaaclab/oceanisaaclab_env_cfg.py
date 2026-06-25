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
from isaaclab.sensors import ContactSensorCfg
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

    # contact sensor on both feet (leg end links) for gait/air-time rewards
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/leg_[lr]5_link",
        history_length=3,
        track_air_time=True,
    )

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
    # - velocity command sampling (walking task)
    command_vx_range = (-0.3, 0.8)  # [m/s]  前进为主，允许少量后退
    stand_still_prob = 1.0  # 纯站立版：所有 env 零速度指令（站稳不抖优先，sim2real 调通）
    air_time_target = 0.4  # [s]  feet air time 奖励的目标腾空时长
    lin_vel_track_sigma = 0.25  # vx 跟踪奖励的 exp 带宽
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    rew_scale_upright = 2.0
    rew_scale_height = 1.0
    rew_scale_ang_vel = -0.1  # 惩罚 roll/pitch 角速度（机身平稳），站立版加大
    rew_scale_track_lin_vel = 1.5  # vx 指令跟踪（站立版即奖励速度≈0，正号）
    rew_scale_lateral = -1.0  # 惩罚非指令的侧移 vy 和自转 yaw（站立版加大防漂移）
    rew_scale_joint_pos = -0.08
    rew_scale_joint_vel = -0.02  # 站立版加大，抑制关节微动抖
    rew_scale_action_rate = -0.1  # 加大平滑电机指令（站立版，直击真机高频抖动根源）
    rew_scale_feet_air_time = 0.0  # 纯站立版关闭抬脚奖励（不鼓励迈步）
    rew_scale_feet_slide = -1.0  # 接地时惩罚水平滑移（站立版加大防漂移）
    rew_scale_stand_still = -2.0  # 站立 env 惩罚脚移动（站立版核心，加大）
    rew_scale_contact_force_rate = -3.0e-3  # 惩罚接触力跳变（抗抖核心）；实测原项~44/步，×此权重≈-0.13/步，与其他惩罚可比
    # - observation noise (gaussian std, applied on raw physical units before scaling)
    enable_obs_noise = True
    noise_ang_vel = 0.03  # [rad/s]  静置实测 ~0.003，留余量覆盖运动振动
    noise_proj_g = 0.02  # 单位向量  静置实测 ~0.0004，留余量覆盖姿态估计误差
    noise_joint_pos = 0.01  # [rad]  编码器噪声
    noise_joint_vel = 1.0  # [rad/s]  差分速度噪声大，常为腿抖主因
    # - action latency (randomized per episode, in control steps; control dt = decimation/sim_hz)
    enable_action_latency = True
    action_latency_steps = 2  # 最大延迟控制步数（0~该值间按 env 随机）
    # - reset states/conditions
    reset_joint_pos_noise = 0.02  # [rad]
    target_base_height = 0.42  # [m]
    min_base_height = 0.25  # [m]
    min_upright_projection = 0.65
    # - random lateral push disturbance
    enable_random_push = True
    push_force_range = (20.0, 40.0)  # [N]  对 ~10kg 小机器人降低推力，站立更易收敛仍保留抗扰
    push_duration_s = 0.18
    push_interval_s = (1.0, 2.0)