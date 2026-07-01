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
    observation_space = 41
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
    command_scale = (2.0, 2.0, 1.0)
    # - velocity command sampling (walking task)
    command_vx_range = (0.10, 0.25)  # [m/s]  低速前进范围（课程终点）；上限压到 0.25，先把能走稳的速度练扎实（实测 0.3 会后仰摔），以后再抬
    command_wz_range = (-0.8, 0.8)  # [rad/s]  yaw 命令范围；sim2sim 左右旋转必须在训练分布内
    stand_still_prob = 0.15  # 保留少量零速站立样本，削弱站立吸引子占比，逼大多数 env 学行走/转向
    turn_in_place_prob = 0.25  # 非站立样本中一部分设 vx=0，只训练原地转向迈步
    move_command_threshold = 0.08  # [m/s or rad/s]  低于该阈值按站立命令处理
    gait_cycle_period = 0.6  # [s]  左右脚完整步态周期
    gait_duty_factor = 0.5  # 单脚期望接触占空比；=0.5 明确交替，双支撑窗口最小
    air_time_target = 0.15  # [s]  行走目标腾空时长（匹配 0.6s 周期/duty 0.5 的实际摆动时长）
    foot_clearance_target = 0.05  # [m]  摆动脚脚底目标离地间隙（已按 foot_origin_offset 校正度量）
    lin_vel_track_sigma = 0.04  # vx 跟踪奖励的 exp 带宽；再收紧，原地踏步(vx≈0)能骗到的跟踪奖励更低
    ang_vel_track_sigma = 0.25  # yaw rate 跟踪奖励的 exp 带宽
    # - 脚 link 原点到脚底的结构偏移：站立接触态 leg_[lr]5_link 原点实测离地约 0.067m。
    #   feet_clearance 用 body_pos_w[feet,z]-env_origin_z 度量的是 link 原点高度，需减去该偏移
    #   才是真正的脚底离地间隙，否则 clamp(target-height) 恒为 0（旧版 feet_clearance 全程失效）。
    foot_origin_offset = 0.067  # [m]
    # - 命令幅度课程：训练早期把 vx 命令上限压窄（先让步态成形），按 common_step 线性放开到 command_vx_range。
    enable_command_curriculum = True
    command_vx_range_start = (0.10, 0.18)  # [m/s]  课程起点：一开始只学慢速走稳，别一上来就要求 0.3
    command_curriculum_steps = 120_000  # common_step 数（≈5000 iter×24）内从起点线性放开到终点；放慢，每个速度档有时间巩固
    # - push-recovery via stepping (instability-gated, IMU-observable signals only)
    #   失衡度 = sigmoid 组合躯干倾斜 |proj_g_xy| 与倾倒角速度 |ang_vel_xy|（真机 IMU 均可得）；
    #   稳态≈0 时迈步惩罚全开（钉地不抖），失衡≈1 时解除迈步惩罚并奖励有效迈步。
    instability_tilt_thresh = 0.15  # |proj_g_xy| 阈值（约 8.6°）：超过视为开始失衡
    instability_tilt_rate_thresh = 1.0  # |ang_vel_xy| 阈值 [rad/s]：倾倒角速度
    instability_sharpness = 8.0  # sigmoid 陡度（越大越接近硬阈值）
    # - reward scales
    rew_scale_alive = 0.5  # 降低站立类正奖励占比，避免站着不动主导 return
    rew_scale_terminated = -5.0
    rew_scale_upright = 1.0  # 降低站立类正奖励占比
    rew_scale_height = 0.5  # 降低站立类正奖励占比
    rew_scale_ang_vel = -0.06  # 惩罚 roll/pitch 角速度（机身平稳）；行走时调弱，避免压制自然躯干摆动
    rew_scale_track_lin_vel = 3.0  # vx 指令跟踪（行走主驱动），提权
    rew_scale_track_ang_vel = 1.2  # yaw 命令跟踪（原地转向/左右旋转主驱动）
    rew_scale_progress = 1.5  # 线性前进奖励 min(vx,cmd)/cmd：给 vx≈0 附近持续的"往前走"梯度（exp 项此处太平），只对前进命令生效
    rew_scale_lateral = -0.8  # 惩罚非指令的侧移 vy，保留一定恢复余量
    rew_scale_joint_pos = -0.08
    rew_scale_joint_vel = -0.01  # 行走阶段放松关节速度，避免把迈步压成小幅抖动
    rew_scale_action_rate = -0.05  # 行走阶段放松动作变化率；抗抖主要交给零速/接触项
    rew_scale_feet_air_time = 3.0  # 行走时奖励有效迈步；短腾空不再倒扣（clamp min=0）。降低落地尖峰方差
    rew_scale_feet_slide = -0.5  # 接地滑移惩罚；前期减弱避免压制正常摆动落地微滑
    rew_scale_stand_still = -2.0  # 仅零速且稳定时钉脚；非零速度命令必须允许迈步
    rew_scale_phase_contact = 1.0  # 支撑相按 gait clock 奖励接触（前进命令下按 fwd_gate 折减，杜绝原地踏步白拿）
    rew_scale_swing_contact = -1.5  # 摆动相仍接触的硬惩罚（逼出"该抬就抬"的交替）
    rew_scale_single_support = 0.6  # 非零速度命令下奖励单脚支撑（前进命令下按 fwd_gate 折减）
    rew_scale_no_fly = -0.6  # 避免双脚同时离地跳步
    rew_scale_feet_clearance = 0.6  # 摆动脚脚底接近 clearance target 的正奖励（前进命令下按 fwd_gate 折减）
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
    push_force_range = (8.0, 20.0)  # [N]  从头学步态先轻推，避免初期只学摔倒/硬撑
    push_duration_s = 0.18
    push_interval_s = (1.0, 2.0)
    # - push curriculum: ramp push magnitude up over training to force stepping recovery
    enable_push_curriculum = True
    push_force_range_max = (20.0, 40.0)  # [N]  课程终点：对 10kg 机身，0.18s 下 Δv≈0.36~0.72m/s，仍有挑战但可迈步恢复（70N=Δv1.26 太大直接踹飞）
    # common_step_counter 每个控制步 +1（不乘 num_envs！）→ 1 iter = num_steps_per_env(24) 步。
    # 旧值 250_000_000 误按乘了 num_envs(4096) 估，导致 7999 iter 时 frac≈0.0008，推力全程卡在起点
    # (20,40)N 从未逼出迈步。60_000 ≈ 2500 iter 跑满课程。
    # 分阶段：推力课程延迟到 push_curriculum_start_step 才开始爬，之前保持地板值 push_force_range=(8,20)N，
    # 让策略先专心学走稳（命令课程 iter~5000 走满），走稳后再叠加抗推，避免"还没会走就被大推力砸"。
    push_curriculum_start_step = 120_000  # common_step（≈5000 iter×24）：命令课程走满后推力才开始往上爬
    push_curriculum_steps = 120_000  # 在多少 common_step 内从起点线性升到终点（≈5000 iter）；从 start_step 起算
