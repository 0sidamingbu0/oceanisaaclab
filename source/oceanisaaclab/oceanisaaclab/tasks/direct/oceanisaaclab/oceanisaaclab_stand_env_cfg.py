# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文复刻）站立（perpetual）任务配置。

对照论文 divide-and-conquer：periodic 行走策略之外，单独训练 perpetual 站立策略
π(a | s, g_perp)，**无相位**。参考动作是双脚支撑的静态姿态，但策略在受扰恢复时允许
像论文实验所述偏离参考接触序列并迈步。命令 g_perp = (Δh_head, Δθ_head, h_torso,
θ_torso)（式 5）——躯干高度/朝向 + 头部高度/朝向。观测去掉行走策略的相位二阶谐波，
命令由行走的 (cmd3 + head4) 换成 (torso4 + head4)。

- 继承 OceanisaaclabWalkEnvCfg：复用附录 B 执行器模型、表 V 扰动、域随机化、torso 等效触地
  终止、非对称 critic、path frame、脖子位置伺服、脖子/头部命令映射。
- 观测 77 维（去 phase 谐波4；命令 torso4+head4=8）：
  p_pf2 + yaw_pf2 + projected_gravity3 + lin_vel3 + ang_vel3 + q_leg10 + q_neck4
  + qd_leg10 + qd_neck4 + a_{t-1}14 + a_{t-2}14 + g_perp8 = 77。
  critic + 摩擦 + 质量 = 79。
- 参考 = 站立姿态库（scripts/gen_stand_pose.py，head4+torso4 联合全身 CoM 平衡腿角）
  + 同源脖子头部映射。
- 奖励 = 静态姿态模仿：躯干位置/朝向 + 线/角速度→0 + 腿/脖子关节角模仿 + 双脚接触
  + 正则 + 存活。去掉行走的步态时序/参考速度跟踪。
"""

from isaaclab.utils.configclass import configclass

from .oceanisaaclab_env_cfg import OCEAN_ASSET_DIR
from .oceanisaaclab_walk_env_cfg import OceanisaaclabWalkEnvCfg


@configclass
class OceanisaaclabStandEnvCfg(OceanisaaclabWalkEnvCfg):
    """BDX 论文复刻站立任务配置（Ocean-BDX-StandPaper-Direct-v0）。"""

    # Perpetual standing must expose slow contact-preserving foot drift within one rollout.
    # The inherited 8 s horizon was too short to represent the observed 10-30 s widening.
    episode_length_s = 20.0

    # ------------------------------------------------------------------
    # 空间维度（无相位 + g_perp 8 维命令）
    # ------------------------------------------------------------------
    action_space = 14  # 同行走：10 腿 + 4 脖子
    # 77 = p_pf2 + yaw_pf2 + projected_gravity3 + lin_vel3 + ang_vel3 + q_leg10
    #      + q_neck4 + qd_leg10 + qd_neck4 + a_{t-1}14 + a_{t-2}14
    #      + torso_cmd4 + head_cmd4（无 phase）
    observation_space = 77
    state_space = 79  # + 摩擦 + 质量

    # ------------------------------------------------------------------
    # 站立姿态参考库（scripts/gen_stand_pose.py 生成）
    # ------------------------------------------------------------------
    stand_pose_path = str(OCEAN_ASSET_DIR / "gaits" / "stand_pose.npz")
    # Stable post-capture poses used only as reset states. Reward/reference targets remain
    # stand_pose.npz, so narrow and staggered stances must be corrected rather than imitated.
    stand_recovery_reset_path = str(
        OCEAN_ASSET_DIR / "gaits" / "stand_recovery_resets.npz"
    )

    # Keep a larger expressive range than walking, but leave one-third margin from the full
    # neck_head_map domain. Full-range head inertia was enough to trigger unnecessary steps.
    head_command_dh_range = (-0.013333, 0.013333)
    head_command_pitch_range = (-0.333333, 0.333333)
    head_command_yaw_range = (-0.666667, 0.666667)
    head_command_roll_range = (-0.4, 0.4)

    # ------------------------------------------------------------------
    # 躯干命令 g_perp 的 (h_torso, θ_torso) 部分：4-DOF 高度 + 朝向。
    # 范围须与 gen_stand_pose.py 网格一致（超范围会被采样器钳到边界）。
    # ------------------------------------------------------------------
    # 对本机 5^4 torso 网格做 IK 扫描后的可行域；联合 head 补偿由 stand_pose.npz 提供。
    torso_command_h_range = (-0.04, 0.01)  # [m] 躯干高度偏移（蹲下为负；升高受腿伸直限）
    torso_command_pitch_range = (-0.17, 0.17)  # [rad] 前后倾
    torso_command_yaw_range = (-0.24, 0.24)  # [rad] 原地偏航
    torso_command_roll_range = (-0.09, 0.09)  # [rad] 侧倾
    # obs 缩放：h 量级小放大到与角度可比（obs_normalization 亦会归一）
    torso_command_scale = (10.0, 1.0, 1.0, 1.0)
    # 站立命令零概率：一部分 env 命令全零（标称直立），其余均匀采样各姿态
    stand_zero_command_prob = 0.5

    # ------------------------------------------------------------------
    # 站立参考初始化（RSI）：从命令对应的站立参考姿态出生，避免第一步从 default 跳变。
    # 站立不存在"站着不动=最优"的局部最优（本来就要站稳），rsi 用来加速姿态成形。
    # ------------------------------------------------------------------
    stand_rsi_prob = 0.8
    stand_rsi_joint_pos_noise = 0.01  # [rad]，仅非零命令 RSI 使用
    # Two sevenths of the 70% no-disturbance episodes use a stable non-canonical foot
    # placement: 50% canonical clean + 20% displaced clean + 30% full Table V overall.
    stand_displaced_reset_prob_within_quiet = 2.0 / 7.0
    stand_displaced_reset_initial_scale = 0.30
    stand_displaced_reset_curriculum_steps = 36_000  # 1500 iter × 24 steps
    # 禁止父类 walking RSI 写入随机步态帧；站立环境使用上面的独立 RSI。
    rsi_prob = 0.0

    # ------------------------------------------------------------------
    # 奖励权重（静态姿态模仿）。权重×step_dt。
    # 对齐论文表 I。躯干高度命令通过腿关节参考姿态进入模仿项，不额外增加论文之外的
    # height reward；线/角速度目标改为 0，腿关节角模仿站立参考。
    # ------------------------------------------------------------------
    rew_w_torso_pos_xy = 1.0  # 跟踪离线 solved 躯干相对双脚中心的 path-frame xy
    rew_k_torso_pos_xy = 200.0
    rew_w_torso_orient = 1.0  # 躯干朝向跟命令 (pitch,yaw,roll)
    rew_k_torso_orient = 20.0
    rew_w_torso_height = 0.0  # 论文表 I 无独立躯干高度奖励
    rew_w_lin_vel_xy = 1.0  # 目标 0：惩罚水平移动
    rew_k_lin_vel = 8.0
    rew_w_lin_vel_z = 1.0
    rew_w_ang_vel_xy = 0.5  # 目标 0：惩罚躯干转动
    rew_k_ang_vel = 2.0
    rew_w_ang_vel_z = 0.5
    rew_w_leg_joint_pos = -15.0  # 腿关节角模仿站立参考（负 L2）
    rew_w_leg_joint_vel = -1.0e-3
    rew_w_contact_match = 1.0  # 双脚均着地 Σ I[c=1]（站立参考接触恒 [1,1]）
    # Contact match cannot see tangential motion. Penalize contact-foot XY speed directly so
    # maintaining double support by pushing the soles across the floor is no longer free.
    rew_w_feet_slide = -10.0
    # Together with the lost +1 contact reward, one airborne foot costs 5 reward/s. This is
    # deliberately finite: a short capture step can still pay for itself by avoiding a fall,
    # while command-induced or exploratory lifts cannot.
    rew_w_feet_airborne = -4.0
    enable_contact_match_curriculum = False  # perpetual stand 不使用行走接触课程
    rew_w_torque = -1.0e-3
    rew_w_joint_acc = -2.5e-6
    rew_w_action_rate = -1.5
    rew_w_action_acc = -0.45
    rew_w_neck_joint_pos = -100.0  # 脖子跟头部命令参考角（同行走）
    rew_w_neck_joint_vel = -1.0
    rew_w_neck_action_rate = -5.0
    rew_w_neck_action_acc = -5.0
    rew_w_survival = 20.0

    # ------------------------------------------------------------------
    # 受扰恢复诊断（不参与奖励）。
    # 论文对所有非 episodic policy 固定使用 Table I 权重，并明确指出去掉 leg imitation /
    # contact 会造成 rapid foot shuffling。因此这里只用真机可获得的 torso/IMU/接触量标记
    # 明显失稳窗口，统计捕获步；绝不根据该状态修改 contact、腿参考或动作平滑权重。
    # 进入/退出使用 Schmitt 阈值，普通晃动保持 inactive；恢复期间单脚支撑会锁存状态，
    # 双脚落地且重新稳定后再保持一小段时间，避免把同一次恢复拆成多个事件。
    # ------------------------------------------------------------------
    recovery_tilt_error_start = 0.08  # [rad]
    recovery_tilt_error_full = 0.22  # [rad]
    recovery_lin_vel_start = 0.18  # [m/s]
    recovery_lin_vel_full = 0.45  # [m/s]
    recovery_ang_vel_start = 0.45  # [rad/s]
    recovery_ang_vel_full = 1.40  # [rad/s]
    recovery_pos_error_start = 0.025  # [m], torso 相对参考支撑中心的平面偏移
    recovery_pos_error_full = 0.065  # [m]
    recovery_activation_severity = 0.35
    recovery_deactivation_severity = 0.10
    recovery_command_grace_s = 0.35  # 命令重采样后暂不使用瞬时姿态误差触发
    recovery_hold_s = 0.30
    recovery_capture_min_air_time_s = 0.12
    recovery_capture_min_step_distance_m = 0.03
    # Canonical stance completion is diagnostic-only and does not change Table I rewards.
    stance_recovery_width_tolerance_m = 0.008
    stance_recovery_stagger_tolerance_m = 0.010
    stance_recovery_yaw_tolerance_rad = 0.05
    stance_recovery_projected_gravity_xy_max = 0.08
    stance_recovery_lin_vel_xy_max = 0.08
    stance_recovery_ang_vel_xy_max = 0.20
    stance_recovery_joint_vel_rms_max = 0.25
    stance_recovery_stable_hold_s = 0.30

    # 论文站立扰动从训练开始用 1500 iteration（24 steps/iter）线性放开。
    disturbance_curriculum_delay_steps = 0
    disturbance_curriculum_steps = 36_000

    # 训练分布：保留一部分完全无外力的 nominal episode，给策略稳定双支撑和锁脚
    # 样本；其余 episode 使用完整 Table V 独立扰动过程。该采样不改变奖励，也不
    # 改变扰动 episode 内的力/矩/开关时序，只避免小扰动几乎覆盖所有样本而诱发
    # “无扰动也挪脚”的局部解。真机部署不使用该标志。
    stand_disturbance_quiet_prob = 0.70
