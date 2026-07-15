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
- 参考 = 站立姿态库（scripts/gen_stand_pose.py，按躯干命令 4-DOF 插值腿角）+ 脖子头部映射。
- 奖励 = 静态姿态模仿：躯干位置/朝向 + 线/角速度→0 + 腿/脖子关节角模仿 + 双脚接触
  + 正则 + 存活。去掉行走的步态时序/参考速度跟踪。
"""

from isaaclab.utils.configclass import configclass

from .oceanisaaclab_env_cfg import OCEAN_ASSET_DIR
from .oceanisaaclab_walk_env_cfg import OceanisaaclabWalkEnvCfg


@configclass
class OceanisaaclabStandEnvCfg(OceanisaaclabWalkEnvCfg):
    """BDX 论文复刻站立任务配置（Ocean-BDX-StandPaper-Direct-v0）。"""

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

    # Walking 为抑制行进时甩头使用约 1/3 的硬件适配范围。Perpetual standing 是论文的
    # 表现性姿态策略，应独立恢复 neck_head_map.npz 覆盖的完整可达域，不能继承 walking 限幅。
    head_command_dh_range = (-0.02, 0.02)
    head_command_pitch_range = (-0.5, 0.5)
    head_command_yaw_range = (-1.0, 1.0)
    head_command_roll_range = (-0.6, 0.6)

    # ------------------------------------------------------------------
    # 躯干命令 g_perp 的 (h_torso, θ_torso) 部分：4-DOF 高度 + 朝向。
    # 范围须与 gen_stand_pose.py 网格一致（超范围会被采样器钳到边界）。
    # ------------------------------------------------------------------
    # 对本机 5^4 完整网格做 IK 扫描后的可行域：最大脚位置残差 2.899 mm。
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
    # 受扰恢复奖励门控。
    # 稳态 gate=0 时严格使用上面的论文 Table I 权重；失衡后 gate 平滑升到 1，仅放松会
    # 直接阻止抬脚的双脚接触、静态腿参考与腿动作平滑约束。躯干位姿/速度、存活、力矩和
    # 关节加速度奖励始终保留，负责驱动策略通过跨步把身体带回稳定区。
    # gate 只使用 policy 已观测且真机可靠可得的 path-frame 躯干位置、IMU 姿态/角速度、
    # 状态估计水平速度和双脚接触。快速开启、慢速释放，避免摆动脚尚未落地就恢复强约束。
    # ------------------------------------------------------------------
    enable_recovery_reward_gating = True
    recovery_tilt_error_start = 0.08  # [rad]
    recovery_tilt_error_full = 0.22  # [rad]
    recovery_lin_vel_start = 0.18  # [m/s]
    recovery_lin_vel_full = 0.45  # [m/s]
    recovery_ang_vel_start = 0.45  # [rad/s]
    recovery_ang_vel_full = 1.40  # [rad/s]
    recovery_pos_error_start = 0.025  # [m], torso 相对参考支撑中心的平面偏移
    recovery_pos_error_full = 0.065  # [m]
    recovery_activation_gate = 0.10
    recovery_command_grace_s = 0.35  # 命令重采样后暂不使用瞬时姿态误差触发
    recovery_hold_s = 0.30  # 重新双脚着地后仍保持放松，允许落脚稳定
    recovery_release_s = 0.40  # 保持结束后从当前 gate 线性释放到稳态权重
    # 门控对奖励权重的作用随论文扰动课程同步从 0 放开到 1，避免随机初始策略在第一个
    # update 就关闭双脚接触/强姿态模仿，先形成稳定站立基线再学习主动捕获步。
    recovery_gate_curriculum_delay_steps = 0
    recovery_gate_curriculum_steps = 36_000
    recovery_contact_weight_scale = 0.0
    recovery_rew_w_leg_joint_pos = -3.0
    recovery_rew_w_action_rate = -0.25
    recovery_rew_w_action_acc = -0.075
    recovery_metric_gate_threshold = 0.10

    # 论文站立扰动从训练开始用 1500 iteration（24 steps/iter）线性放开。
    disturbance_curriculum_delay_steps = 0
    disturbance_curriculum_steps = 36_000
