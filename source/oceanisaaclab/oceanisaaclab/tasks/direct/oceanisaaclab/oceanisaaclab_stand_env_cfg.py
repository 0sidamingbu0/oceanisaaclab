# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文复刻）站立（perpetual）任务配置。

对照论文 divide-and-conquer：periodic 行走策略之外，单独训练 perpetual 站立策略
π(a | s, g_perp)，**无相位**、脚不迈步。命令 g_perp = (Δh_head, Δθ_head, h_torso,
θ_torso)（式 5）——躯干高度/朝向 + 头部高度/朝向。观测去掉行走策略的相位二阶谐波，
命令由行走的 (cmd3 + head4) 换成 (torso4 + head4)。

- 继承 OceanisaaclabWalkEnvCfg：复用附录 B 执行器模型、表 V 扰动、域随机化、torso 触地
  终止、非对称 critic、path frame、脖子位置伺服、脖子/头部命令映射。
- 观测 74 维（去 phase 谐波4；命令 torso4+head4=8）：
  p_pf2 + yaw_pf2 + lin_vel3 + ang_vel3 + q_leg10 + q_neck4 + qd_leg10 + qd_neck4
  + a_{t-1}14 + a_{t-2}14 + g_perp8 = 74。critic + 摩擦 + 质量 = 76。
- 参考 = 站立姿态库（scripts/gen_stand_pose.py，按躯干命令 4-DOF 插值腿角）+ 脖子头部映射。
- 奖励 = 静态姿态模仿：躯干位置/朝向/高度 + 线/角速度→0 + 腿/脖子关节角模仿 + 双脚接触
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
    # 74 = p_pf2 + yaw_pf2 + lin_vel3 + ang_vel3 + q_leg10 + q_neck4 + qd_leg10
    #      + qd_neck4 + a_{t-1}14 + a_{t-2}14 + torso_cmd4 + head_cmd4（无 phase）
    observation_space = 74
    state_space = 76  # + 摩擦 + 质量

    # ------------------------------------------------------------------
    # 站立姿态参考库（scripts/gen_stand_pose.py 生成）
    # ------------------------------------------------------------------
    stand_pose_path = str(OCEAN_ASSET_DIR / "gaits" / "stand_pose.npz")

    # ------------------------------------------------------------------
    # 躯干命令 g_perp 的 (h_torso, θ_torso) 部分：4-DOF 高度 + 朝向。
    # 范围须与 gen_stand_pose.py 网格一致（超范围会被采样器钳到边界）。
    # ------------------------------------------------------------------
    torso_command_h_range = (-0.05, 0.02)  # [m] 躯干高度偏移（蹲下为负；升高受腿伸直限）
    torso_command_pitch_range = (-0.25, 0.25)  # [rad] 前后倾
    torso_command_yaw_range = (-0.35, 0.35)  # [rad] 原地偏航
    torso_command_roll_range = (-0.18, 0.18)  # [rad] 侧倾
    # obs 缩放：h 量级小放大到与角度可比（obs_normalization 亦会归一）
    torso_command_scale = (10.0, 1.0, 1.0, 1.0)
    # 站立命令零概率：一部分 env 命令全零（标称直立），其余均匀采样各姿态
    stand_zero_command_prob = 0.1

    # ------------------------------------------------------------------
    # 站立参考初始化（RSI）：从命令对应的站立参考姿态出生，避免第一步从 default 跳变。
    # 站立不存在"站着不动=最优"的局部最优（本来就要站稳），rsi 用来加速姿态成形。
    # ------------------------------------------------------------------
    stand_rsi_prob = 0.8

    # ------------------------------------------------------------------
    # 奖励权重（静态姿态模仿）。权重×step_dt。
    # 复用行走的公共项，改：加大躯干朝向权重（站立核心=摆姿态）、新增躯干高度模仿、
    # 线/角速度目标改为 0（惩罚移动）、腿关节角模仿站立参考（-15 即可，无需 -25 逼迈步）。
    # ------------------------------------------------------------------
    rew_w_torso_pos_xy = 1.0  # 躯干在双脚中心上方（pos_pf → 参考 sway=0）
    rew_k_torso_pos_xy = 30.0
    rew_w_torso_orient = 2.0  # 躯干朝向跟命令 (pitch,yaw,roll)；站立核心项，加大
    rew_k_torso_orient = 20.0
    rew_w_torso_height = 2.0  # 躯干高度跟命令 base_height + h_torso（站立新增项）
    rew_k_torso_height = 200.0  # exp(-200·Δh²)：3cm 误差 exp≈0.16
    rew_w_lin_vel_xy = 1.0  # 目标 0：惩罚水平移动
    rew_k_lin_vel = 8.0
    rew_w_lin_vel_z = 1.0
    rew_w_ang_vel_xy = 0.5  # 目标 0：惩罚躯干转动
    rew_k_ang_vel = 2.0
    rew_w_ang_vel_z = 0.5
    rew_w_leg_joint_pos = -15.0  # 腿关节角模仿站立参考（负 L2）
    rew_w_leg_joint_vel = -1.0e-3
    rew_w_contact_match = 1.0  # 双脚均着地 Σ I[c=1]（站立参考接触恒 [1,1]）
    rew_w_torque = -1.0e-3
    rew_w_joint_acc = -2.5e-6
    rew_w_action_rate = -1.5
    rew_w_action_acc = -0.45
    rew_w_neck_joint_pos = -100.0  # 脖子跟头部命令参考角（同行走）
    rew_w_neck_joint_vel = -1.0
    rew_w_neck_action_rate = -5.0
    rew_w_neck_action_acc = -5.0
    rew_w_survival = 2.0
