# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文完整复刻）训练环境配置。

对照迪士尼论文《Design and Control of a Bipedal Robotic Character》(RSS 2024,
工程根目录 BD_X_paper.pdf) 的 periodic walking policy 复刻，固定脖子、只训 10 个
腿部电机：

- 观测 = 论文公式 (8) 状态 s_t + 相位二阶谐波特征 + 命令 g_peri（式 (6) 去掉头部命令）：
  path 系躯干 xy(2) + path 系 yaw sin/cos(2) + body 系线速度(3) + body 系角速度(3)
  + q(10) + q̇(10) + a_{t-1}(10) + a_{t-2}(10) + phase 谐波(4) + cmd(3) = 57 维。
- 非对称 critic（附录 A）：critic 额外收无噪声观测 + 摩擦/质量随机化系数 = 59 维。
- 奖励 = 论文表 I 腿部子集（neck 项因固定脖子删除），权重×step_dt（legged_gym 约定）。
- path frame（V-A 节 / Fig.4）：行走按命令积分、站立收敛双脚中心、最大偏差投影。
- 动作管线（V-C/V-D 节）：50Hz 策略 → 逐关节线性映射（0=标称站姿）→ 围绕实测关节角
  限幅（δ=τmax/kP）→ 一阶保持插值 + 37.5Hz 低通 → 200Hz 附录 B 执行器模型
  （软件 PD + 编码器偏移 + 摩擦 + 速度相关力矩限幅 + 背隙/噪声编码器读数 + 反射惯量）。
- 扰动 = 论文表 V 三档独立进程（幅度按整机质量 ≈10kg/15.4kg 缩放），前 1500 iter
  线性课程。
"""

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .oceanisaaclab_env_cfg import OCEAN_ASSET_DIR, OceanisaaclabEnvCfg


@configclass
class OceanisaaclabWalkEnvCfg(OceanisaaclabEnvCfg):
    """BDX 论文复刻行走任务配置（Ocean-BDX-Walk-Direct-v0）。"""

    # ------------------------------------------------------------------
    # 控制频率链（论文 V-D）：策略 50Hz，低层 200Hz（论文真机 600Hz，仿真取 4 倍抽取）
    # ------------------------------------------------------------------
    decimation = 4
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4)

    # ------------------------------------------------------------------
    # 空间维度（改布局旧 checkpoint 全部作废）
    # ------------------------------------------------------------------
    # 14 = 10 腿 + 4 脖子（脖子改为学习控制的位置目标，不再锁死）
    action_space = 14
    # 77 维：p_pf2 + yaw_pf(sin,cos)2 + lin_vel_b3 + ang_vel_b3 + q_leg10 + q_neck4
    #        + qd_leg10 + qd_neck4 + a_{t-1}14 + a_{t-2}14 + phase谐波4 + cmd3 + head_cmd4
    observation_space = 77
    # 非对称 critic：无噪声 77 维 + 摩擦系数缩放 1 + 质量缩放 1
    state_space = 79

    # ------------------------------------------------------------------
    # 参考步态库（scripts/gen_reference_gait.py 生成；含 path 系躯干轨迹与 φ̇ 表）
    # ------------------------------------------------------------------
    gait_library_path = str(OCEAN_ASSET_DIR / "gaits" / "reference_gait.npz")
    # URDF 全零姿态（BDX 标准屈膝站立）FK 站立高度，不再压低
    target_base_height = 0.385

    # ------------------------------------------------------------------
    # 脖子/头部（论文 g_peri 的头部命令 Δh_head, Δθ_head；本机脖子 4 关节实现
    # 4-DOF 头姿：Δh 头高 + pitch 点头 + yaw 摇头 + roll 歪头）
    # scripts/gen_neck_head_map.py 生成头命令→脖子参考角映射表，NeckHeadMap 四线性插值。
    # ------------------------------------------------------------------
    neck_head_map_path = str(OCEAN_ASSET_DIR / "gaits" / "neck_head_map.npz")
    # 脖子动作线性映射范围（0=默认位，±1→±range）；覆盖映射表参考角跨度，clamp 兜软限位。
    # 顺序 [n1, n2, n3, n4] = [矢状 pitch/高度对 ×2, yaw, roll]
    neck_action_joint_ranges = (0.8, 0.8, 1.2, 0.7)  # [rad]
    # 头部命令采样范围（与 neck_head_map 网格一致）。每 env reset 均匀采样，整段 episode 恒定。
    head_command_dh_range = (-0.02, 0.02)  # [m] 头高偏移（本机脖子高度权限弱，压到 ±2cm）
    head_command_pitch_range = (-0.5, 0.5)  # [rad] 点头
    head_command_yaw_range = (-1.0, 1.0)  # [rad] 摇头
    head_command_roll_range = (-0.6, 0.6)  # [rad] 歪头
    # 头命令 obs 缩放（dh 量级小，放大到与其它命令可比；obs_normalization 亦会归一）
    head_command_scale = (20.0, 1.0, 1.0, 1.0)
    # 头命令课程（07-08 复盘：脖子直接满范围进 RL 使前进步态退化——歪头/扭头=base 顶非对称
    # 质量偏置，早期把对称参考步态带偏，策略退回蹭步、torso_orient/pos/contact 全塌，见 memory
    # ocean-neck-walk-gait-regression）。头命令采样范围按 common_step_counter 从 0 线性放开到满
    # 范围，让腿步态先长熟再引入头部扰动。0→满：5000 iter × 24 steps/env = 120_000 common steps。
    head_command_curriculum_steps = 120_000

    # ------------------------------------------------------------------
    # path frame（论文 V-A / Fig.4）
    # ------------------------------------------------------------------
    path_frame_stand_time_constant = 1.0  # [s] 站立时向双脚中心收敛的一阶时间常数
    path_frame_max_pos_deviation = 0.25  # [m] path frame 与躯干的最大位置偏差（投影拉回）
    path_frame_max_yaw_deviation = 0.6  # [rad] 最大朝向偏差

    # ------------------------------------------------------------------
    # 动作变换（论文附录 A：0 → 标称关节角，1 → 每关节预期活动范围）
    # 顺序 [r1..r5, l1..l5] = [髋yaw, 髋roll, 髋pitch, 膝, 踝] × 2。
    # 参考库摆动峰值 ≈ (0.16, 0.16, 0.41, 0.50, 0.43)，范围取 ≈2 倍留平衡余量，
    # 超出软限位部分由 clamp 兜底。
    action_joint_ranges = (0.35, 0.35, 0.8, 0.9, 0.8, 0.35, 0.35, 0.8, 0.9, 0.8)  # [rad]
    # 低通滤波截止频率（论文 V-D：一阶保持插值 + 37.5Hz 低通）
    action_lowpass_cutoff_hz = 37.5

    # ------------------------------------------------------------------
    # 附录 B 执行器模型（表 VI）。⚠ 参数为 Unitree A1 / Go1 辨识值——本机电机若非
    # 同款需重新系统辨识后替换。论文 IV 节原方案髋 roll/髋 pitch/膝用 A1（34Nm），
    # 本机真机全部使用 Go1 电机（23.7Nm），故所有腿关节统一为 Go1 组。
    # 每关节参数元组顺序：(kP, kD, τmax, q̇τmax, q̇max, µs, µd, b_min, b_max,
    #                     εq_max, σq0, σq1, Im)
    actuator_params_a1 = (15.0, 0.6, 34.0, 7.4, 20.0, 0.45, 0.023, 0.005, 0.015, 0.02, 1.80e-4, 3.61e-5, 0.011)
    actuator_params_go1 = (10.0, 0.3, 23.7, 10.6, 28.8, 0.15, 0.016, 0.002, 0.005, 0.02, 1.89e-4, 5.47e-5, 0.0043)
    # 每条腿 5 关节的类型（"a1"/"go1"），顺序 [1髋yaw, 2髋roll, 3髋pitch, 4膝, 5踝]
    leg_actuator_types = ("go1", "go1", "go1", "go1", "go1")
    actuator_friction_qdot_s = 0.1  # [rad/s] 静摩擦 tanh 激活速度（论文未给，取典型值）
    actuator_backlash_tau_b = 1.0  # [Nm] 背隙 tanh 激活力矩（论文未给，取典型值）
    actuator_gain_rand_range = (0.9, 1.1)  # 每 episode kP/kD 随机化（论文：辨识区间内随机）
    actuator_armature_rand = 0.2  # 反射惯量 ±20%（论文附录 B 末段）

    # ------------------------------------------------------------------
    # 奖励（论文表 I 腿部子集；neck 项删除）。权重按 legged_gym 约定乘 step_dt(0.02)。
    # ------------------------------------------------------------------
    rew_w_torso_pos_xy = 1.0  # exp(-k·‖p_pf - p̂_pf‖²)
    # 07-08 回归论文表I原值 200（强前进模仿，核心）。历史：无脖子阶段曾放平到 30 破站立盆地
    # （k=200 时 anchor 到 0.25m 钳位处 exp≈0，前进梯度稀疏，见 memory ocean-walk-phase-0701-diverged）。
    # 但加脖子后 neck=100 + head 扰动压垮弱信号 k=30 → 退回站立（memory ocean-neck-walk-gait-regression）；
    # 论文靠 k=200 强模仿信号才扛得住 neck=100。配 rsi=0.8 缓解初期梯度稀疏，重训验证。
    rew_k_torso_pos_xy = 200.0
    rew_w_torso_orient = 1.0  # exp(-20·‖θ ⊟ θ̂‖²)
    rew_k_torso_orient = 20.0
    rew_w_lin_vel_xy = 1.0  # exp(-8·‖v_xy - v̂_xy‖²)
    rew_k_lin_vel = 8.0
    rew_w_lin_vel_z = 1.0
    rew_w_ang_vel_xy = 0.5  # exp(-2·‖ω_xy - ω̂_xy‖²)
    rew_k_ang_vel = 2.0
    rew_w_ang_vel_z = 0.5
    # 07-08 回归论文表I原值 -15（历史为破站立盆地加大到 -25 逼摆腿；论文路线靠强 torso 位置
    # 模仿而非加大 leg 逼迈步，回退到论文值）。
    rew_w_leg_joint_pos = -15.0  # -‖q - q̂‖²（负 L2，非 exp 核）
    rew_w_leg_joint_vel = -1.0e-3
    # 07-08 回归论文表I原值 1.0（历史加大到 1.5 垫低站立盆地，论文路线回退）。
    rew_w_contact_match = 1.0  # Σᵢ I[cᵢ = ĉᵢ]，每脚一致 +1
    rew_w_torque = -1.0e-3
    rew_w_joint_acc = -2.5e-6
    rew_w_action_rate = -1.5  # 腿动作率（论文表 I：leg action rate 1.5）
    rew_w_action_acc = -0.45  # 腿动作加速度（论文：leg action acc 0.45）
    # 脖子/头部模仿（论文表 I neck 项，脖子启用后恢复）：
    # 脖子关节角模仿参考头姿（NeckHeadMap(head_cmd)），论文权重 100；脖子位置驱动、
    # 轻质、与腿解耦，追踪容易，大权重不干扰腿。脖子关节速度惩罚使头姿稳定不抖。
    rew_w_neck_joint_pos = -100.0  # -‖q_neck - q̂_neck‖²（论文 neck joint positions 100）
    rew_w_neck_joint_vel = -1.0  # -‖q̇_neck‖²（论文 neck joint velocities 1.0；参考头姿静止）
    rew_w_neck_action_rate = -5.0  # 论文 neck action rate 5.0
    rew_w_neck_action_acc = -5.0  # 论文 neck action acc 5.0
    # 07-08 回归论文表I原值 20。历史降到 2 是因为 k=30 弱模仿信号下 survival 主导总回报、
    # 逼出站立盆地；现在 torso_pos_xy 回 k=200 强信号，论文范式里 survival=20 与强模仿共存
    # （前进是精确模仿副产物，不靠低 survival 逼迈步）。见 memory ocean-walk-standing-local-optimum。
    rew_w_survival = 20.0

    # ------------------------------------------------------------------
    # 终止（论文 V-B：躯干/头触地才终止；此处用等效的高度+倾角判定，无需额外接触传感）
    # ------------------------------------------------------------------
    walk_min_base_height = 0.2  # [m] 低于视为躯干触地
    walk_min_upright_projection = 0.2  # -proj_g_z 低于（倾角 >≈78°）视为倒地

    # ------------------------------------------------------------------
    # RSI（论文未用但不冲突、已验证有效）
    # 07-07 第二轮：从参考迈步中途（且在移动）起步的 env 越多，越能亲身体验
    # "追上 anchor = 拿高分"、模仿信号越密。论文重度依赖 RSI。0.5→0.8 破站立盆地。
    # ------------------------------------------------------------------
    rsi_prob = 0.8
    rsi_joint_pos_noise = 0.03  # [rad]

    # ------------------------------------------------------------------
    # 表 V 三档扰动（每 body 独立进程）。力/矩幅值已按整机质量 ≈10kg / 论文 15.4kg
    # （×0.65）缩放；短/小与长/小档幅值小，不缩放。前 1500 iter 线性课程
    # （1500 iter × 24 steps = 36_000 common steps）。
    # ------------------------------------------------------------------
    enable_paper_disturbance = True
    disturbance_curriculum_steps = 36_000
    # 短/小：髋 + 脚
    dist_small_short_bodies = ("leg_r2_link", "leg_l2_link", "leg_r5_link", "leg_l5_link")
    dist_small_short_force_xy = (0.0, 5.0)  # [N]
    dist_small_short_force_z = (0.0, 5.0)
    dist_small_short_torque = (0.0, 0.25)  # [Nm]
    dist_small_short_on_s = (0.25, 2.0)
    dist_small_short_off_s = (1.0, 3.0)
    # 长/小：盆骨（脖子固定，头档并入盆骨）
    dist_small_long_bodies = ("base_link",)
    dist_small_long_force_xy = (0.0, 5.0)
    dist_small_long_force_z = (0.0, 5.0)
    dist_small_long_torque = (0.0, 0.25)
    dist_small_long_on_s = (2.0, 10.0)
    dist_small_long_off_s = (1.0, 3.0)
    # 短/大：盆骨。论文 [90,150]N / [0,15]Nm ×0.65 质量比
    dist_large_bodies = ("base_link",)
    dist_large_force_xy = (58.0, 97.0)
    dist_large_force_z = (0.0, 6.5)
    dist_large_torque = (0.0, 9.7)
    dist_large_on_s = (0.1, 0.1)
    dist_large_off_s = (12.0, 15.0)
    # 关闭基类的单一推力机制（被表 V 扰动引擎替换）
    enable_random_push = False
    enable_push_curriculum = False

    # ------------------------------------------------------------------
    # 观测噪声/缩放（论文附录 A：输入按预期范围归一化；关节角噪声由附录 B 编码器
    # 模型提供——q̂ = q̃ + 背隙 + 速度相关高斯，不再叠加独立关节角噪声）
    # ------------------------------------------------------------------
    pos_pf_scale = 4.0  # ≈1/max_pos_deviation
    lin_vel_scale = 2.0
    noise_lin_vel = 0.05  # [m/s] 真机线速度来自状态估计器，训练时加噪声覆盖估计误差
    # ang_vel/proj_g/joint_vel 噪声沿用基类 cfg（noise_ang_vel / noise_joint_vel）

    # ------------------------------------------------------------------
    # 课程 override：模仿范式不需要命令由慢到快（RSI + 全网格参考帧，学习信号密集）
    # ------------------------------------------------------------------
    enable_command_curriculum = False

    def __post_init__(self):
        # 腿执行器改为力矩直驱（stiffness/damping 置 0，PD 由附录 B 软件执行器模型
        # 在 200Hz 内步计算并 set_joint_effort_target），力矩上限放到 Go1 峰值。
        legs = self.robot_cfg.actuators["legs"]
        legs.stiffness = 0.0
        legs.damping = 0.0
        legs.effort_limit_sim = 23.7
        legs.velocity_limit_sim = 30.0
        # 脖子：改为学习控制的位置伺服（动作后 4 维 → 脖子位置目标，追踪头部命令参考角）。
        # 保持位置驱动（ImplicitActuator，脖子轻、位置伺服稳、sim2real 简单），高刚度使其
        # 快速跟上学习目标。effort/velocity 上限适当放开以驱动脖子。
        neck = self.robot_cfg.actuators["neck"]
        neck.stiffness = 50.0
        neck.damping = 2.0
