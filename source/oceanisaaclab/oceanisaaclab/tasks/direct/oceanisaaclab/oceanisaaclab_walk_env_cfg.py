# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文适配复刻）训练环境配置。

对照迪士尼论文《Design and Control of a Bipedal Robotic Character》(RSS 2024,
工程根目录 BD_X_paper.pdf) 的 periodic walking policy 复刻，学习控制 10 个腿部和
4 个脖子电机：

- 观测 = 论文公式 (8) 状态 s_t + 相位二阶谐波特征 + 完整命令 g_peri：
  path 系躯干 xy(2) + path 系 yaw sin/cos(2) + body 系线速度(3) + body 系角速度(3)
  + 腿/脖子 q、q̇ + 14 维前两步动作 + phase 谐波(4) + cmd(3) + head cmd(4) = 80 维。
- 非对称 critic（附录 A）：critic 额外收无噪声观测 + 摩擦/质量随机化系数 = 82 维。
- 奖励 = 论文表 I 的腿部与 neck 项，权重×step_dt（legged_gym 约定）。
- path frame（V-A 节 / Fig.4）：行走按命令积分、站立收敛双脚中心、最大偏差投影。
- 动作管线（V-C/V-D 节）：50Hz 策略 → 逐关节线性映射（0=标称站姿）→ 围绕实测关节角
  限幅（δ=τmax/kP）→ 一阶保持插值 + 37.5Hz 低通 → 200Hz 附录 B 执行器模型
  （软件 PD + 编码器偏移 + 摩擦 + 速度相关力矩限幅 + 背隙/噪声编码器读数 + 反射惯量）。
- 扰动 = 论文表 V 三档独立进程（幅度按整机质量 ≈10kg/15.4kg 缩放），从训练开始
  并在前 1500 iteration 线性放开。
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from .oceanisaaclab_env_cfg import OCEAN_ASSET_DIR, OceanisaaclabEnvCfg


WALK_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=10.0,
    num_rows=8,
    num_cols=16,
    curriculum=False,
    horizontal_scale=0.05,
    vertical_scale=0.0025,
    slope_threshold=0.75,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.40),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(-0.012, 0.012),
            noise_step=0.004,
            border_width=0.25,
        ),
        # Isaac Lab's height-field implementation interprets slope as rise/run. 0.0875
        # corresponds to atan(0.0875) ~= 5 degrees. A 2 m center platform matches
        # TerrainImporter's spawn-height search window; walking outward then enters a
        # 1 m sustained uphill/downhill segment without a reset drop.
        "up_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.0875),
            platform_width=2.0,
            border_width=0.25,
        ),
        "down_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.0875),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class OceanisaaclabWalkEnvCfg(OceanisaaclabEnvCfg):
    """BDX 论文复刻行走任务配置（Ocean-BDX-Walk-Direct-v0）。"""

    # ------------------------------------------------------------------
    # 控制频率链（论文 V-D）：策略 50Hz，低层 200Hz（论文真机 600Hz，仿真取 4 倍抽取）
    # ------------------------------------------------------------------
    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        # Flat pre-training has far fewer contact patches than rough-terrain training.
        # These capacities leave headroom for early-policy falls at 8192 environments.
        physics=PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**20,
        ),
    )

    # Restore the paper's parallel rollout shape for the inexpensive flat pre-training stage.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=2.0, replicate_physics=True
    )

    # ------------------------------------------------------------------
    # 空间维度（改布局旧 checkpoint 全部作废）
    # ------------------------------------------------------------------
    # 14 = 10 腿 + 4 脖子（脖子改为学习控制的位置目标，不再锁死）
    action_space = 14
    # 80 维：p_pf2 + yaw_pf(sin,cos)2 + projected_gravity3 + lin_vel_b3 + ang_vel_b3 + q_leg10 + q_neck4
    #        + qd_leg10 + qd_neck4 + a_{t-1}14 + a_{t-2}14 + phase谐波4 + cmd3 + head_cmd4
    observation_space = 80
    # 非对称 critic：无噪声 80 维 + 摩擦系数缩放 1 + 质量缩放 1
    state_space = 82
    gait_duty_factor = 0.6

    # Learn the nominal periodic gait on a PhysX plane first. Rough terrain is exposed by
    # OceanisaaclabWalkRoughEnvCfg after the policy no longer falls on almost every reset.
    # This keeps the paper's terrain randomization while avoiding all-body mesh contacts
    # during the expensive initial exploration stage.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ------------------------------------------------------------------
    # 参考步态库（scripts/gen_reference_gait.py 生成；含 path 系躯干轨迹与 φ̇ 表）
    # ------------------------------------------------------------------
    gait_library_path = str(OCEAN_ASSET_DIR / "gaits" / "reference_gait.npz")
    # URDF 全零姿态（BDX 标准屈膝站立）FK 站立高度，不再压低
    target_base_height = 0.385
    # Full-speed torso lean reduced from 5 to 3 degrees. The learned gait already tracks
    # vx=0.15 accurately; the larger reference lean reduces the sim2sim forward-fall margin.
    walk_lean_angle = 0.052

    # ------------------------------------------------------------------
    # 脖子/头部（论文 g_peri 的头部命令 Δh_head, Δθ_head；本机脖子 4 关节实现
    # 4-DOF 头姿：Δh 头高 + pitch 点头 + yaw 摇头 + roll 歪头）
    # scripts/gen_neck_head_map.py 生成头命令→脖子参考角映射表，NeckHeadMap 四线性插值。
    # ------------------------------------------------------------------
    neck_head_map_path = str(OCEAN_ASSET_DIR / "gaits" / "neck_head_map.npz")
    # 脖子动作线性映射范围（0=默认位，±1→±range）；覆盖映射表参考角跨度，clamp 兜软限位。
    # 顺序 [n1, n2, n3, n4] = [矢状 pitch/高度对 ×2, yaw, roll]
    neck_action_joint_ranges = (0.8, 0.8, 1.2, 0.7)  # [rad]
    # model_1400 remains stable while the old curriculum is at 20%, but neck tracking loss
    # grows rapidly as the four simultaneous commands approach their old limits. Keep the
    # paper's 4-DOF command interface and map, with a hardware-adapted range near one third.
    head_command_dh_range = (-0.007, 0.007)  # [m] 头高偏移
    head_command_pitch_range = (-0.17, 0.17)  # [rad] 点头
    head_command_yaw_range = (-0.33, 0.33)  # [rad] 摇头
    head_command_roll_range = (-0.20, 0.20)  # [rad] 歪头
    # 头命令 obs 缩放（dh 量级小，放大到与其它命令可比；obs_normalization 亦会归一）
    head_command_scale = (20.0, 1.0, 1.0, 1.0)
    # Keep command transitions in training, but allow the 1rad/s neck servos to settle before
    # another independent four-axis target is sampled.
    control_resample_interval_s = (4.0, 8.0)

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
    # 奖励（论文表 I 的腿部与 neck 项 + 硬件适配启动课程）。
    # 权重按 legged_gym 约定乘 step_dt(0.02)。
    # ------------------------------------------------------------------
    rew_w_torso_pos_xy = 1.0  # exp(-k·‖p_pf - p̂_pf‖²)
    # Hardware-adapted static reward balance. At the 0.25m path-frame clamp, the paper
    # k=200 kernel is effectively zero and provides no recovery gradient to a stationary
    # policy. k=30 keeps exp(-30*0.25^2)=0.153, so moving environments can still discover
    # the reference trajectory without a time-varying bootstrap reward.
    rew_k_torso_pos_xy = 30.0
    rew_w_torso_orient = 1.0  # exp(-20·‖θ ⊟ θ̂‖²)
    rew_k_torso_orient = 20.0
    rew_w_lin_vel_xy = 1.0  # exp(-8·‖v_xy - v̂_xy‖²)
    rew_k_lin_vel = 8.0
    rew_w_lin_vel_z = 1.0
    rew_w_ang_vel_xy = 0.5  # exp(-2·‖ω_xy - ω̂_xy‖²)
    rew_k_ang_vel = 2.0
    rew_w_ang_vel_z = 0.5
    # Keep the proven no-neck tracking weight. Stronger early tracking made the policy
    # chase an open-loop reference that only survives about 0.7 s instead of learning
    # stabilizing corrections around it.
    rew_w_leg_joint_pos = -25.0  # -‖q - q̂‖²（负 L2，非 exp 核）
    rew_w_leg_joint_pos_initial = -25.0
    rew_w_leg_joint_vel = -1.0e-3
    # Contact imitation is implemented as -Σ I[c_i != c_ref_i]. This is the same existing
    # contact term with its action-independent positive baseline removed: correct contact
    # gets 0, while a foot that stays planted during reference swing is explicitly costly.
    # A weight of 3 forced model_22100 to follow the swing schedule before it could balance:
    # every episode still fell after about one second. Keep explicit contact imitation, but
    # leave enough reward margin for stabilizing corrections around the reference motion.
    rew_w_contact_match = 2.0
    # Restore normal action smoothness over roughly the first 1000 flat-training iterations.
    # Head commands have an independent curriculum below so they cannot destabilize this stage.
    enable_contact_match_curriculum = True
    rew_w_contact_match_initial = 2.0
    contact_match_anneal_steps = 24_000
    # Filled automatically by the training entry point when resuming a checkpoint. Despite
    # the legacy name, this offset is shared by reward, head-command and disturbance curricula.
    contact_match_curriculum_step_offset = 0
    rew_w_torque = -1.0e-3
    rew_w_joint_acc = -2.5e-6
    rew_w_action_rate = -1.5  # 腿动作率（论文表 I：leg action rate 1.5）
    rew_w_action_rate_initial = -0.5
    rew_w_action_acc = -0.45  # 腿动作加速度（论文：leg action acc 0.45）
    rew_w_action_acc_initial = -0.15
    # Paper Table I values. The previous reduced weights allowed zero-command n3/n4
    # motion around 0.04-0.05rad RMS instead of tracking their zero reference.
    rew_w_neck_joint_pos = -100.0
    rew_w_neck_joint_vel = -1.0
    rew_w_neck_action_rate = -5.0
    rew_w_neck_action_acc = -5.0
    # Survival=2 did not outweigh aggressive reference/contact tracking: model_22100 had a
    # 100% fall rate for the entire run. Five restores a useful fall cost without returning
    # to the old value of 20 that encouraged permanent double support.
    rew_w_survival = 5.0

    # Learn the leg gait first. Keep head targets at zero for about 1000 iterations, then
    # smoothly expose the hardware-adapted head-command range over the following 2000 iterations.
    head_command_curriculum_delay_steps = 24_000
    head_command_curriculum_ramp_steps = 48_000

    # ------------------------------------------------------------------
    # 终止（论文 V-B）：躯干或头部触地。当前 nested URDF 的 PhysX contact
    # view 在大规模场景中存在漏报/错误聚合，因此用 base collision 高度与倾角
    # 做等效判定。base mesh 最低点相对 root 约 -0.185m，0.20m 覆盖 ±12mm roughness。
    # ------------------------------------------------------------------
    walk_min_base_height = 0.20  # [m], root height relative to the local ground below the pelvis
    walk_min_upright_projection = 0.20  # -projected_gravity_z; about 78.5 degrees tilt

    # ------------------------------------------------------------------
    # RSI：从参考迈步中途起步，提高周期状态覆盖并避免只探索站立盆地。
    # ------------------------------------------------------------------
    rsi_prob = 0.8
    rsi_joint_pos_noise = 0.03  # [rad]
    # Sample reflected inertia once per parallel environment. With 8192 environments this
    # still covers the randomized dynamics distribution, without a PhysX property write on
    # every high-frequency early-policy reset.
    randomize_armature_each_reset = False

    # ------------------------------------------------------------------
    # 表 V 三档扰动（每 body 独立进程）。力/矩幅值已按整机质量 ≈10kg / 论文 15.4kg
    # （×0.65）缩放；短/小与长/小档幅值小，不缩放。按论文附录 A，从训练开始并在
    # 前 1500 iter 线性放开完整扰动；1500 iter 只用于预览 nominal behavior，最终仍训 100k。
    # ------------------------------------------------------------------
    enable_paper_disturbance = True
    disturbance_curriculum_delay_steps = 0
    disturbance_curriculum_steps = 36_000
    # 短/小：髋 + 脚
    dist_small_short_bodies = ("leg_r2_link", "leg_l2_link", "leg_r5_link", "leg_l5_link")
    dist_small_short_force_xy = (0.0, 5.0)  # [N]
    dist_small_short_force_z = (0.0, 5.0)
    dist_small_short_torque = (0.0, 0.25)  # [Nm]
    dist_small_short_on_s = (0.25, 2.0)
    dist_small_short_off_s = (1.0, 3.0)
    # 长/小：盆骨 + 头部，各 body 独立进程。
    dist_small_long_bodies = ("base_link", "neck_n4_link")
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
        # 只为两只脚手动添加 contact-report API；避免 URDF importer 默认给 base
        # 添加不再需要的 report，也不创建不可靠的 torso/head nested views。
        self.robot_cfg.spawn.activate_contact_sensors = False
        # 当前 URDF 的凸包在名义姿态存在内部重叠；启用全局 self-collision 会让头部
        # 持续承受约 3.4 kN 的伪接触力。保留关节限位并使用高度/倾角终止。
        self.robot_cfg.spawn.self_collision = False
        self.robot_cfg.spawn.articulation_props.enabled_self_collisions = False


@configclass
class OceanisaaclabWalkRoughEnvCfg(OceanisaaclabWalkEnvCfg):
    """Rough and mild-slope fine-tuning after the flat walking policy is stable."""

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics=PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**22,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )
    # One downward ray follows the pelvis XY position. It is a training-only termination/
    # diagnostic signal and is deliberately excluded from policy and critic observations.
    terrain_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
        mesh_prim_paths=["/World/ground"],
        max_distance=3.0,
        debug_vis=False,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WALK_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # Rough/slope runner uses 96 control steps per iteration; preserve the paper's 1500-iteration ramp.
    disturbance_curriculum_delay_steps = 0
    disturbance_curriculum_steps = 144_000
    # Rough terrain is a fine-tuning stage entered after the flat walking curriculum.
    enable_contact_match_curriculum = False
