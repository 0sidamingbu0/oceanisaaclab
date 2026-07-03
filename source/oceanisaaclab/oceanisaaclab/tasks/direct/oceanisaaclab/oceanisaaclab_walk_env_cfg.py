# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 式参考轨迹模仿）训练环境配置。

对照迪士尼 BDX 复刻（Open Duck / AWD go_bdx）的范式：奖励以「参考步态逐帧匹配」为
主导（关节角 L2 匹配 + 接触时序匹配 + 基座姿态/高度/速度匹配），彻底删除路线 A 的
fwd_gate / instability gate / phase_contact / swing_contact / single_support /
feet_clearance / feet_air_time / stand_still 手工塑形联动栈——这些打地鼠问题由参考
轨迹一次性根治。观测布局与路线 A 完全一致（41 维，无足底接触量），sim2sim 部署链路不变。
"""

from isaaclab.utils.configclass import configclass

from .oceanisaaclab_env_cfg import OCEAN_ASSET_DIR, OceanisaaclabEnvCfg


@configclass
class OceanisaaclabWalkEnvCfg(OceanisaaclabEnvCfg):
    """参考模仿行走任务配置（Ocean-BDX-Walk-Direct-v0）。"""

    # 参考步态库（scripts/gen_reference_gait.py 生成，placo IK + meshcat 可视化；
    # 改库参数需同步重新生成，并可用 scripts/play_reference_gait.py 播放确认）
    gait_library_path = str(OCEAN_ASSET_DIR / "gaits" / "reference_gait.npz")

    # 参考步态基座高度 = URDF 全零姿态（BDX 标准屈膝站立，腿部有伸直角度预留）的
    # FK 站立高度 ≈0.385m。不额外下蹲——伸直余量留给摆动/迈步。
    target_base_height = 0.385

    # 动作幅度：target = default_leg(0) + action_scale * action，action 天然 ~[-1,1]。
    # 参考步态摆动关节角峰值：膝 leg_[lr]4≈0.41rad、踝 leg_[lr]5≈0.31rad、全库 abs max≈0.5rad。
    # 基类的 0.25 只能达 ±0.25rad → 膝/踝够不到参考摆动 → 策略只能匹配小幅髋摆(身体扭)、
    # 抬脚饱和抬不起来（实测「完全不抬脚」的根因）。放大到 0.5 让参考幅度落在 action∈[-1,1] 内
    # （膝峰值 0.41→action 0.82，留余量给平衡修正）。仅路线 B override，不动路线 A(浅步态 0.25 够用)。
    # ⚠ 改动作语义，须从头重训；sim2sim/onnx 部署侧 action_scale 也要同步为 0.5。
    action_scale = 0.5

    # - 模仿奖励（正奖励为主，DeepMimic/AWD 风格 exp 核）
    rew_scale_imit_joint_pos = 3.0  # 关节角匹配：主导项
    imit_joint_pos_sigma = 0.4  # exp(-Σ误差²/σ)；10 关节合计均方差 ~0.04 rad² 时 ≈0.9
    rew_scale_imit_joint_vel = 0.3  # 关节速度匹配：低权重（有限差分参考速度较噪）
    imit_joint_vel_sigma = 40.0
    rew_scale_imit_contact = 1.0  # 双脚接触时序匹配（0/1 参考 schedule）
    rew_scale_imit_height = 0.5  # 基座高度匹配
    imit_height_sigma = 2.5e-3  # exp(-Δh²/σ)；Δh=0.05m 时 ≈0.37
    rew_scale_imit_orient = 1.0  # 基座姿态匹配（proj_g 对参考前倾姿态）
    imit_orient_sigma = 0.05
    # - 速度命令跟踪（与参考 base 系速度比较；权重低于关节匹配，避免拖着参考跑）
    rew_scale_walk_track_lin_vel = 1.5
    rew_scale_walk_track_ang_vel = 0.75
    walk_lin_vel_track_sigma = 0.04
    walk_ang_vel_track_sigma = 0.25
    # - 正则项（保留少量平滑/防滑，其余交给参考匹配）
    rew_scale_walk_alive = 0.25
    rew_scale_walk_action_rate = -0.05
    rew_scale_walk_feet_slide = -0.2
    # - 参考态初始化（RSI，DeepMimic 关键技巧）：该比例的 env 直接从参考帧的关节角/
    #   基座姿态/速度出发，episode 从步态中段开始，绕过「从静止起步」这一最难阶段
    rsi_prob = 0.9
    rsi_joint_pos_noise = 0.03  # [rad] RSI 关节角附加噪声

    # - 课程 override（仅路线 B）
    # 命令幅度课程关掉：模仿范式不需要「由慢到快」的爬坡。RSI（90% env 从随机相位参考帧
    # 出发）已绕过「从静止起步」这一最难阶段；参考库对全命令网格 vx∈{-0.25..0.25} 都有
    # 逐帧参考、零命令点即站立帧、插值天然给出「命令→0 步幅→0」，任何速度下学习信号都
    # 密集且良定义。开着反而前期把 vx 压在起点、高速参考帧长期见不到，拖慢高速段收敛。
    enable_command_curriculum = False
    # 推力课程保留（抗扰鲁棒性模仿一样需要）：但不再有命令课程走满作为锚点，且 RSI 收敛快，
    # 把起爬点从 120_000 提前到 60_000（≈2500 iter），让抗推早点介入。
    push_curriculum_start_step = 60_000
