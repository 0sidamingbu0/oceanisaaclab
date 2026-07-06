# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文完整复刻）训练环境。

对照迪士尼论文（工程根目录 BD_X_paper.pdf）的 periodic walking policy，固定脖子、
只训 10 个腿部电机。相对基类（路线 A）替换了全部核心环节：

1. **path frame**（V-A 节）：行走按命令积分、站立收敛双脚中心、最大偏差投影；
   躯干 path 系位姿进观测与奖励（速度跟踪由 path 系位置模仿隐式实现）。
2. **观测**（公式 (8) + 附录 A）：path 系躯干 xy/yaw + body 系线/角速度 + q/q̇ +
   前两步动作 + 相位二阶谐波 + 命令，57 维；非对称 critic 额外收无噪声观测 +
   摩擦/质量随机化系数（59 维）。
3. **奖励**（表 I 腿部子集）：躯干位姿/速度 exp 核 + 腿关节负 L2 + 接触匹配 +
   力矩/加速度/动作率/动作加速度正则 + 存活，权重×step_dt。
4. **动作管线**（V-C/V-D + 附录 A）：逐关节线性映射（0=标称站姿）→ 围绕实测
   关节角 ±τmax/kP 限幅 → 一阶保持插值 + 37.5Hz 低通 → 200Hz 执行器模型。
5. **执行器模型**（附录 B / 表 VI）：软件 PD（q̃=q+εq 编码器偏移）+ tanh 摩擦 +
   速度相关力矩限幅 + 背隙/速度相关噪声编码器读数 + 反射惯量随机化，全部
   每 episode 重采样。
6. **扰动**（表 V）：三档独立进程（髋/脚短小扰动、盆骨长小扰动、盆骨短大推力），
   前 1500 iter 线性课程。
7. **相位**：φ̇ 从参考库按命令插值（步频随速度变化），逐步积分。
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude, quat_from_euler_xyz, sample_uniform

from .oceanisaaclab_env import OceanisaaclabEnv
from .oceanisaaclab_walk_env_cfg import OceanisaaclabWalkEnvCfg
from .path_frame import PathFrame
from .reference_gait import ReferenceGait


class _DisturbanceSchedule:
    """表 V 的一档扰动进程：每 (env, body) 独立采样力/力矩与开/关时长。"""

    def __init__(
        self,
        num_envs: int,
        body_ids: list[int],
        force_xy: tuple[float, float],
        force_z: tuple[float, float],
        torque: tuple[float, float],
        on_s: tuple[float, float],
        off_s: tuple[float, float],
        device: torch.device,
    ):
        self.body_ids = body_ids
        self.force_xy = force_xy
        self.force_z = force_z
        self.torque = torque
        self.on_s = on_s
        self.off_s = off_s
        n_b = len(body_ids)
        self.forces = torch.zeros(num_envs, n_b, 3, device=device)
        self.torques = torch.zeros(num_envs, n_b, 3, device=device)
        self.on_left = torch.zeros(num_envs, n_b, device=device)
        self.off_left = torch.zeros(num_envs, n_b, device=device)

    @staticmethod
    def _signed_uniform(lo: float, hi: float, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        mag = sample_uniform(lo, hi, shape, device)
        sign = torch.where(torch.rand(shape, device=device) < 0.5, -1.0, 1.0)
        return mag * sign

    def step(self, dt: float, scale: float) -> None:
        self.on_left = torch.clamp(self.on_left - dt, min=0.0)
        self.off_left = torch.clamp(self.off_left - dt, min=0.0)
        device = self.forces.device
        # "on" 结束 → 力清零，抽下一次 "off" 间隔
        ended = (self.on_left <= 0.0) & (self.forces.abs().sum(dim=-1) > 0.0)
        if torch.any(ended):
            self.forces[ended] = 0.0
            self.torques[ended] = 0.0
            self.off_left[ended] = sample_uniform(
                self.off_s[0], self.off_s[1], (int(ended.sum()),), device
            )
        # "off" 结束 → 抽新扰动（逐维独立采样，随机符号；课程缩放幅值）
        start = (self.on_left <= 0.0) & (self.off_left <= 0.0)
        if torch.any(start):
            n = int(start.sum())
            f = torch.zeros(n, 3, device=device)
            f[:, 0] = self._signed_uniform(*self.force_xy, (n,), device)
            f[:, 1] = self._signed_uniform(*self.force_xy, (n,), device)
            f[:, 2] = self._signed_uniform(*self.force_z, (n,), device)
            t = self._signed_uniform(*self.torque, (n, 3), device)
            self.forces[start] = f * scale
            self.torques[start] = t * scale
            self.on_left[start] = sample_uniform(self.on_s[0], self.on_s[1], (n,), device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.forces[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.on_left[env_ids] = 0.0
        self.off_left[env_ids] = sample_uniform(
            self.off_s[0], self.off_s[1], (len(env_ids), self.off_left.shape[1]), self.forces.device
        )


class OceanisaaclabWalkEnv(OceanisaaclabEnv):
    cfg: OceanisaaclabWalkEnvCfg

    def __init__(self, cfg: OceanisaaclabWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._reference_gait = ReferenceGait(self.cfg.gait_library_path, self.device)
        if abs(self._reference_gait.gait_period - self.cfg.gait_cycle_period) > 1e-6:
            raise RuntimeError(
                f"gait library period {self._reference_gait.gait_period} != "
                f"cfg.gait_cycle_period {self.cfg.gait_cycle_period}; regenerate the library."
            )

        # ---- path frame（+x = 头部前向）与朝向约定 ----
        self._path_frame = PathFrame(
            self.num_envs,
            self.device,
            stand_time_constant=self.cfg.path_frame_stand_time_constant,
            max_pos_deviation=self.cfg.path_frame_max_pos_deviation,
            max_yaw_deviation=self.cfg.path_frame_max_yaw_deviation,
        )
        # head_yaw = base_yaw + offset（URDF base_link +x 指尾部 → offset = π）
        self._head_yaw_offset = 0.0 if self.cfg.forward_vx_sign > 0.0 else torch.pi

        # ---- 相位：φ̇ 按命令从库插值、逐步积分（论文：步频随命令变化） ----
        self._phase = torch.rand(self.num_envs, device=self.device)

        # ---- 动作管线缓冲（前两步动作历史 / FOH / 低通） ----
        self._prev_prev_actions = torch.zeros_like(self._actions)
        self._joint_ranges = torch.tensor(
            self.cfg.action_joint_ranges, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self._target_joint_pos = self._default_leg_joint_pos.clone()
        self._prev_target_joint_pos = self._default_leg_joint_pos.clone()
        self._filtered_joint_target = self._default_leg_joint_pos.clone()
        # 低通系数：y += α(u−y)，α = 1 − exp(−2π·f_c·dt_sim)
        self._lowpass_alpha = 1.0 - math.exp(
            -2.0 * math.pi * self.cfg.action_lowpass_cutoff_hz * self.physics_dt
        )
        self._substep = 0

        # ---- 附录 B 执行器模型：每关节参数（表 VI，[r1..r5,l1..l5]） ----
        params = {"a1": self.cfg.actuator_params_a1, "go1": self.cfg.actuator_params_go1}
        per_joint = [params[t] for t in self.cfg.leg_actuator_types] * 2
        cols = list(zip(*per_joint))

        def joint_param(idx: int) -> torch.Tensor:
            return torch.tensor(cols[idx], dtype=torch.float, device=self.device).unsqueeze(0)

        self._act_kp = joint_param(0)
        self._act_kd = joint_param(1)
        self._act_tau_max = joint_param(2)
        self._act_qd_tau_max = joint_param(3)
        self._act_qd_max = joint_param(4)
        self._act_mu_s = joint_param(5)
        self._act_mu_d = joint_param(6)
        self._act_b_min = joint_param(7)
        self._act_b_max = joint_param(8)
        self._act_eps_max = joint_param(9)
        self._act_sigma0 = joint_param(10)
        self._act_sigma1 = joint_param(11)
        self._act_armature = joint_param(12)
        # 动作 setpoint 围绕实测关节角的限幅：δ = τmax/kP（保证最大力矩仍可产生）
        self._setpoint_max_dev = self._act_tau_max / self._act_kp
        # 每 episode 随机量
        self._encoder_offset = torch.zeros(self.num_envs, 10, device=self.device)
        self._backlash = torch.zeros(self.num_envs, 10, device=self.device)
        self._gain_scale_kp = torch.ones(self.num_envs, 10, device=self.device)
        self._gain_scale_kd = torch.ones(self.num_envs, 10, device=self.device)
        self._last_tau_m = torch.zeros(self.num_envs, 10, device=self.device)
        self._applied_leg_torque = torch.zeros(self.num_envs, 10, device=self.device)

        # ---- 表 V 扰动进程 ----
        self._disturbances: list[_DisturbanceSchedule] = []
        if self.cfg.enable_paper_disturbance:
            groups = [
                (
                    self.cfg.dist_small_short_bodies,
                    self.cfg.dist_small_short_force_xy,
                    self.cfg.dist_small_short_force_z,
                    self.cfg.dist_small_short_torque,
                    self.cfg.dist_small_short_on_s,
                    self.cfg.dist_small_short_off_s,
                ),
                (
                    self.cfg.dist_small_long_bodies,
                    self.cfg.dist_small_long_force_xy,
                    self.cfg.dist_small_long_force_z,
                    self.cfg.dist_small_long_torque,
                    self.cfg.dist_small_long_on_s,
                    self.cfg.dist_small_long_off_s,
                ),
                (
                    self.cfg.dist_large_bodies,
                    self.cfg.dist_large_force_xy,
                    self.cfg.dist_large_force_z,
                    self.cfg.dist_large_torque,
                    self.cfg.dist_large_on_s,
                    self.cfg.dist_large_off_s,
                ),
            ]
            for bodies, f_xy, f_z, tq, on_s, off_s in groups:
                body_ids, _ = self.robot.find_bodies(list(bodies), preserve_order=True)
                self._disturbances.append(
                    _DisturbanceSchedule(
                        self.num_envs, body_ids, f_xy, f_z, tq, on_s, off_s, self.device
                    )
                )
            self._dist_body_ids = [bid for sched in self._disturbances for bid in sched.body_ids]

        # ---- 非对称 critic 特权量（基类 DR 未开时回退到 1） ----
        if not hasattr(self, "_dr_mass_scale"):
            self._dr_mass_scale = torch.ones(self.num_envs, device=self.device)
        if not hasattr(self, "_dr_friction_scale"):
            self._dr_friction_scale = torch.ones(self.num_envs, device=self.device)

        # 覆盖基类奖励分项统计（表 I 分项集合）
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "torso_pos_xy",
                "torso_orient",
                "lin_vel_xy",
                "lin_vel_z",
                "ang_vel_xy",
                "ang_vel_z",
                "leg_joint_pos",
                "leg_joint_vel",
                "contact_match",
                "torque",
                "joint_acc",
                "action_rate",
                "action_acc",
                "survival",
            ]
        }

        # 反射惯量（armature）：先写标称值，reset 时每 episode ±20% 随机
        self._write_armature(torch.arange(self.num_envs, device=self.device), randomize=False)
        # path frame 初始化到出生位姿
        self._reset_path_frame(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _head_yaw(self) -> torch.Tensor:
        """躯干头部前向朝向（世界系 yaw）。"""
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w.torch)
        return yaw + self._head_yaw_offset

    def _moving_mask(self) -> torch.Tensor:
        return (
            torch.max(torch.abs(self._commands[:, :2]), dim=1).values > self.cfg.move_command_threshold
        ) | (torch.abs(self._commands[:, 2]) > self.cfg.move_command_threshold)

    def _gait_phase(self) -> torch.Tensor:
        return self._phase

    def _write_armature(self, env_ids: torch.Tensor, randomize: bool) -> None:
        """写腿关节反射惯量 I_m（附录 B 末段：每 episode ±armature_rand 随机）。"""
        armature = self._act_armature.expand(len(env_ids), -1)
        if randomize:
            r = self.cfg.actuator_armature_rand
            armature = armature * sample_uniform(1.0 - r, 1.0 + r, armature.shape, self.device)
        joint_ids = torch.tensor(self._leg_dof_idx, dtype=torch.int32, device=self.device)
        writer = getattr(self.robot, "write_joint_armature_to_sim_index", None) or getattr(
            self.robot, "write_joint_armature_to_sim", None
        )
        if writer is None:
            if not hasattr(self, "_warned_no_armature"):
                import omni.log

                omni.log.warn("walk env: articulation has no armature writer API; reflected inertia not applied.")
                self._warned_no_armature = True
            return
        try:
            writer(armature.contiguous(), joint_ids=joint_ids, env_ids=env_ids.to(dtype=torch.int32))
        except TypeError:
            writer(armature.contiguous(), joint_ids=self._leg_dof_idx, env_ids=env_ids)

    def _reset_path_frame(self, env_ids: torch.Tensor, pf_offset: torch.Tensor | None = None) -> None:
        """path frame 初始化到躯干出生位姿。reset 时朝向为 default（yaw=0 → head_yaw=offset）。

        pf_offset: (len(env_ids), 2) 可选——RSI 用，参考帧的 path 系躯干偏移，
        path frame 原点 = 躯干位置 − R(yaw)·offset，使躯干一出生就位于参考的 sway 相位上。
        注意：不读 root_pos_w（reset 写入后 data buffer 可能滞后），直接用刚写入的
        env_origins + default 位姿重算。
        """
        base_xy = (
            self.scene.env_origins[env_ids, :2] + self.robot.data.default_root_pose.torch[env_ids, :2]
        ).clone()
        head_yaw = torch.full((len(env_ids),), self._head_yaw_offset, device=self.device)
        if pf_offset is not None:
            cos_y, sin_y = torch.cos(head_yaw), torch.sin(head_yaw)
            base_xy[:, 0] -= pf_offset[:, 0] * cos_y - pf_offset[:, 1] * sin_y
            base_xy[:, 1] -= pf_offset[:, 0] * sin_y + pf_offset[:, 1] * cos_y
        self._path_frame.reset(env_ids, base_xy, head_yaw)

    # ------------------------------------------------------------------
    # actions: per-joint mapping + FOH + low-pass + appendix-B actuator model
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clamp(-1.0, 1.0)
        # 随机动作延迟（保留基类机制：sim2real 通信延迟）
        if self.cfg.enable_action_latency:
            self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
            self._action_history[:, 0] = self._actions
            delayed_actions = torch.gather(
                self._action_history,
                1,
                self._action_delay.view(-1, 1, 1).expand(-1, 1, self.cfg.action_space),
            ).squeeze(1)
        else:
            delayed_actions = self._actions
        # 逐关节线性映射：0 → 标称关节角（全零屈膝站姿），1 → 每关节预期活动范围
        desired = self._default_leg_joint_pos + self._joint_ranges * delayed_actions
        # 围绕实测关节角限幅（δ = τmax/kP，保证最大力矩可达，防 setpoint 飞出）
        q_meas = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        desired = torch.clamp(desired, q_meas - self._setpoint_max_dev, q_meas + self._setpoint_max_dev)
        desired = torch.clamp(
            desired,
            self._soft_leg_joint_pos_limits[:, :, 0],
            self._soft_leg_joint_pos_limits[:, :, 1],
        )
        self._prev_target_joint_pos = self._target_joint_pos
        self._target_joint_pos = desired
        self._processed_actions = desired
        self._substep = 0
        # 相位积分：φ̇ 按命令插值（零命令参考为常量站立帧，相位照走与论文一致）
        phase_rate = self._reference_gait.sample_phase_rate(self._commands)
        self._phase = torch.remainder(self._phase + phase_rate * self.step_dt, 1.0)
        # 表 V 扰动进程
        self._update_paper_disturbances()

    def _apply_action(self) -> None:
        """每个 200Hz 物理子步调用：FOH 插值 + 低通 + 附录 B 执行器模型 → 关节力矩。"""
        self._substep += 1
        frac = min(1.0, self._substep / self.cfg.decimation)
        u = self._prev_target_joint_pos + frac * (self._target_joint_pos - self._prev_target_joint_pos)
        self._filtered_joint_target = self._filtered_joint_target + self._lowpass_alpha * (
            u - self._filtered_joint_target
        )
        q = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        qd = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        q_tilde = q + self._encoder_offset  # 编码器偏移 εq（式 15）
        # PD 电机力矩（式 14）：τ_m = kP(a − q̃) − kD·q̇（增益每 episode 随机缩放）
        kp = self._act_kp * self._gain_scale_kp
        kd = self._act_kd * self._gain_scale_kd
        tau_m = kp * (self._filtered_joint_target - q_tilde) - kd * qd
        self._last_tau_m = tau_m
        # 速度相关力矩限幅（式 18）：恒定 τmax 到 q̇τmax，线性降到 q̇max 归零；
        # 反向（制动）恒 τmax
        ramp_hi = torch.clamp(
            self._act_tau_max * (self._act_qd_max - qd) / (self._act_qd_max - self._act_qd_tau_max),
            min=torch.zeros_like(qd),
            max=self._act_tau_max.expand_as(qd),
        )
        tau_hi = torch.where(qd <= self._act_qd_tau_max, self._act_tau_max.expand_as(qd), ramp_hi)
        ramp_lo = torch.clamp(
            self._act_tau_max * (self._act_qd_max + qd) / (self._act_qd_max - self._act_qd_tau_max),
            min=torch.zeros_like(qd),
            max=self._act_tau_max.expand_as(qd),
        )
        tau_lo = -torch.where(-qd <= self._act_qd_tau_max, self._act_tau_max.expand_as(qd), ramp_lo)
        # 关节摩擦（式 17）
        tau_f = self._act_mu_s * torch.tanh(qd / self.cfg.actuator_friction_qdot_s) + self._act_mu_d * qd
        tau = torch.clamp(tau_m, tau_lo, tau_hi) - tau_f
        self._applied_leg_torque = tau
        if hasattr(self.robot, "set_joint_effort_target_index"):
            self.robot.set_joint_effort_target_index(target=tau, joint_ids=self._leg_dof_idx)
        else:
            self.robot.set_joint_effort_target(tau, joint_ids=self._leg_dof_idx)
        # 脖子固定：高刚度位置驱动锁死默认位
        self.robot.set_joint_position_target_index(target=self._default_neck_joint_pos, joint_ids=self._neck_dof_idx)

    def _update_paper_disturbances(self) -> None:
        if not self._disturbances:
            return
        scale = min(1.0, self.common_step_counter / float(self.cfg.disturbance_curriculum_steps))
        for sched in self._disturbances:
            sched.step(self.step_dt, scale)
        forces = torch.cat([s.forces for s in self._disturbances], dim=1)
        torques = torch.cat([s.torques for s in self._disturbances], dim=1)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces, torques, body_ids=self._dist_body_ids, is_global=True
        )

    # ------------------------------------------------------------------
    # observations: paper eq. (8) + phase harmonics + command; asymmetric critic
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        head_yaw = self._head_yaw()
        pos_pf, yaw_pf = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        # 相位特征（附录 A）：前两阶谐波 sin/cos(k·2πφ), k∈{1,2}
        two_pi_phase = 2.0 * torch.pi * self._phase
        phase_feat = torch.stack(
            (
                torch.sin(two_pi_phase),
                torch.cos(two_pi_phase),
                torch.sin(2.0 * two_pi_phase),
                torch.cos(2.0 * two_pi_phase),
            ),
            dim=1,
        )
        yaw_feat = torch.stack((torch.sin(yaw_pf), torch.cos(yaw_pf)), dim=1)

        def assemble(
            pos_pf_t: torch.Tensor,
            yaw_feat_t: torch.Tensor,
            lin_vel_t: torch.Tensor,
            ang_vel_t: torch.Tensor,
            joint_pos_t: torch.Tensor,
            joint_vel_t: torch.Tensor,
        ) -> torch.Tensor:
            return torch.cat(
                (
                    pos_pf_t * self.cfg.pos_pf_scale,
                    yaw_feat_t,
                    lin_vel_t * self.cfg.lin_vel_scale,
                    ang_vel_t * self.cfg.ang_vel_scale,
                    joint_pos_t * self.cfg.dof_pos_scale,
                    joint_vel_t * self.cfg.dof_vel_scale,
                    self._previous_actions,
                    self._prev_prev_actions,
                    phase_feat,
                    self._commands * self._command_scale,
                ),
                dim=-1,
            )

        # critic：无噪声真值 + 特权信息（附录 A：simulation state without noise + friction）
        critic_obs = torch.cat(
            (
                assemble(pos_pf, yaw_feat, lin_vel, ang_vel, joint_pos, joint_vel),
                self._dr_friction_scale.unsqueeze(1),
                self._dr_mass_scale.unsqueeze(1),
            ),
            dim=-1,
        )
        # policy：附录 B 编码器读数模型 q̂ = q̃ + 0.5·b·tanh(τm/τb) + N(0, σq0+σq1|q̇|)
        if self.cfg.enable_obs_noise:
            sigma = self._act_sigma0 + self._act_sigma1 * torch.abs(joint_vel)
            joint_pos_hat = (
                joint_pos
                + self._encoder_offset
                + 0.5 * self._backlash * torch.tanh(self._last_tau_m / self.cfg.actuator_backlash_tau_b)
                + torch.randn_like(joint_pos) * sigma
            )
            lin_vel_n = lin_vel + torch.randn_like(lin_vel) * self.cfg.noise_lin_vel
            ang_vel_n = ang_vel + torch.randn_like(ang_vel) * self.cfg.noise_ang_vel
            joint_vel_n = joint_vel + torch.randn_like(joint_vel) * self.cfg.noise_joint_vel
            obs = assemble(pos_pf, yaw_feat, lin_vel_n, ang_vel_n, joint_pos_hat, joint_vel_n)
        else:
            obs = critic_obs[:, : self.cfg.observation_space]
        return {"policy": obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # dones: paper V-B — terminate only when the torso hits the ground
    # (equivalent height/tilt test; feet-only contact sensing keeps hardware simple)
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # path frame 动力学每控制步推进一次（物理步进后、奖励/观测前）
        feet_center = self.robot.data.body_pos_w.torch[:, self._feet_body_ids, :2].mean(dim=1)
        self._path_frame.step(
            self.step_dt,
            self._commands,
            self._moving_mask(),
            self.robot.data.root_pos_w.torch[:, :2],
            self._head_yaw(),
            feet_center,
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        torso_down = base_height < self.cfg.walk_min_base_height
        not_upright = -self.robot.data.projected_gravity_b.torch[:, 2] < self.cfg.walk_min_upright_projection
        return torso_down | not_upright, time_out

    # ------------------------------------------------------------------
    # rewards: paper Table I (leg subset), weights × step_dt
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        ref = self._reference_gait.sample(self._commands, self._phase)

        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        head_yaw = self._head_yaw()
        pos_pf, _ = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        joint_acc = self.robot.data.joint_acc.torch[:, self._leg_dof_idx]
        in_contact = (
            self.contact_sensor.data.current_contact_time.torch[:, self._feet_contact_ids] > 0.0
        ).float()

        # 1) 躯干 path 系 xy 位置模仿（速度跟踪由此隐式实现：path frame 按命令前进）
        pos_err = torch.sum(torch.square(pos_pf - ref["base_pos_pf"]), dim=1)
        rew_pos_xy = cfg.rew_w_torso_pos_xy * torch.exp(-cfg.rew_k_torso_pos_xy * pos_err)
        # 2) 躯干朝向模仿：参考四元数 = path 朝向 + 参考 yaw 振荡 + 参考俯仰（前倾）
        base_yaw_ref = self._path_frame.yaw + ref["base_yaw_pf"] - self._head_yaw_offset
        zeros = torch.zeros_like(base_yaw_ref)
        quat_ref = quat_from_euler_xyz(zeros, ref["base_pitch"], base_yaw_ref)
        orient_err = quat_error_magnitude(self.robot.data.root_quat_w.torch, quat_ref)
        rew_orient = cfg.rew_w_torso_orient * torch.exp(-cfg.rew_k_torso_orient * torch.square(orient_err))
        # 3) body 系线速度 xy / z
        lin_err_xy = torch.sum(torch.square(lin_vel[:, :2] - ref["lin_vel_b"][:, :2]), dim=1)
        rew_lin_xy = cfg.rew_w_lin_vel_xy * torch.exp(-cfg.rew_k_lin_vel * lin_err_xy)
        lin_err_z = torch.square(lin_vel[:, 2] - ref["lin_vel_b"][:, 2])
        rew_lin_z = cfg.rew_w_lin_vel_z * torch.exp(-cfg.rew_k_lin_vel * lin_err_z)
        # 4) body 系角速度 xy / z
        ang_err_xy = torch.sum(torch.square(ang_vel[:, :2] - ref["ang_vel_b"][:, :2]), dim=1)
        rew_ang_xy = cfg.rew_w_ang_vel_xy * torch.exp(-cfg.rew_k_ang_vel * ang_err_xy)
        ang_err_z = torch.square(ang_vel[:, 2] - ref["ang_vel_b"][:, 2])
        rew_ang_z = cfg.rew_w_ang_vel_z * torch.exp(-cfg.rew_k_ang_vel * ang_err_z)
        # 5) 腿关节角 / 角速度（负 L2，论文原样）
        rew_joint_pos = cfg.rew_w_leg_joint_pos * torch.sum(torch.square(joint_pos - ref["joint_pos"]), dim=1)
        rew_joint_vel = cfg.rew_w_leg_joint_vel * torch.sum(torch.square(joint_vel - ref["joint_vel"]), dim=1)
        # 6) 接触匹配：Σᵢ I[cᵢ = ĉᵢ]
        ref_contact = (ref["feet_contact"] >= 0.5).float()
        rew_contact = cfg.rew_w_contact_match * torch.sum((in_contact == ref_contact).float(), dim=1)
        # 7) 正则：力矩 / 关节加速度 / 动作率 / 动作加速度
        rew_torque = cfg.rew_w_torque * torch.sum(torch.square(self._applied_leg_torque), dim=1)
        rew_joint_acc = cfg.rew_w_joint_acc * torch.sum(torch.square(joint_acc), dim=1)
        rew_action_rate = cfg.rew_w_action_rate * torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        rew_action_acc = cfg.rew_w_action_acc * torch.sum(
            torch.square(self._actions - 2.0 * self._previous_actions + self._prev_prev_actions), dim=1
        )
        # 8) 存活
        rew_survival = cfg.rew_w_survival * (1.0 - self.reset_terminated.float())

        reward_terms = {
            "torso_pos_xy": rew_pos_xy,
            "torso_orient": rew_orient,
            "lin_vel_xy": rew_lin_xy,
            "lin_vel_z": rew_lin_z,
            "ang_vel_xy": rew_ang_xy,
            "ang_vel_z": rew_ang_z,
            "leg_joint_pos": rew_joint_pos,
            "leg_joint_vel": rew_joint_vel,
            "contact_match": rew_contact,
            "torque": rew_torque,
            "joint_acc": rew_joint_acc,
            "action_rate": rew_action_rate,
            "action_acc": rew_action_acc,
            "survival": rew_survival,
        }
        # 表 I 权重按 legged_gym/Isaac Gym 约定乘 step_dt（论文同一代码谱系）
        total_reward = torch.stack(list(reward_terms.values()), dim=0).sum(dim=0) * self.step_dt
        for key, value in reward_terms.items():
            self._episode_sums[key] += value * self.step_dt
        return total_reward

    # ------------------------------------------------------------------
    # reset: RSI + per-episode actuator-model randomization + path frame init
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        # 基类完成命令采样、默认状态写入与统计上报
        super()._reset_idx(env_ids)
        if not hasattr(self, "_phase"):
            return  # __init__ 里父类构造期间的首次全量 reset：本类缓冲尚未建立

        n = len(env_ids)
        # 相位随机偏置
        self._phase[env_ids] = torch.rand(n, device=self.device)
        # 动作管线缓冲复位
        self._prev_prev_actions[env_ids] = 0.0
        self._target_joint_pos[env_ids] = self._default_leg_joint_pos[env_ids]
        self._prev_target_joint_pos[env_ids] = self._default_leg_joint_pos[env_ids]
        self._filtered_joint_target[env_ids] = self._default_leg_joint_pos[env_ids]
        self._last_tau_m[env_ids] = 0.0
        self._applied_leg_torque[env_ids] = 0.0
        # 附录 B 每 episode 随机量：编码器偏移 / 背隙 / PD 增益 / 反射惯量
        self._encoder_offset[env_ids] = (
            torch.rand(n, 10, device=self.device) * 2.0 - 1.0
        ) * self._act_eps_max
        self._backlash[env_ids] = self._act_b_min + (self._act_b_max - self._act_b_min) * torch.rand(
            n, 10, device=self.device
        )
        lo, hi = self.cfg.actuator_gain_rand_range
        self._gain_scale_kp[env_ids] = sample_uniform(lo, hi, (n, 10), self.device)
        self._gain_scale_kd[env_ids] = sample_uniform(lo, hi, (n, 10), self.device)
        self._write_armature(env_ids, randomize=True)
        # 表 V 扰动进程复位
        for sched in self._disturbances:
            sched.reset(env_ids)

        # ---- RSI：部分 env 从随机相位参考帧出发 ----
        pf_offset = None
        rsi = torch.rand(n, device=self.device) < self.cfg.rsi_prob
        if torch.any(rsi):
            rsi_ids = env_ids[rsi]
            ref = self._reference_gait.sample(self._commands[rsi_ids], self._phase[rsi_ids])

            joint_pos = self.robot.data.default_joint_pos.torch[rsi_ids].clone()
            leg_pos = ref["joint_pos"] + torch.empty_like(ref["joint_pos"]).uniform_(
                -self.cfg.rsi_joint_pos_noise, self.cfg.rsi_joint_pos_noise
            )
            soft_limits = self.robot.data.soft_joint_pos_limits.torch[rsi_ids][:, self._leg_dof_idx]
            joint_pos[:, self._leg_dof_idx] = torch.clamp(leg_pos, soft_limits[..., 0], soft_limits[..., 1])
            joint_vel = self.robot.data.default_joint_vel.torch[rsi_ids].clone()
            joint_vel[:, self._leg_dof_idx] = ref["joint_vel"]

            # 基座位姿：参考高度；朝向沿用 default（正立、head_yaw=offset）。
            # 不手写四元数（历史 bug：该管线 root_pose 四元数存储顺序与手写假设不一致）。
            root_pose = self.robot.data.default_root_pose.torch[rsi_ids].clone()
            root_pose[:, :3] = self.scene.env_origins[rsi_ids]
            root_pose[:, 2] += ref["base_height"]
            root_vel = self.robot.data.default_root_vel.torch[rsi_ids].clone()
            root_vel[:, :3] = ref["lin_vel_b"]
            root_vel[:, 3:5] = 0.0
            root_vel[:, 5] = self._commands[rsi_ids, 2]

            self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=rsi_ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=rsi_ids)
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=rsi_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=rsi_ids)
            # RSI 的动作管线缓冲对齐参考关节角（避免第一步 setpoint 从 0 跳变）
            self._target_joint_pos[rsi_ids] = joint_pos[:, self._leg_dof_idx]
            self._prev_target_joint_pos[rsi_ids] = joint_pos[:, self._leg_dof_idx]
            self._filtered_joint_target[rsi_ids] = joint_pos[:, self._leg_dof_idx]
            # path frame 原点偏移，使躯干出生即位于参考 sway 相位上
            pf_offset = torch.zeros(n, 2, device=self.device)
            pf_offset[rsi] = ref["base_pos_pf"]
        self._reset_path_frame(env_ids, pf_offset)
