# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文适配复刻）训练环境。

对照迪士尼论文（工程根目录 BD_X_paper.pdf）的 periodic walking policy，学习控制
10 个腿部和 4 个脖子电机。相对基类（路线 A）替换了全部核心环节：

1. **path frame**（V-A 节）：行走按命令积分、站立收敛双脚中心、最大偏差投影；
   躯干 path 系位姿进观测与奖励（速度跟踪由 path 系位置模仿隐式实现）。
2. **观测**（公式 (8) + 附录 A）：path 系躯干 xy/yaw + body 系线/角速度 + q/q̇ +
   前两步动作 + 相位二阶谐波 + 行走/头部命令，80 维；非对称 critic 额外收
   无噪声观测 + 摩擦/质量随机化系数（82 维）。
3. **奖励**（表 I）：躯干位姿/速度 exp 核 + 腿/脖子关节模仿 + 接触匹配 +
   力矩/加速度/动作率/动作加速度正则 + 存活，权重×step_dt。
4. **动作管线**（V-C/V-D + 附录 A）：逐关节线性映射（0=标称站姿）→ 围绕实测
   关节角 ±τmax/kP 限幅 → 一阶保持插值 + 37.5Hz 低通 → 200Hz 执行器模型。
5. **执行器模型**（附录 B / 表 VI）：软件 PD（q̃=q+εq 编码器偏移）+ tanh 摩擦 +
   速度相关力矩限幅 + 背隙/速度相关噪声编码器读数 + 反射惯量随机化，全部
   每 episode 重采样。
6. **扰动**（表 V）：三档独立进程（髋/脚短小扰动、盆骨长小扰动、盆骨短大推力），
   基础步态等待阶段后线性放开。
7. **相位**：φ̇ 从参考库按命令插值（步频随速度变化），逐步积分。
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude, quat_from_euler_xyz, sample_uniform

from .oceanisaaclab_env import OceanisaaclabEnv
from .oceanisaaclab_walk_env_cfg import OceanisaaclabWalkEnvCfg
from .path_frame import PathFrame, wrap_angle
from .reference_gait import NeckHeadMap, ReferenceGait


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
        if abs(self._reference_gait.gait_duty - self.cfg.gait_duty_factor) > 1e-6:
            raise RuntimeError(
                f"gait library duty {self._reference_gait.gait_duty} != "
                f"cfg.gait_duty_factor {self.cfg.gait_duty_factor}; regenerate the library."
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

        # ---- 脖子/头部：4 关节位置伺服跟随头部命令参考角（论文头部命令 Δh/Δθ_head） ----
        self._neck_head_map = NeckHeadMap(self.cfg.neck_head_map_path, self.device)
        self._neck_joint_ranges = torch.tensor(
            self.cfg.neck_action_joint_ranges, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self._soft_neck_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[
            :, self._neck_dof_idx
        ].clone()
        self._neck_target = self._default_neck_joint_pos.clone()
        self._prev_neck_target = self._default_neck_joint_pos.clone()
        self._filtered_neck_target = self._default_neck_joint_pos.clone()
        self._head_commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._control_resample_left = torch.zeros(self.num_envs, device=self.device)
        self._head_command_scale = torch.tensor(
            self.cfg.head_command_scale, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        # 动作切片：前 10 = 腿（力矩直驱），后 4 = 脖子（位置伺服）
        self._leg_action_slice = slice(0, len(self._leg_dof_idx))
        self._neck_action_slice = slice(len(self._leg_dof_idx), self.cfg.action_space)
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
            self._dist_body_ids = sorted({bid for sched in self._disturbances for bid in sched.body_ids})
            body_slot = {body_id: slot for slot, body_id in enumerate(self._dist_body_ids)}
            self._dist_body_slots = [
                torch.tensor([body_slot[bid] for bid in sched.body_ids], device=self.device)
                for sched in self._disturbances
            ]
            wrench_shape = (self.num_envs, len(self._dist_body_ids), 3)
            self._dist_forces = torch.zeros(wrench_shape, device=self.device)
            self._dist_torques = torch.zeros(wrench_shape, device=self.device)

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
                "neck_joint_pos",
                "neck_joint_vel",
                "neck_action_rate",
                "neck_action_acc",
                "survival",
            ]
        }

        # Give each parallel environment a randomized reflected inertia once. Rewriting
        # PhysX articulation properties on every early-policy fall is disproportionately slow.
        self._write_armature(torch.arange(self.num_envs, device=self.device), randomize=True)
        # path frame 初始化到出生位姿
        self._reset_path_frame(torch.arange(self.num_envs, device=self.device))

        # Episode diagnostics use physical errors/rates rather than reward kernels.
        self._metric_vel_error = torch.zeros(self.num_envs, device=self.device)
        self._metric_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self._metric_left_contact = torch.zeros(self.num_envs, device=self.device)
        self._metric_right_contact = torch.zeros(self.num_envs, device=self.device)
        self._metric_double_support = torch.zeros(self.num_envs, device=self.device)
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        # Direction-conditioned diagnostics: [forward/backward, right/left]. Global
        # contact averages hid the model_39200 backward and left-foot collapse.
        self._metric_direction_contact = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._metric_direction_steps = torch.zeros(self.num_envs, 2, device=self.device)
        self._metric_direction_swing_clearance = torch.zeros(
            self.num_envs, 2, 2, device=self.device
        )
        self._metric_direction_swing_samples = torch.zeros(
            self.num_envs, 2, 2, device=self.device
        )
        self._metric_direction_vx = torch.zeros(self.num_envs, 2, device=self.device)
        self._metric_direction_command_vx = torch.zeros(self.num_envs, 2, device=self.device)
        self._metric_direction_vx_error = torch.zeros(self.num_envs, 2, device=self.device)
        self._metric_neck_tracking_sq = torch.zeros(self.num_envs, device=self.device)
        self._metric_action_saturation = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _head_yaw(self) -> torch.Tensor:
        """躯干头部前向朝向（世界系 yaw）。"""
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w.torch)
        return yaw + self._head_yaw_offset

    def _feet_heading_yaw(self) -> torch.Tensor:
        """Return the calibrated circular mean heading of both feet.

        The URDF mirrors the right/left terminal link frames, so their raw yaw differs
        by approximately pi in the shared q=0 stance. The gait asset stores those q=0
        FK offsets in right/left order; subtracting them before averaging recovers one
        physical heading in the same head-forward convention as the path frame.
        """
        foot_quat = self.robot.data.body_quat_w.torch[:, self._feet_body_ids]
        _, _, foot_yaw = euler_xyz_from_quat(foot_quat.reshape(-1, 4))
        foot_yaw = foot_yaw.view(self.num_envs, len(self._feet_body_ids))
        relative_yaw = wrap_angle(
            foot_yaw - self._reference_gait.foot_yaw_neutral.unsqueeze(0)
        )
        heading = relative_yaw + self._head_yaw_offset
        return torch.atan2(torch.sin(heading).mean(dim=1), torch.cos(heading).mean(dim=1))

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
            # 新 API（write_joint_armature_to_sim_index）：armature 为关键字专用参数
            writer(armature=armature.contiguous(), joint_ids=joint_ids, env_ids=env_ids.to(dtype=torch.int32))
        except TypeError:
            # 旧 API（deprecated write_joint_armature_to_sim）：armature 可位置或关键字
            writer(armature=armature.contiguous(), joint_ids=self._leg_dof_idx, env_ids=env_ids)

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
        self._metric_action_saturation += (torch.abs(self._actions) > 0.98).float()
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
        delayed_leg = delayed_actions[:, self._leg_action_slice]
        desired = self._default_leg_joint_pos + self._joint_ranges * delayed_leg
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
        # 脖子位置目标：后 4 维动作线性映射到默认位 ± range，clamp 到脖子软限位
        delayed_neck = delayed_actions[:, self._neck_action_slice]
        neck_target = self._default_neck_joint_pos + self._neck_joint_ranges * delayed_neck
        self._prev_neck_target = self._neck_target
        self._neck_target = torch.clamp(
            neck_target,
            self._soft_neck_joint_pos_limits[:, :, 0],
            self._soft_neck_joint_pos_limits[:, :, 1],
        )
        self._substep = 0
        # 相位积分：φ̇ 按命令插值，零速时冻结。站立参考与脖子参考
        # 都是静态的，不再向策略输入持续变化的步态时钟。
        phase_rate = self._reference_gait.sample_phase_rate(self._commands)
        phase_rate = phase_rate * self._moving_mask().float()
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
        neck_u = self._prev_neck_target + frac * (self._neck_target - self._prev_neck_target)
        self._filtered_neck_target = self._filtered_neck_target + self._lowpass_alpha * (
            neck_u - self._filtered_neck_target
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
        # 脖子同样经过论文的 FOH + 37.5Hz 低通，再交给位置伺服。
        self.robot.set_joint_position_target_index(
            target=self._filtered_neck_target, joint_ids=self._neck_dof_idx
        )

    def _update_paper_disturbances(self) -> None:
        if not self._disturbances:
            return
        elapsed = max(
            0, self._walk_curriculum_steps() - self.cfg.disturbance_curriculum_delay_steps
        )
        scale = min(1.0, elapsed / float(self.cfg.disturbance_curriculum_steps))
        for sched in self._disturbances:
            sched.step(self.step_dt, scale)
        self._dist_forces.zero_()
        self._dist_torques.zero_()
        for sched, slots in zip(self._disturbances, self._dist_body_slots):
            self._dist_forces.index_add_(1, slots, sched.forces)
            self._dist_torques.index_add_(1, slots, sched.torques)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            self._dist_forces, self._dist_torques, body_ids=self._dist_body_ids, is_global=True
        )

    def _sample_range(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        lo, hi = value_range
        if abs(hi - lo) < 1e-12:
            return torch.full(shape, lo, dtype=torch.float, device=self.device)
        return sample_uniform(lo, hi, shape, self.device)

    def _head_command_curriculum_scale(self) -> float:
        elapsed = max(
            0, self._walk_curriculum_steps() - self.cfg.head_command_curriculum_delay_steps
        )
        return min(1.0, elapsed / float(self.cfg.head_command_curriculum_ramp_steps))

    def _walk_curriculum_steps(self) -> int:
        return self.cfg.contact_match_curriculum_step_offset + self.common_step_counter

    def _resample_controls(self, env_ids: torch.Tensor, initialize_phase: bool = False) -> None:
        """Sample the full paper control range, including within-episode transitions."""
        n = len(env_ids)
        was_moving = self._moving_mask()[env_ids]
        self._commands[env_ids] = 0.0

        vx = self._sample_range(self.cfg.command_vx_range, (n,))
        small_vx = torch.abs(vx) < self.cfg.move_command_threshold
        vx[small_vx] = 0.0

        non_standing = torch.ones(n, dtype=torch.bool, device=self.device)
        standing = torch.rand(n, device=self.device) < self.cfg.stand_still_prob
        vx[standing] = 0.0
        non_standing[standing] = False

        turn_in_place = (torch.rand(n, device=self.device) < self.cfg.turn_in_place_prob) & non_standing
        vx[turn_in_place] = 0.0

        moving = non_standing & ~turn_in_place
        backward = (torch.rand(n, device=self.device) < self.cfg.backward_prob) & moving
        vx[backward] *= -1.0

        moving_or_turning = ~standing
        vy = self._sample_range(self.cfg.command_vy_range, (n,))
        vy[torch.abs(vy) < self.cfg.move_command_threshold] = 0.0
        vy[turn_in_place] = 0.0
        wz = self._sample_range(self.cfg.command_wz_range, (n,))
        wz[torch.abs(wz) < self.cfg.move_command_threshold] = 0.0

        self._commands[env_ids, 0] = vx
        self._commands[env_ids[moving_or_turning], 1] = vy[moving_or_turning]
        self._commands[env_ids[moving_or_turning], 2] = wz[moving_or_turning]
        self._is_standing[env_ids] = standing
        # Hold a neutral head while learning the base gait, then smoothly expose the full
        # command range independently of the action-smoothness curriculum.
        head_command_scale = self._head_command_curriculum_scale()
        self._head_commands[env_ids, 0] = head_command_scale * self._sample_range(
            self.cfg.head_command_dh_range, (n,)
        )
        self._head_commands[env_ids, 1] = head_command_scale * self._sample_range(
            self.cfg.head_command_pitch_range, (n,)
        )
        self._head_commands[env_ids, 2] = head_command_scale * self._sample_range(
            self.cfg.head_command_yaw_range, (n,)
        )
        self._head_commands[env_ids, 3] = head_command_scale * self._sample_range(
            self.cfg.head_command_roll_range, (n,)
        )
        self._control_resample_left[env_ids] = self._sample_range(
            self.cfg.control_resample_interval_s, (n,)
        )

        newly_moving = (~was_moving & moving_or_turning) | initialize_phase
        if torch.any(newly_moving):
            local_ids = torch.nonzero(newly_moving, as_tuple=False).squeeze(-1)
            selected = env_ids[local_ids]
            turn = wz[local_ids]
            left_step_phase = self._reference_gait.gait_duty - 0.5
            right_step_phase = self._reference_gait.gait_duty
            random_side = torch.rand(len(selected), device=self.device) < 0.5
            start_left = torch.where(torch.abs(turn) > 1.0e-4, turn > 0.0, random_side)
            self._phase[selected] = torch.where(
                start_left,
                torch.full_like(turn, left_step_phase),
                torch.full_like(turn, right_step_phase),
            )

    # ------------------------------------------------------------------
    # observations: paper eq. (8) + phase harmonics + command; asymmetric critic
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # The completed transition used the previous command for dynamics and
        # reward. Resample now so the next policy action sees the new target first.
        self._control_resample_left -= self.step_dt
        resample_ids = torch.nonzero(self._control_resample_left <= 0.0, as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._resample_controls(resample_ids)
        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        head_yaw = self._head_yaw()
        pos_pf, yaw_pf = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        projected_gravity = self.robot.data.projected_gravity_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        neck_joint_pos = self.robot.data.joint_pos.torch[:, self._neck_dof_idx]
        neck_joint_vel = self.robot.data.joint_vel.torch[:, self._neck_dof_idx]
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
            projected_gravity_t: torch.Tensor,
            lin_vel_t: torch.Tensor,
            ang_vel_t: torch.Tensor,
            joint_pos_t: torch.Tensor,
            joint_vel_t: torch.Tensor,
        ) -> torch.Tensor:
            return torch.cat(
                (
                    pos_pf_t * self.cfg.pos_pf_scale,
                    yaw_feat_t,
                    projected_gravity_t,
                    lin_vel_t * self.cfg.lin_vel_scale,
                    ang_vel_t * self.cfg.ang_vel_scale,
                    joint_pos_t * self.cfg.dof_pos_scale,
                    neck_joint_pos * self.cfg.dof_pos_scale,
                    joint_vel_t * self.cfg.dof_vel_scale,
                    neck_joint_vel * self.cfg.dof_vel_scale,
                    self._previous_actions,
                    self._prev_prev_actions,
                    phase_feat,
                    self._commands * self._command_scale,
                    self._head_commands * self._head_command_scale,
                ),
                dim=-1,
            )

        # critic：无噪声真值 + 特权信息（附录 A：simulation state without noise + friction）
        critic_obs = torch.cat(
            (
                assemble(pos_pf, yaw_feat, projected_gravity, lin_vel, ang_vel, joint_pos, joint_vel),
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
            projected_gravity_n = projected_gravity + torch.randn_like(projected_gravity) * self.cfg.noise_proj_g
            joint_vel_n = joint_vel + torch.randn_like(joint_vel) * self.cfg.noise_joint_vel
            obs = assemble(
                pos_pf,
                yaw_feat,
                projected_gravity_n,
                lin_vel_n,
                ang_vel_n,
                joint_pos_hat,
                joint_vel_n,
            )
        else:
            obs = critic_obs[:, : self.cfg.observation_space]
        return {"policy": obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # dones: paper V-B ground-contact termination adapted to the nested URDF collision geometry
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
            self._feet_heading_yaw(),
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        upright_projection = -self.robot.data.projected_gravity_b.torch[:, 2]
        terminated = (base_height < self.cfg.walk_min_base_height) | (
            upright_projection < self.cfg.walk_min_upright_projection
        )
        return terminated, time_out

    # ------------------------------------------------------------------
    # rewards: paper Table I (leg subset), weights × step_dt
    # ------------------------------------------------------------------
    def _reward_curriculum_blend(self) -> float:
        """Return 1 at bootstrap start and cosine-anneal to 0 at normal weights."""
        if not self.cfg.enable_contact_match_curriculum:
            return 0.0
        elapsed_steps = self._walk_curriculum_steps()
        progress = min(1.0, elapsed_steps / float(self.cfg.contact_match_anneal_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _current_gait_reward_weights(self) -> tuple[float, float, float, float]:
        """Schedule existing gait terms without introducing a separate bootstrap reward."""
        cfg = self.cfg
        blend = self._reward_curriculum_blend()

        def anneal(initial: float, normal: float) -> float:
            return normal + (initial - normal) * blend

        return (
            anneal(cfg.rew_w_contact_match_initial, cfg.rew_w_contact_match),
            anneal(cfg.rew_w_leg_joint_pos_initial, cfg.rew_w_leg_joint_pos),
            anneal(cfg.rew_w_action_rate_initial, cfg.rew_w_action_rate),
            anneal(cfg.rew_w_action_acc_initial, cfg.rew_w_action_acc),
        )

    def _current_contact_match_weight(self) -> float:
        return self._current_gait_reward_weights()[0]

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        ref = self._reference_gait.sample(self._commands, self._phase)
        contact_weight, leg_joint_pos_weight, action_rate_weight, action_acc_weight = (
            self._current_gait_reward_weights()
        )

        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        head_yaw = self._head_yaw()
        pos_pf, _ = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        joint_acc = self.robot.data.joint_acc.torch[:, self._leg_dof_idx]
        neck_joint_acc = self.robot.data.joint_acc.torch[:, self._neck_dof_idx]
        neck_joint_pos = self.robot.data.joint_pos.torch[:, self._neck_dof_idx]
        neck_joint_vel = self.robot.data.joint_vel.torch[:, self._neck_dof_idx]
        neck_ref = ref["neck_pos"] + self._neck_head_map.sample(self._head_commands)
        neck_vel_ref = ref["neck_vel"]
        in_contact = (self._feet_current_contact_time() > 0.0).float()

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
        rew_joint_pos = leg_joint_pos_weight * torch.sum(torch.square(joint_pos - ref["joint_pos"]), dim=1)
        rew_joint_vel = cfg.rew_w_leg_joint_vel * torch.sum(torch.square(joint_vel - ref["joint_vel"]), dim=1)
        # 6) 接触模仿：去掉动作无关的正基线，只惩罚与参考不一致的脚。
        #    零速 ref=[1,1] 仍要求双脚支撑；行走摆动相持续着地则明确扣分。
        ref_contact = (ref["feet_contact"] >= 0.5).float()
        rew_contact = -contact_weight * torch.sum(
            (in_contact != ref_contact).float(), dim=1
        )
        # 7) 正则：力矩 / 关节加速度 / 动作率 / 动作加速度
        neck_torque = self.robot.data.applied_torque.torch[:, self._neck_dof_idx]
        rew_torque = cfg.rew_w_torque * (
            torch.sum(torch.square(self._applied_leg_torque), dim=1)
            + torch.sum(torch.square(neck_torque), dim=1)
        )
        rew_joint_acc = cfg.rew_w_joint_acc * (
            torch.sum(torch.square(joint_acc), dim=1)
            + torch.sum(torch.square(neck_joint_acc), dim=1)
        )
        rew_action_rate = action_rate_weight * torch.sum(
            torch.square(self._actions[:, self._leg_action_slice]
                         - self._previous_actions[:, self._leg_action_slice]), dim=1
        )
        rew_action_acc = action_acc_weight * torch.sum(
            torch.square(self._actions[:, self._leg_action_slice]
                         - 2.0 * self._previous_actions[:, self._leg_action_slice]
                         + self._prev_prev_actions[:, self._leg_action_slice]), dim=1
        )
        # 8) 脖子/头部模仿（论文表 I neck 项）：脖子关节角跟随头命令参考角 + 速度惩罚
        #    + 脖子动作率/加速度（腿与脖子动作率分开加权，避免脖子快动作被腿的权重误伤）
        rew_neck_pos = cfg.rew_w_neck_joint_pos * torch.sum(torch.square(neck_joint_pos - neck_ref), dim=1)
        rew_neck_vel = cfg.rew_w_neck_joint_vel * torch.sum(
            torch.square(neck_joint_vel - neck_vel_ref), dim=1
        )
        rew_neck_action_rate = cfg.rew_w_neck_action_rate * torch.sum(
            torch.square(self._actions[:, self._neck_action_slice]
                         - self._previous_actions[:, self._neck_action_slice]), dim=1
        )
        rew_neck_action_acc = cfg.rew_w_neck_action_acc * torch.sum(
            torch.square(self._actions[:, self._neck_action_slice]
                         - 2.0 * self._previous_actions[:, self._neck_action_slice]
                         + self._prev_prev_actions[:, self._neck_action_slice]), dim=1
        )
        # 9) 存活
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
            "neck_joint_pos": rew_neck_pos,
            "neck_joint_vel": rew_neck_vel,
            "neck_action_rate": rew_neck_action_rate,
            "neck_action_acc": rew_neck_action_acc,
            "survival": rew_survival,
        }
        # 表 I 权重按 legged_gym/Isaac Gym 约定乘 step_dt（论文同一代码谱系）
        total_reward = torch.stack(list(reward_terms.values()), dim=0).sum(dim=0) * self.step_dt
        for key, value in reward_terms.items():
            self._episode_sums[key] += value * self.step_dt
        vel_error = torch.linalg.norm(lin_vel[:, :2] - ref["lin_vel_b"][:, :2], dim=1)
        self._metric_vel_error += vel_error
        self._metric_yaw_error += torch.abs(ang_vel[:, 2] - ref["ang_vel_b"][:, 2])
        self._metric_left_contact += in_contact[:, 1]
        self._metric_right_contact += in_contact[:, 0]
        self._metric_double_support += torch.prod(in_contact, dim=1)
        self._metric_steps += 1.0
        self._metric_neck_tracking_sq += torch.mean(
            torch.square(neck_joint_pos - neck_ref), dim=1
        )
        foot_clearance = (
            self.robot.data.body_pos_w.torch[:, self._feet_body_ids, 2]
            - self.scene.env_origins[:, 2].unsqueeze(1)
            - self.cfg.foot_origin_offset
        )
        ref_swing = 1.0 - ref_contact
        direction_masks = (
            self._commands[:, 0] > self.cfg.move_command_threshold,
            self._commands[:, 0] < -self.cfg.move_command_threshold,
        )
        for direction_id, direction_mask in enumerate(direction_masks):
            mask = direction_mask.float().unsqueeze(1)
            self._metric_direction_contact[:, direction_id] += in_contact * mask
            self._metric_direction_steps[:, direction_id] += direction_mask.float()
            direction_weight = direction_mask.float()
            physical_vx = self.cfg.forward_vx_sign * lin_vel[:, 0]
            self._metric_direction_vx[:, direction_id] += physical_vx * direction_weight
            self._metric_direction_command_vx[:, direction_id] += (
                self._commands[:, 0] * direction_weight
            )
            self._metric_direction_vx_error[:, direction_id] += (
                torch.abs(physical_vx - self._commands[:, 0]) * direction_weight
            )
            swing_mask = ref_swing * mask
            self._metric_direction_swing_clearance[:, direction_id] += foot_clearance * swing_mask
            self._metric_direction_swing_samples[:, direction_id] += swing_mask
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
        metric_steps = self._metric_steps[env_ids].clamp_min(1.0)
        log = self.extras.setdefault("log", {})
        log["Metrics/velocity_mae"] = torch.mean(self._metric_vel_error[env_ids] / metric_steps)
        log["Metrics/yaw_rate_mae"] = torch.mean(self._metric_yaw_error[env_ids] / metric_steps)
        log["Metrics/left_contact_rate"] = torch.mean(self._metric_left_contact[env_ids] / metric_steps)
        log["Metrics/right_contact_rate"] = torch.mean(self._metric_right_contact[env_ids] / metric_steps)
        log["Metrics/double_support_rate"] = torch.mean(self._metric_double_support[env_ids] / metric_steps)
        log["Metrics/neck_tracking_rms_rad"] = torch.sqrt(
            torch.mean(self._metric_neck_tracking_sq[env_ids] / metric_steps)
        )
        for direction_id, direction_name in enumerate(("forward", "backward")):
            direction_steps = self._metric_direction_steps[env_ids, direction_id].sum().clamp_min(1.0)
            log[f"Metrics/{direction_name}_actual_vx"] = (
                self._metric_direction_vx[env_ids, direction_id].sum() / direction_steps
            )
            log[f"Metrics/{direction_name}_command_vx"] = (
                self._metric_direction_command_vx[env_ids, direction_id].sum() / direction_steps
            )
            log[f"Metrics/{direction_name}_vx_mae"] = (
                self._metric_direction_vx_error[env_ids, direction_id].sum() / direction_steps
            )
            for foot_id, foot_name in enumerate(("right", "left")):
                contact_sum = self._metric_direction_contact[env_ids, direction_id, foot_id].sum()
                swing_samples = self._metric_direction_swing_samples[
                    env_ids, direction_id, foot_id
                ].sum().clamp_min(1.0)
                clearance_sum = self._metric_direction_swing_clearance[
                    env_ids, direction_id, foot_id
                ].sum()
                log[f"Metrics/{direction_name}_{foot_name}_contact_rate"] = (
                    contact_sum / direction_steps
                )
                log[f"Metrics/{direction_name}_{foot_name}_ref_swing_clearance_cm"] = (
                    100.0 * clearance_sum / swing_samples
                )
        contact_weight, leg_joint_pos_weight, action_rate_weight, action_acc_weight = (
            self._current_gait_reward_weights()
        )
        log["Curriculum/contact_match_weight"] = contact_weight
        log["Curriculum/leg_joint_pos_weight"] = leg_joint_pos_weight
        log["Curriculum/action_rate_weight"] = action_rate_weight
        log["Curriculum/action_acc_weight"] = action_acc_weight
        log["Curriculum/head_command_scale"] = self._head_command_curriculum_scale()
        disturbance_elapsed = max(
            0, self._walk_curriculum_steps() - self.cfg.disturbance_curriculum_delay_steps
        )
        log["Curriculum/disturbance_scale"] = min(
            1.0, disturbance_elapsed / float(self.cfg.disturbance_curriculum_steps)
        )
        saturation = self._metric_action_saturation[env_ids] / metric_steps.unsqueeze(1)
        for action_id, name in enumerate(self.cfg.leg_joint_names + self.cfg.neck_joint_names):
            log[f"Metrics/action_saturation/{name}"] = torch.mean(saturation[:, action_id])
        for metric in (
            self._metric_vel_error,
            self._metric_yaw_error,
            self._metric_left_contact,
            self._metric_right_contact,
            self._metric_double_support,
            self._metric_neck_tracking_sq,
            self._metric_steps,
        ):
            metric[env_ids] = 0.0
        self._metric_direction_contact[env_ids] = 0.0
        self._metric_direction_steps[env_ids] = 0.0
        self._metric_direction_swing_clearance[env_ids] = 0.0
        self._metric_direction_swing_samples[env_ids] = 0.0
        self._metric_direction_vx[env_ids] = 0.0
        self._metric_direction_command_vx[env_ids] = 0.0
        self._metric_direction_vx_error[env_ids] = 0.0
        self._metric_action_saturation[env_ids] = 0.0

        self._phase[env_ids] = torch.rand(n, device=self.device)
        self._resample_controls(env_ids, initialize_phase=True)
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        # 动作管线缓冲复位
        self._prev_prev_actions[env_ids] = 0.0
        self._target_joint_pos[env_ids] = self._default_leg_joint_pos[env_ids]
        self._prev_target_joint_pos[env_ids] = self._default_leg_joint_pos[env_ids]
        self._filtered_joint_target[env_ids] = self._default_leg_joint_pos[env_ids]
        self._last_tau_m[env_ids] = 0.0
        self._applied_leg_torque[env_ids] = 0.0
        reset_ref = self._reference_gait.sample(self._commands[env_ids], self._phase[env_ids])
        neck_ref_reset = reset_ref["neck_pos"] + self._neck_head_map.sample(self._head_commands[env_ids])
        self._neck_target[env_ids] = neck_ref_reset
        self._prev_neck_target[env_ids] = neck_ref_reset
        self._filtered_neck_target[env_ids] = neck_ref_reset
        reset_actions = torch.zeros(n, self.cfg.action_space, device=self.device)
        reset_actions[:, self._neck_action_slice] = torch.clamp(
            (neck_ref_reset - self._default_neck_joint_pos[env_ids]) / self._neck_joint_ranges,
            -1.0,
            1.0,
        )
        self._actions[env_ids] = reset_actions
        self._previous_actions[env_ids] = reset_actions
        self._prev_prev_actions[env_ids] = reset_actions
        self._action_history[env_ids] = reset_actions.unsqueeze(1)
        # 附录 B 每 episode 随机量：编码器偏移 / 背隙 / PD 增益。
        self._encoder_offset[env_ids] = (
            torch.rand(n, 10, device=self.device) * 2.0 - 1.0
        ) * self._act_eps_max
        self._backlash[env_ids] = self._act_b_min + (self._act_b_max - self._act_b_min) * torch.rand(
            n, 10, device=self.device
        )
        lo, hi = self.cfg.actuator_gain_rand_range
        self._gain_scale_kp[env_ids] = sample_uniform(lo, hi, (n, 10), self.device)
        self._gain_scale_kd[env_ids] = sample_uniform(lo, hi, (n, 10), self.device)
        if self.cfg.randomize_armature_each_reset:
            self._write_armature(env_ids, randomize=True)
        # 表 V 扰动进程复位
        for sched in self._disturbances:
            sched.reset(env_ids)

        # ---- RSI：部分 env 从随机相位参考帧出发 ----
        # Zero-speed resets start at the converged shared stand/walk path frame so q=0 is
        # already the common hand-off state. Moving resets still begin at the torso state.
        pf_offset = torch.zeros(n, 2, device=self.device)
        standing_reset = ~self._moving_mask()[env_ids]
        if torch.any(standing_reset):
            standing_count = int(standing_reset.sum().item())
            neutral_ref = self._reference_gait.sample(
                torch.zeros(standing_count, 3, device=self.device),
                torch.zeros(standing_count, device=self.device),
            )
            pf_offset[standing_reset] = neutral_ref["base_pos_pf"]
        rsi_prob = self.cfg.rsi_prob
        rsi_joint_pos_noise = self.cfg.rsi_joint_pos_noise
        rsi = torch.rand(n, device=self.device) < rsi_prob
        if torch.any(rsi):
            rsi_ids = env_ids[rsi]
            ref = self._reference_gait.sample(self._commands[rsi_ids], self._phase[rsi_ids])

            joint_pos = self.robot.data.default_joint_pos.torch[rsi_ids].clone()
            leg_pos = ref["joint_pos"] + torch.empty_like(ref["joint_pos"]).uniform_(
                -rsi_joint_pos_noise, rsi_joint_pos_noise
            )
            soft_limits = self.robot.data.soft_joint_pos_limits.torch[rsi_ids][:, self._leg_dof_idx]
            joint_pos[:, self._leg_dof_idx] = torch.clamp(leg_pos, soft_limits[..., 0], soft_limits[..., 1])
            # 脖子初始角 = 该 env 头命令对应参考角（与位置目标一致，出生即到位不跳变）
            joint_pos[:, self._neck_dof_idx] = neck_ref_reset[rsi]
            joint_vel = self.robot.data.default_joint_vel.torch[rsi_ids].clone()
            joint_vel[:, self._leg_dof_idx] = ref["joint_vel"]
            joint_vel[:, self._neck_dof_idx] = ref["neck_vel"]

            root_pose = self.robot.data.default_root_pose.torch[rsi_ids].clone()
            root_pose[:, :3] = self.scene.env_origins[rsi_ids]
            root_pose[:, 2] += ref["base_height"]
            zeros = torch.zeros_like(ref["base_pitch"])
            root_pose[:, 3:7] = quat_from_euler_xyz(zeros, ref["base_pitch"], zeros)
            root_vel = self.robot.data.default_root_vel.torch[rsi_ids].clone()
            root_vel[:, :3] = ref["lin_vel_b"]
            root_vel[:, 3:6] = ref["ang_vel_b"]

            self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=rsi_ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=rsi_ids)
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=rsi_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=rsi_ids)
            # RSI 的动作管线缓冲对齐参考关节角（避免第一步 setpoint 从 0 跳变）
            self._target_joint_pos[rsi_ids] = joint_pos[:, self._leg_dof_idx]
            self._prev_target_joint_pos[rsi_ids] = joint_pos[:, self._leg_dof_idx]
            self._filtered_joint_target[rsi_ids] = joint_pos[:, self._leg_dof_idx]
            rsi_actions = reset_actions[rsi].clone()
            rsi_actions[:, self._leg_action_slice] = torch.clamp(
                (joint_pos[:, self._leg_dof_idx] - self._default_leg_joint_pos[rsi_ids])
                / self._joint_ranges,
                -1.0,
                1.0,
            )
            self._actions[rsi_ids] = rsi_actions
            self._previous_actions[rsi_ids] = rsi_actions
            self._prev_prev_actions[rsi_ids] = rsi_actions
            self._action_history[rsi_ids] = rsi_actions.unsqueeze(1)
            # path frame 原点偏移，使躯干出生即位于参考 sway 相位上
            pf_offset[rsi] = ref["base_pos_pf"]
        self._reset_path_frame(env_ids, pf_offset)
