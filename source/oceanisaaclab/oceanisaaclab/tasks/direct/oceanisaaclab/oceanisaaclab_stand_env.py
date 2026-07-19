# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文复刻）站立（perpetual）训练环境。

论文 divide-and-conquer：periodic 行走策略之外单独训练 perpetual 站立策略
π(a | s, g_perp)，**无相位**，命令 g_perp = (Δh_head, Δθ_head, h_torso,
θ_torso)（式 5）。静态参考保持双脚支撑，但受扰时允许策略偏离参考并跨步恢复。继承行走环境
OceanisaaclabWalkEnv，复用附录 B 执行器模型、表 V 扰动、
域随机化、torso 等效触地终止、非对称 critic、path frame、脖子位置伺服、脖子/头部命令映射；
覆盖：观测（去相位谐波、命令换 torso4+head4）、奖励（静态姿态模仿）、reset（站立命令
采样 + 从标称站姿出生）。行走的动作管线 / 执行器模型 / 脖子通路完全沿用。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    sample_uniform,
)

from .oceanisaaclab_stand_env_cfg import OceanisaaclabStandEnvCfg
from .oceanisaaclab_walk_env import OceanisaaclabWalkEnv
from .path_frame import wrap_angle
from .reference_gait import StandPose, StandRecoveryResetLibrary


class OceanisaaclabStandEnv(OceanisaaclabWalkEnv):
    cfg: OceanisaaclabStandEnvCfg

    def __init__(self, cfg: OceanisaaclabStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 站立姿态参考库（torso4 + head4 -> 全身 CoM 平衡腿角）；脖子角沿用 _neck_head_map。
        self._stand_pose = StandPose(self.cfg.stand_pose_path, self.device)
        self._stand_recovery_resets = StandRecoveryResetLibrary(
            self.cfg.stand_recovery_reset_path, self.device
        )
        neutral_torso = torch.zeros(1, 4, device=self.device)
        neutral_head = torch.zeros(1, 4, device=self.device)
        neutral_leg = self._stand_pose.sample(neutral_torso, neutral_head)
        if not torch.allclose(
            neutral_leg,
            self._default_leg_joint_pos[:1],
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "stand_pose.npz zero command must equal the shared URDF q=0 pose; "
                "regenerate it with scripts/gen_stand_pose.py."
            )
        if not torch.allclose(
            self._stand_pose.foot_yaw_neutral,
            self._reference_gait.foot_yaw_neutral,
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise RuntimeError("Stand/walk q=0 foot-frame yaw calibration does not match.")
        walk_neutral = self._reference_gait.sample(
            torch.zeros(1, 3, device=self.device),
            torch.zeros(1, device=self.device),
        )
        if not torch.allclose(
            walk_neutral["joint_pos"],
            self._default_leg_joint_pos[:1],
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise RuntimeError("Walking zero-command reference must equal the shared URDF q=0 pose.")
        if abs(self._stand_pose.base_height - float(walk_neutral["base_height"][0])) > 1.0e-6:
            raise RuntimeError("Stand/walk zero-command base heights do not match.")
        if not torch.allclose(
            self._stand_pose.sample_base_pos_pf(neutral_torso, neutral_head),
            walk_neutral["base_pos_pf"],
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise RuntimeError("Stand/walk zero-command path-frame torso positions do not match.")
        if self._stand_recovery_resets.joint_names != self._stand_pose.joint_names:
            raise RuntimeError("Stand recovery reset joint order does not match stand_pose.npz.")
        if abs(self._stand_recovery_resets.base_height - self._stand_pose.base_height) > 1.0e-6:
            raise RuntimeError("Stand recovery reset and reference base heights do not match.")
        self._torso_commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._torso_command_scale = torch.tensor(
            self.cfg.torso_command_scale, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self._stand_command_grace_left = torch.zeros(self.num_envs, device=self.device)
        # Recovery is a diagnostic Schmitt state only. It never changes the reward.
        self._recovery_state = torch.zeros(self.num_envs, device=self.device)
        self._recovery_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._recovery_hold_left = torch.zeros(self.num_envs, device=self.device)
        # 覆盖奖励分项统计：换成论文站立项集合，去掉 gait 与额外 height 项。
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
                "feet_slide",
                "feet_airborne",
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
        # Stand-specific diagnostics. Parent contact/action metrics are also updated below so
        # their TensorBoard series no longer stay at a misleading constant zero.
        self._stand_metric_steps = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_signal = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_state = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_steps = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_single_support = torch.zeros(self.num_envs, device=self.device)
        self._metric_liftoff_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_liftoff_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_touchdown_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_touchdown_step_distance = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_touchdown_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_recovery_touchdown_step_distance = torch.zeros(
            self.num_envs, device=self.device
        )
        self._metric_recovery_capture_steps = torch.zeros(self.num_envs, device=self.device)
        self._metric_quiet_liftoff_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_quiet_foot_drift = torch.zeros(self.num_envs, device=self.device)
        self._metric_clean_steps = torch.zeros(self.num_envs, device=self.device)
        self._metric_clean_liftoff_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_clean_double_support = torch.zeros(self.num_envs, device=self.device)
        self._metric_stance_width_error = torch.zeros(self.num_envs, device=self.device)
        self._metric_stance_stagger_error = torch.zeros(self.num_envs, device=self.device)
        self._metric_stance_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self._metric_displaced_steps = torch.zeros(self.num_envs, device=self.device)
        self._metric_displaced_liftoff_events = torch.zeros(self.num_envs, device=self.device)
        self._metric_displaced_double_support = torch.zeros(self.num_envs, device=self.device)
        self._stand_displaced_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stand_displaced_reset_category = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._stand_displaced_recovery_hold = torch.zeros(self.num_envs, device=self.device)
        self._stand_displaced_recovery_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._stand_displaced_recovery_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stand_displaced_recovery_time = torch.zeros(self.num_envs, device=self.device)
        self._stand_previous_contact = torch.ones(self.num_envs, 2, device=self.device)
        self._stand_foot_airborne = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._stand_liftoff_xy = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._stand_liftoff_recovery = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._stand_airborne_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._stand_previous_foot_xy = torch.zeros(self.num_envs, 2, 2, device=self.device)
        # Each episode is either a nominal no-disturbance baseline or a full Table V
        # disturbance rollout. The policy never observes this label.
        self._stand_disturbance_enabled = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._sample_stand_disturbance_modes(torch.arange(self.num_envs, device=self.device))

    def _sample_stand_disturbance_modes(self, env_ids: torch.Tensor) -> None:
        """Sample clean/disturbed episode modes while retaining the paper force process."""
        if not self._disturbances:
            self._stand_disturbance_enabled[env_ids] = False
            return
        quiet = torch.rand(len(env_ids), device=self.device) < self.cfg.stand_disturbance_quiet_prob
        self._stand_disturbance_enabled[env_ids] = ~quiet

    def _displaced_reset_curriculum_scale(self) -> float:
        initial = float(self.cfg.stand_displaced_reset_initial_scale)
        duration = max(1, int(self.cfg.stand_displaced_reset_curriculum_steps))
        # Include the checkpoint-derived offset so resume does not restart only this
        # curriculum while the paper disturbance/head curricula remain fully open.
        progress = min(1.0, float(self._walk_curriculum_steps()) / duration)
        return initial + (1.0 - initial) * progress

    def _sample_stand_reset_modes(self, env_ids: torch.Tensor) -> None:
        """Select non-canonical resets only inside no-disturbance episodes."""
        quiet = ~self._stand_disturbance_enabled[env_ids]
        displaced = quiet & (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.stand_displaced_reset_prob_within_quiet
        )
        self._stand_displaced_reset[env_ids] = displaced
        self._stand_displaced_reset_category[env_ids] = -1

    def _stance_relative_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left-to-right foot translation and yaw in the calibrated feet frame."""
        foot_pos = self.robot.data.body_pos_w.torch[:, self._feet_body_ids, :2]
        delta_w = foot_pos[:, 1] - foot_pos[:, 0]
        heading = self._feet_heading_yaw()
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        relative_xy = torch.stack(
            (
                delta_w[:, 0] * cos_h + delta_w[:, 1] * sin_h,
                -delta_w[:, 0] * sin_h + delta_w[:, 1] * cos_h,
            ),
            dim=1,
        )
        foot_quat = self.robot.data.body_quat_w.torch[:, self._feet_body_ids]
        _, _, raw_yaw = euler_xyz_from_quat(foot_quat.reshape(-1, 4))
        raw_yaw = raw_yaw.view(self.num_envs, 2)
        calibrated_yaw = wrap_angle(
            raw_yaw - self._stand_pose.foot_yaw_neutral.unsqueeze(0)
        )
        relative_yaw = wrap_angle(calibrated_yaw[:, 1] - calibrated_yaw[:, 0])
        return relative_xy, relative_yaw

    def _update_paper_disturbances(self) -> None:
        """Apply Table V, then suppress all wrenches for selected clean episodes."""
        super()._update_paper_disturbances()
        if not self._disturbances or not hasattr(self, "_stand_disturbance_enabled"):
            return
        enabled = self._stand_disturbance_enabled.view(self.num_envs, 1, 1)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            self._dist_forces * enabled,
            self._dist_torques * enabled,
            body_ids=self._dist_body_ids,
            is_global=True,
        )

    # ------------------------------------------------------------------
    # observations: 论文式 (8) 状态 + g_perp（无相位；命令 = torso4 + head4）
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._control_resample_left -= self.step_dt
        resample_ids = torch.nonzero(
            self._control_resample_left <= 0.0, as_tuple=False
        ).squeeze(-1)
        if len(resample_ids) > 0:
            self._resample_stand_controls(resample_ids)
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
        yaw_feat = torch.stack((torch.sin(yaw_pf), torch.cos(yaw_pf)), dim=1)
        g_perp = torch.cat(
            (self._torso_commands * self._torso_command_scale,
             self._head_commands * self._head_command_scale),
            dim=-1,
        )

        def assemble(projected_gravity_t, lin_vel_t, ang_vel_t, joint_pos_t, joint_vel_t):
            return torch.cat(
                (
                    pos_pf * self.cfg.pos_pf_scale,
                    yaw_feat,
                    projected_gravity_t,
                    lin_vel_t * self.cfg.lin_vel_scale,
                    ang_vel_t * self.cfg.ang_vel_scale,
                    joint_pos_t * self.cfg.dof_pos_scale,
                    neck_joint_pos * self.cfg.dof_pos_scale,
                    joint_vel_t * self.cfg.dof_vel_scale,
                    neck_joint_vel * self.cfg.dof_vel_scale,
                    self._previous_actions,
                    self._prev_prev_actions,
                    g_perp,
                ),
                dim=-1,
            )

        critic_obs = torch.cat(
            (
                assemble(projected_gravity, lin_vel, ang_vel, joint_pos, joint_vel),
                self._dr_friction_scale.unsqueeze(1),
                self._dr_mass_scale.unsqueeze(1),
            ),
            dim=-1,
        )
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
            projected_gravity_n = (
                projected_gravity + torch.randn_like(projected_gravity) * self.cfg.noise_proj_g
            )
            joint_vel_n = joint_vel + torch.randn_like(joint_vel) * self.cfg.noise_joint_vel
            obs = assemble(projected_gravity_n, lin_vel_n, ang_vel_n, joint_pos_hat, joint_vel_n)
        else:
            obs = critic_obs[:, : self.cfg.observation_space]
        return {"policy": obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # rewards: 静态姿态模仿（躯干位姿 + 速度→0 + 腿/脖子模仿 + 双脚接触 + 正则）
    # ------------------------------------------------------------------
    @staticmethod
    def _linear_recovery_signal(value: torch.Tensor, start: float, full: float) -> torch.Tensor:
        """Map a physical instability measure to a normalized diagnostic signal."""
        return torch.clamp((value - start) / max(full - start, 1.0e-6), 0.0, 1.0)

    def _recovery_signal_and_state(
        self,
        projected_gravity: torch.Tensor,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        pitch_cmd: torch.Tensor,
        roll_cmd: torch.Tensor,
        planar_error: torch.Tensor,
        in_contact: torch.Tensor,
    ) -> torch.Tensor:
        # Gravity expressed in the desired torso frame. Yaw does not change gravity, so a
        # commanded yaw pose cannot accidentally enable the diagnostic state.
        cos_pitch = torch.cos(pitch_cmd)
        desired_gravity = torch.stack(
            (
                torch.sin(pitch_cmd),
                -torch.sin(roll_cmd) * cos_pitch,
                -torch.cos(roll_cmd) * cos_pitch,
            ),
            dim=1,
        )
        gravity_cos = torch.sum(projected_gravity * desired_gravity, dim=1).clamp(-1.0, 1.0)
        tilt_error = torch.acos(gravity_cos)
        horizontal_speed = torch.linalg.vector_norm(lin_vel[:, :2], dim=1)
        tilt_rate = torch.linalg.vector_norm(ang_vel[:, :2], dim=1)

        tilt_gate = self._linear_recovery_signal(
            tilt_error, self.cfg.recovery_tilt_error_start, self.cfg.recovery_tilt_error_full
        )
        velocity_gate = self._linear_recovery_signal(
            horizontal_speed, self.cfg.recovery_lin_vel_start, self.cfg.recovery_lin_vel_full
        )
        angular_gate = self._linear_recovery_signal(
            tilt_rate, self.cfg.recovery_ang_vel_start, self.cfg.recovery_ang_vel_full
        )
        position_gate = self._linear_recovery_signal(
            planar_error, self.cfg.recovery_pos_error_start, self.cfg.recovery_pos_error_full
        )

        # A new posture command changes the desired orientation instantaneously. Do not mistake
        # that reference jump for a physical push. The planar reference moves by less than 2 mm
        # over the full stand-pose grid, so position error remains a valid recovery signal here.
        command_grace = self._stand_command_grace_left > 0.0
        tilt_gate = torch.where(command_grace, torch.zeros_like(tilt_gate), tilt_gate)
        instant_signal = torch.maximum(
            torch.maximum(tilt_gate, velocity_gate),
            torch.maximum(angular_gate, position_gate),
        )

        # Diagnostic-only Schmitt state. It is deliberately not connected to the reward: Table I
        # remains active in both quiet standing and recovery, as in the paper.
        double_support = torch.sum(in_contact, dim=1) > 1.5
        triggered = instant_signal >= self.cfg.recovery_activation_severity
        was_active = self._recovery_active
        stable_for_release = instant_signal <= self.cfg.recovery_deactivation_severity
        refresh_hold = triggered | (was_active & (~double_support | ~stable_for_release))
        hold_duration = torch.full_like(self._recovery_hold_left, self.cfg.recovery_hold_s)
        self._recovery_hold_left = torch.where(
            refresh_hold,
            hold_duration,
            torch.clamp(self._recovery_hold_left - self.step_dt, min=0.0),
        )
        still_active = was_active & (
            ~double_support | ~stable_for_release | (self._recovery_hold_left > 0.0)
        )
        active = triggered | still_active
        self._recovery_active.copy_(active)
        self._recovery_state.copy_(active.float())
        self._stand_command_grace_left.sub_(self.step_dt).clamp_(min=0.0)
        return instant_signal

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        stand_ref = self._stand_pose.sample_reference(
            self._torso_commands, self._head_commands
        )
        ref_leg = stand_ref["joint_pos"]
        ref_base_pos_pf = stand_ref["base_pos_pf"]
        ref_base_yaw_pf = stand_ref["base_yaw_pf"]
        neck_ref = self._neck_head_map.sample(self._head_commands)  # (N,4)

        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        head_yaw = self._head_yaw()
        pos_pf, _ = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        planar_error = torch.linalg.vector_norm(pos_pf - ref_base_pos_pf, dim=1)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        joint_acc = self.robot.data.joint_acc.torch[:, self._leg_dof_idx]
        neck_joint_acc = self.robot.data.joint_acc.torch[:, self._neck_dof_idx]
        neck_joint_pos = self.robot.data.joint_pos.torch[:, self._neck_dof_idx]
        neck_joint_vel = self.robot.data.joint_vel.torch[:, self._neck_dof_idx]
        in_contact = (self._feet_current_contact_time() > 0.0).float()
        feet_lin_vel_xy = self.robot.data.body_lin_vel_w.torch[:, self._feet_body_ids, :2]

        # 命令：躯干 (h, pitch, yaw, roll)
        pitch_cmd = self._torso_commands[:, 1]
        roll_cmd = self._torso_commands[:, 3]
        projected_gravity = self.robot.data.projected_gravity_b.torch
        recovery_signal = self._recovery_signal_and_state(
            projected_gravity,
            lin_vel,
            ang_vel,
            pitch_cmd,
            roll_cmd,
            planar_error,
            in_contact,
        )
        # Fixed Table I weights in all states. Recovery is a diagnostic label only; relaxing
        # contact/leg imitation here reproduces the paper's documented foot-shuffling failure.
        recovery_active = self._recovery_active
        leg_joint_pos_weight = torch.full_like(recovery_signal, cfg.rew_w_leg_joint_pos)
        action_rate_weight = torch.full_like(recovery_signal, cfg.rew_w_action_rate)
        action_acc_weight = torch.full_like(recovery_signal, cfg.rew_w_action_acc)
        contact_weight_scale = torch.ones_like(recovery_signal)

        # 1) 躯干 path 系 xy：跟随离线运动学参考。q=0 时躯干相对双脚
        # link 原点中心保留真实前向偏移，不能强制成 path-frame 原点。
        pos_err = torch.square(planar_error)
        rew_pos_xy = cfg.rew_w_torso_pos_xy * torch.exp(-cfg.rew_k_torso_pos_xy * pos_err)
        # 2) 躯干朝向跟可实现 reference state：yaw 是 torso/head 相对 solved
        # 双脚平均 heading，语义与 walking reference 的 base_yaw_pf 一致。
        base_yaw_ref = self._path_frame.yaw + ref_base_yaw_pf - self._head_yaw_offset
        quat_ref = quat_from_euler_xyz(roll_cmd, pitch_cmd, base_yaw_ref)
        orient_err = quat_error_magnitude(self.robot.data.root_quat_w.torch, quat_ref)
        rew_orient = cfg.rew_w_torso_orient * torch.exp(-cfg.rew_k_torso_orient * torch.square(orient_err))
        # 3) 线速度 → 0（惩罚水平/竖直移动）
        rew_lin_xy = cfg.rew_w_lin_vel_xy * torch.exp(-cfg.rew_k_lin_vel * torch.sum(torch.square(lin_vel[:, :2]), dim=1))
        rew_lin_z = cfg.rew_w_lin_vel_z * torch.exp(-cfg.rew_k_lin_vel * torch.square(lin_vel[:, 2]))
        # 4) 角速度 → 0
        rew_ang_xy = cfg.rew_w_ang_vel_xy * torch.exp(-cfg.rew_k_ang_vel * torch.sum(torch.square(ang_vel[:, :2]), dim=1))
        rew_ang_z = cfg.rew_w_ang_vel_z * torch.exp(-cfg.rew_k_ang_vel * torch.square(ang_vel[:, 2]))
        # 5) 腿关节角模仿站立参考 / 关节速度 → 0
        rew_joint_pos = leg_joint_pos_weight * torch.sum(torch.square(joint_pos - ref_leg), dim=1)
        rew_joint_vel = cfg.rew_w_leg_joint_vel * torch.sum(torch.square(joint_vel), dim=1)
        # 6) 双脚接触（站立参考接触恒 [1,1]）
        rew_contact = (
            cfg.rew_w_contact_match * contact_weight_scale * torch.sum(in_contact, dim=1)
        )
        # Contact alone cannot distinguish a planted foot from a sole sliding under tangential
        # load. Use the standard Isaac Lab contact-conditioned planar foot-speed penalty.
        foot_speed_xy = torch.linalg.vector_norm(feet_lin_vel_xy, dim=2)
        rew_feet_slide = cfg.rew_w_feet_slide * torch.sum(foot_speed_xy * in_contact, dim=1)
        # Keep the lift cost finite and independent of the diagnostic recovery state. Large
        # disturbances can still justify a short capture step through survival/fall avoidance.
        rew_feet_airborne = cfg.rew_w_feet_airborne * torch.sum(1.0 - in_contact, dim=1)
        # 7) 正则：力矩 / 关节加速度 / 腿动作率 / 腿动作加速度
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
        # 8) 脖子模仿头部命令参考角 + 速度惩罚 + 脖子动作率/加速度
        rew_neck_pos = cfg.rew_w_neck_joint_pos * torch.sum(torch.square(neck_joint_pos - neck_ref), dim=1)
        rew_neck_vel = cfg.rew_w_neck_joint_vel * torch.sum(torch.square(neck_joint_vel), dim=1)
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
            "feet_slide": rew_feet_slide,
            "feet_airborne": rew_feet_airborne,
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
        total_reward = torch.stack(list(reward_terms.values()), dim=0).sum(dim=0) * self.step_dt
        for key, value in reward_terms.items():
            self._episode_sums[key] += value * self.step_dt

        # Parent metrics are allocated by OceanisaaclabWalkEnv but its reward function is not
        # called in this subclass. Update their physical equivalents here.
        self._metric_left_contact += in_contact[:, 1]
        self._metric_right_contact += in_contact[:, 0]
        self._metric_double_support += torch.prod(in_contact, dim=1)
        self._metric_steps += 1.0
        self._metric_vel_error += torch.linalg.vector_norm(lin_vel[:, :2], dim=1)
        self._metric_yaw_error += torch.abs(ang_vel[:, 2])
        self._metric_neck_tracking_sq += torch.mean(
            torch.square(neck_joint_pos - neck_ref), dim=1
        )

        history_valid = self._stand_metric_steps > 0.0
        contact_count = torch.sum(in_contact, dim=1)
        single_support = (contact_count == 1.0).float()
        liftoff = history_valid.unsqueeze(1) & (self._stand_previous_contact > 0.5) & (in_contact < 0.5)
        touchdown = (
            (self._stand_previous_contact < 0.5)
            & (in_contact > 0.5)
            & self._stand_foot_airborne
        )
        liftoff_count = torch.sum(liftoff.float(), dim=1)
        foot_xy = self.robot.data.body_pos_w.torch[:, self._feet_body_ids, :2]
        # Update per-foot airborne duration before touchdown classification.
        self._stand_airborne_time = torch.where(
            (~in_contact.bool()) & self._stand_foot_airborne,
            self._stand_airborne_time + self.step_dt,
            self._stand_airborne_time,
        )
        self._stand_airborne_time = torch.where(liftoff, torch.zeros_like(self._stand_airborne_time), self._stand_airborne_time)
        self._stand_liftoff_xy = torch.where(
            liftoff.unsqueeze(-1), foot_xy, self._stand_liftoff_xy
        )
        quiet_liftoff = liftoff & ~recovery_active.unsqueeze(1)
        self._stand_liftoff_recovery = torch.where(
            liftoff,
            recovery_active.unsqueeze(1).expand_as(liftoff),
            self._stand_liftoff_recovery,
        )
        touchdown_distance = torch.linalg.vector_norm(
            foot_xy - self._stand_liftoff_xy, dim=2
        )
        recovery_touchdown = touchdown & (
            self._stand_liftoff_recovery
            | recovery_active.unsqueeze(1).expand_as(touchdown)
        )
        self._stand_metric_steps += 1.0
        self._metric_recovery_signal += recovery_signal
        self._metric_recovery_state += recovery_active.float()
        self._metric_recovery_steps += recovery_active.float()
        self._metric_recovery_single_support += single_support * recovery_active.float()
        self._metric_liftoff_events += liftoff_count
        self._metric_recovery_liftoff_events += liftoff_count * recovery_active.float()
        self._metric_touchdown_events += torch.sum(touchdown.float(), dim=1)
        self._metric_touchdown_step_distance += torch.sum(
            touchdown_distance * touchdown.float(), dim=1
        )
        self._metric_recovery_touchdown_events += torch.sum(
            recovery_touchdown.float(), dim=1
        )
        self._metric_recovery_touchdown_step_distance += torch.sum(
            touchdown_distance * recovery_touchdown.float(), dim=1
        )
        capture_touchdown = recovery_touchdown & (
            (self._stand_airborne_time >= cfg.recovery_capture_min_air_time_s)
            & (touchdown_distance >= cfg.recovery_capture_min_step_distance_m)
        )
        self._metric_recovery_capture_steps += torch.sum(capture_touchdown.float(), dim=1)
        self._metric_quiet_liftoff_events += torch.sum(quiet_liftoff.float(), dim=1)
        quiet_contact = (
            self._stand_previous_contact.bool()
            & in_contact.bool()
            & ~recovery_active.unsqueeze(1)
            & history_valid.unsqueeze(1)
        )
        self._metric_quiet_foot_drift += torch.sum(
            torch.linalg.vector_norm(foot_xy - self._stand_previous_foot_xy, dim=2)
            * quiet_contact.float(),
            dim=1,
        )
        clean_episode = ~self._stand_disturbance_enabled
        self._metric_clean_steps += clean_episode.float()
        self._metric_clean_liftoff_events += liftoff_count * clean_episode.float()
        self._metric_clean_double_support += torch.prod(in_contact, dim=1) * clean_episode.float()

        relative_foot_xy, relative_foot_yaw = self._stance_relative_pose()
        nominal_xy = self._stand_recovery_resets.nominal_foot_relative_xy.unsqueeze(0)
        width_error = torch.abs(relative_foot_xy[:, 1] - nominal_xy[:, 1])
        stagger_error = torch.abs(relative_foot_xy[:, 0] - nominal_xy[:, 0])
        yaw_error = torch.abs(
            wrap_angle(
                relative_foot_yaw
                - self._stand_recovery_resets.nominal_foot_relative_yaw
            )
        )
        self._metric_stance_width_error += width_error
        self._metric_stance_stagger_error += stagger_error
        self._metric_stance_yaw_error += yaw_error

        displaced = self._stand_displaced_reset
        displaced_active = displaced & ~self._stand_displaced_recovery_success
        self._stand_displaced_recovery_elapsed += displaced_active.float() * self.step_dt
        canonical = (
            (width_error <= cfg.stance_recovery_width_tolerance_m)
            & (stagger_error <= cfg.stance_recovery_stagger_tolerance_m)
            & (yaw_error <= cfg.stance_recovery_yaw_tolerance_rad)
        )
        physically_stable = (
            (torch.prod(in_contact, dim=1) > 0.5)
            & (
                torch.linalg.vector_norm(projected_gravity[:, :2], dim=1)
                <= cfg.stance_recovery_projected_gravity_xy_max
            )
            & (
                torch.linalg.vector_norm(lin_vel[:, :2], dim=1)
                <= cfg.stance_recovery_lin_vel_xy_max
            )
            & (
                torch.linalg.vector_norm(ang_vel[:, :2], dim=1)
                <= cfg.stance_recovery_ang_vel_xy_max
            )
            & (
                torch.sqrt(torch.mean(torch.square(joint_vel), dim=1))
                <= cfg.stance_recovery_joint_vel_rms_max
            )
        )
        recovery_stable = displaced_active & canonical & physically_stable
        self._stand_displaced_recovery_hold = torch.where(
            recovery_stable,
            self._stand_displaced_recovery_hold + self.step_dt,
            torch.zeros_like(self._stand_displaced_recovery_hold),
        )
        completed = displaced_active & (
            self._stand_displaced_recovery_hold >= cfg.stance_recovery_stable_hold_s
        )
        self._stand_displaced_recovery_success |= completed
        self._stand_displaced_recovery_time = torch.where(
            completed,
            self._stand_displaced_recovery_elapsed,
            self._stand_displaced_recovery_time,
        )
        self._metric_displaced_steps += displaced.float()
        self._metric_displaced_liftoff_events += liftoff_count * displaced.float()
        self._metric_displaced_double_support += torch.prod(in_contact, dim=1) * displaced.float()
        self._stand_foot_airborne |= liftoff
        self._stand_foot_airborne &= ~touchdown
        self._stand_airborne_time = torch.where(touchdown, torch.zeros_like(self._stand_airborne_time), self._stand_airborne_time)
        self._stand_liftoff_recovery &= ~touchdown
        self._stand_previous_contact.copy_(in_contact)
        self._stand_previous_foot_xy.copy_(foot_xy)
        return total_reward

    # ------------------------------------------------------------------
    # reset: 复用行走 reset 的统计/执行器随机化/扰动/头命令采样，再以单一站立参考
    # 同步覆盖物理状态、动作历史、延迟环和 FOH/LPF 目标。
    # ------------------------------------------------------------------
    def _resample_stand_controls(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Resample time-varying perpetual torso/head commands over their full training range."""
        n = len(env_ids)
        self._torso_commands[env_ids, 0] = sample_uniform(
            *self.cfg.torso_command_h_range, (n,), self.device
        )
        self._torso_commands[env_ids, 1] = sample_uniform(
            *self.cfg.torso_command_pitch_range, (n,), self.device
        )
        self._torso_commands[env_ids, 2] = sample_uniform(
            *self.cfg.torso_command_yaw_range, (n,), self.device
        )
        self._torso_commands[env_ids, 3] = sample_uniform(
            *self.cfg.torso_command_roll_range, (n,), self.device
        )
        head_scale = self._head_command_curriculum_scale()
        self._head_commands[env_ids, 0] = head_scale * self._sample_range(
            self.cfg.head_command_dh_range, (n,)
        )
        self._head_commands[env_ids, 1] = head_scale * self._sample_range(
            self.cfg.head_command_pitch_range, (n,)
        )
        self._head_commands[env_ids, 2] = head_scale * self._sample_range(
            self.cfg.head_command_yaw_range, (n,)
        )
        self._head_commands[env_ids, 3] = head_scale * self._sample_range(
            self.cfg.head_command_roll_range, (n,)
        )
        zero = torch.rand(n, device=self.device) < self.cfg.stand_zero_command_prob
        self._torso_commands[env_ids[zero]] = 0.0
        self._head_commands[env_ids[zero]] = 0.0
        self._control_resample_left[env_ids] = self._sample_range(
            self.cfg.control_resample_interval_s, (n,)
        )
        self._stand_command_grace_left[env_ids] = self.cfg.recovery_command_grace_s
        return zero

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if not hasattr(self, "_stand_pose"):
            return  # 父类构造期首次 reset：本类缓冲尚未建立

        metric_steps = self._stand_metric_steps[env_ids].clamp_min(1.0)
        log = self.extras.setdefault("log", {})
        log["Metrics/recovery_signal_mean"] = torch.mean(
            self._metric_recovery_signal[env_ids] / metric_steps
        )
        log["Metrics/recovery_active_state_mean"] = torch.mean(
            self._metric_recovery_state[env_ids] / metric_steps
        )
        log["Metrics/recovery_active_rate"] = torch.mean(
            self._metric_recovery_steps[env_ids] / metric_steps
        )
        log["Metrics/recovery_single_support_rate"] = (
            self._metric_recovery_single_support[env_ids].sum()
            / self._metric_recovery_steps[env_ids].sum().clamp_min(1.0)
        )
        log["Metrics/liftoff_events_per_episode"] = torch.mean(
            self._metric_liftoff_events[env_ids]
        )
        log["Metrics/recovery_liftoff_events_per_episode"] = torch.mean(
            self._metric_recovery_liftoff_events[env_ids]
        )
        touchdown_events = self._metric_touchdown_events[env_ids].sum().clamp_min(1.0)
        recovery_touchdown_events = self._metric_recovery_touchdown_events[
            env_ids
        ].sum().clamp_min(1.0)
        log["Metrics/touchdown_events_per_episode"] = torch.mean(
            self._metric_touchdown_events[env_ids]
        )
        log["Metrics/touchdown_step_distance_cm"] = (
            100.0 * self._metric_touchdown_step_distance[env_ids].sum() / touchdown_events
        )
        log["Metrics/recovery_touchdown_events_per_episode"] = torch.mean(
            self._metric_recovery_touchdown_events[env_ids]
        )
        log["Metrics/recovery_touchdown_step_distance_cm"] = (
            100.0
            * self._metric_recovery_touchdown_step_distance[env_ids].sum()
            / recovery_touchdown_events
        )
        log["Metrics/recovery_capture_steps_per_episode"] = torch.mean(
            self._metric_recovery_capture_steps[env_ids]
        )
        log["Metrics/quiet_liftoff_events_per_episode"] = torch.mean(
            self._metric_quiet_liftoff_events[env_ids]
        )
        log["Metrics/quiet_foot_drift_cm"] = (
            100.0 * torch.mean(self._metric_quiet_foot_drift[env_ids])
        )
        clean_steps = self._metric_clean_steps[env_ids].sum().clamp_min(1.0)
        log["Metrics/clean_episode_fraction"] = torch.mean(
            self._metric_clean_steps[env_ids] / metric_steps
        )
        log["Metrics/clean_liftoff_events_per_episode"] = torch.mean(
            self._metric_clean_liftoff_events[env_ids]
        )
        log["Metrics/clean_double_support_rate"] = (
            self._metric_clean_double_support[env_ids].sum() / clean_steps
        )
        total_metric_steps = self._stand_metric_steps[env_ids].sum().clamp_min(1.0)
        log["Metrics/stance_width_error_cm"] = (
            100.0 * self._metric_stance_width_error[env_ids].sum() / total_metric_steps
        )
        log["Metrics/stance_stagger_error_cm"] = (
            100.0 * self._metric_stance_stagger_error[env_ids].sum() / total_metric_steps
        )
        log["Metrics/stance_yaw_error_rad"] = (
            self._metric_stance_yaw_error[env_ids].sum() / total_metric_steps
        )
        displaced = self._stand_displaced_reset[env_ids]
        displaced_count = displaced.float().sum().clamp_min(1.0)
        displaced_steps = self._metric_displaced_steps[env_ids].sum().clamp_min(1.0)
        successful = displaced & self._stand_displaced_recovery_success[env_ids]
        successful_count = successful.float().sum().clamp_min(1.0)
        log["Metrics/displaced_reset_fraction"] = displaced.float().mean()
        log["Metrics/displaced_recovery_success_rate"] = (
            successful.float().sum() / displaced_count
        )
        log["Metrics/displaced_recovery_time_s"] = (
            self._stand_displaced_recovery_time[env_ids][successful].sum()
            / successful_count
        )
        log["Metrics/displaced_liftoff_events_per_episode"] = (
            self._metric_displaced_liftoff_events[env_ids].sum() / displaced_count
        )
        log["Metrics/displaced_double_support_rate"] = (
            self._metric_displaced_double_support[env_ids].sum() / displaced_steps
        )
        log["Curriculum/stand_displaced_reset_scale"] = (
            self._displaced_reset_curriculum_scale()
        )
        for metric in (
            self._stand_metric_steps,
            self._metric_recovery_signal,
            self._metric_recovery_state,
            self._metric_recovery_steps,
            self._metric_recovery_single_support,
            self._metric_liftoff_events,
            self._metric_recovery_liftoff_events,
            self._metric_touchdown_events,
            self._metric_touchdown_step_distance,
            self._metric_recovery_touchdown_events,
            self._metric_recovery_touchdown_step_distance,
            self._metric_recovery_capture_steps,
            self._metric_quiet_liftoff_events,
            self._metric_quiet_foot_drift,
            self._metric_clean_steps,
            self._metric_clean_liftoff_events,
            self._metric_clean_double_support,
            self._metric_stance_width_error,
            self._metric_stance_stagger_error,
            self._metric_stance_yaw_error,
            self._metric_displaced_steps,
            self._metric_displaced_liftoff_events,
            self._metric_displaced_double_support,
        ):
            metric[env_ids] = 0.0
        self._stand_previous_contact[env_ids] = 1.0
        self._stand_foot_airborne[env_ids] = False
        self._stand_liftoff_xy[env_ids] = 0.0
        self._stand_liftoff_recovery[env_ids] = False
        self._stand_airborne_time[env_ids] = 0.0
        self._stand_previous_foot_xy[env_ids] = 0.0
        self._recovery_state[env_ids] = 0.0
        self._recovery_active[env_ids] = False
        self._recovery_hold_left[env_ids] = 0.0
        self._stand_displaced_recovery_hold[env_ids] = 0.0
        self._stand_displaced_recovery_elapsed[env_ids] = 0.0
        self._stand_displaced_recovery_success[env_ids] = False
        self._stand_displaced_recovery_time[env_ids] = 0.0
        # The parent reset reinitializes the Table V schedules. Select the next clean or
        # disturbed episode only after logging and clearing the old episode's metrics, so
        # clean metrics are attributed to the episode that generated them.
        self._sample_stand_disturbance_modes(env_ids)
        self._sample_stand_reset_modes(env_ids)
        n = len(env_ids)

        # 站立不平移：loco 命令置 0（path frame 走站立收敛分支）。
        self._commands[env_ids] = 0.0
        self._is_standing[env_ids] = True
        self._phase[env_ids] = 0.0

        # 采样 g_perp；共享零命令 env 同时把 torso/head 命令清零，构成真正的
        # stand/walk 公共切换状态，而不是“腿零命令但头仍随机”。
        zero = self._resample_stand_controls(env_ids)
        displaced = self._stand_displaced_reset[env_ids]
        if torch.any(displaced):
            displaced_ids = env_ids[displaced]
            self._torso_commands[displaced_ids] = 0.0
            self._head_commands[displaced_ids] = 0.0
            self._control_resample_left[displaced_ids] = self.cfg.episode_length_s
            zero[displaced] = True

        # stand_rsi_prob 决定是否从命令对应参考状态出生；其余 env 从公共 neutral
        # 状态出生并学习姿态过渡。零命令无论 RSI 抽样结果如何都保持 URDF q=0 精确姿态。
        rsi = (torch.rand(n, device=self.device) < self.cfg.stand_rsi_prob) & ~displaced
        init_torso_commands = torch.zeros(n, 4, device=self.device)
        init_head_commands = torch.zeros(n, 4, device=self.device)
        init_torso_commands[rsi] = self._torso_commands[env_ids[rsi]]
        init_head_commands[rsi] = self._head_commands[env_ids[rsi]]

        stand_ref = self._stand_pose.sample_reference(
            init_torso_commands, init_head_commands
        )
        initial_leg_pos = stand_ref["joint_pos"].clone()
        noisy_rsi = rsi & ~zero
        if torch.any(noisy_rsi) and self.cfg.stand_rsi_joint_pos_noise > 0.0:
            initial_leg_pos[noisy_rsi] += torch.empty_like(initial_leg_pos[noisy_rsi]).uniform_(
                -self.cfg.stand_rsi_joint_pos_noise,
                self.cfg.stand_rsi_joint_pos_noise,
            )
        recovery_reset = self._stand_recovery_resets.sample(
            int(displaced.sum().item()), self._displaced_reset_curriculum_scale()
        )
        if torch.any(displaced):
            initial_leg_pos[displaced] = recovery_reset["joint_pos"]
            self._stand_displaced_reset_category[env_ids[displaced]] = recovery_reset[
                "category"
            ]

        # Convert the initialized pose through the same bounded action mapping used at runtime.
        # Writing the representable targets back to simulation makes q, action history, FOH and
        # LPF state exactly describe one state even at command-grid extremes.
        leg_actions = torch.clamp(
            (initial_leg_pos - self._default_leg_joint_pos[env_ids]) / self._joint_ranges,
            -1.0,
            1.0,
        )
        leg_target = self._default_leg_joint_pos[env_ids] + self._joint_ranges * leg_actions
        soft_leg_limits = self._soft_leg_joint_pos_limits[env_ids]
        leg_target = torch.clamp(leg_target, soft_leg_limits[..., 0], soft_leg_limits[..., 1])
        leg_actions = torch.clamp(
            (leg_target - self._default_leg_joint_pos[env_ids]) / self._joint_ranges,
            -1.0,
            1.0,
        )

        neck_ref = self._neck_head_map.sample(init_head_commands)
        neck_actions = torch.clamp(
            (neck_ref - self._default_neck_joint_pos[env_ids]) / self._neck_joint_ranges,
            -1.0,
            1.0,
        )
        neck_target = self._default_neck_joint_pos[env_ids] + self._neck_joint_ranges * neck_actions
        soft_neck_limits = self._soft_neck_joint_pos_limits[env_ids]
        neck_target = torch.clamp(neck_target, soft_neck_limits[..., 0], soft_neck_limits[..., 1])
        neck_actions = torch.clamp(
            (neck_target - self._default_neck_joint_pos[env_ids]) / self._neck_joint_ranges,
            -1.0,
            1.0,
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._leg_dof_idx] = leg_target
        joint_pos[:, self._neck_dof_idx] = neck_target
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :2] = root_pose[:, :2] + self.scene.env_origins[env_ids, :2]
        if torch.any(displaced):
            root_pose[displaced, :2] += recovery_reset["base_pos_xy"]
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + self._stand_pose.base_height \
            + init_torso_commands[:, 0]
        root_pose[:, 3:7] = quat_from_euler_xyz(
            init_torso_commands[:, 3],
            init_torso_commands[:, 1],
            init_torso_commands[:, 2],
        )
        root_vel = torch.zeros_like(self.robot.data.default_root_vel.torch[env_ids])
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # 动作管线所有历史和目标使用同一初始化姿态，消除 walking RSI 延迟动作重放。
        reset_actions = torch.zeros(n, self.cfg.action_space, device=self.device)
        reset_actions[:, self._leg_action_slice] = leg_actions
        reset_actions[:, self._neck_action_slice] = neck_actions
        self._actions[env_ids] = reset_actions
        self._previous_actions[env_ids] = reset_actions
        self._prev_prev_actions[env_ids] = reset_actions
        self._action_history[env_ids] = reset_actions.unsqueeze(1).expand(
            -1, self._action_history.shape[1], -1
        )
        self._processed_actions[env_ids, : len(self._leg_dof_idx)] = leg_target
        self._target_joint_pos[env_ids] = leg_target
        self._prev_target_joint_pos[env_ids] = leg_target
        self._filtered_joint_target[env_ids] = leg_target
        self._neck_target[env_ids] = neck_target
        self._prev_neck_target[env_ids] = neck_target
        self._filtered_neck_target[env_ids] = neck_target
        self._last_tau_m[env_ids] = 0.0
        self._applied_leg_torque[env_ids] = 0.0
        self._substep = 0

        # 参考 xy 与腿角来自同一个 head+torso 全身 CoM 平衡参考。
        pf_offset = stand_ref["base_pos_pf"].clone()
        init_base_yaw_pf = stand_ref["base_yaw_pf"]
        init_feet_heading = (
            init_torso_commands[:, 2] + self._head_yaw_offset - init_base_yaw_pf
        )
        init_feet_heading = torch.atan2(
            torch.sin(init_feet_heading),
            torch.cos(init_feet_heading),
        )
        base_xy = root_pose[:, :2].clone()
        if torch.any(displaced):
            pf_offset[displaced] = recovery_reset["base_pos_pf"]
            init_feet_heading[displaced] = recovery_reset["feet_heading_yaw"]
        cos_y, sin_y = torch.cos(init_feet_heading), torch.sin(init_feet_heading)
        path_xy = base_xy.clone()
        path_xy[:, 0] -= pf_offset[:, 0] * cos_y - pf_offset[:, 1] * sin_y
        path_xy[:, 1] -= pf_offset[:, 0] * sin_y + pf_offset[:, 1] * cos_y
        self._path_frame.reset(env_ids, path_xy, init_feet_heading)
