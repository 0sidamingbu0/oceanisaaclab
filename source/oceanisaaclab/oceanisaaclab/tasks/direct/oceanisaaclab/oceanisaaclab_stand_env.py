# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 论文复刻）站立（perpetual）训练环境。

论文 divide-and-conquer：periodic 行走策略之外单独训练 perpetual 站立策略
π(a | s, g_perp)，**无相位**、脚不迈步，命令 g_perp = (Δh_head, Δθ_head, h_torso,
θ_torso)（式 5）。继承行走环境 OceanisaaclabWalkEnv，复用附录 B 执行器模型、表 V 扰动、
域随机化、torso 等效触地终止、非对称 critic、path frame、脖子位置伺服、脖子/头部命令映射；
覆盖：观测（去相位谐波、命令换 torso4+head4）、奖励（静态姿态模仿）、reset（站立命令
采样 + 从标称站姿出生）。行走的动作管线 / 执行器模型 / 脖子通路完全沿用。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz, sample_uniform

from .oceanisaaclab_stand_env_cfg import OceanisaaclabStandEnvCfg
from .oceanisaaclab_walk_env import OceanisaaclabWalkEnv
from .reference_gait import StandPose


class OceanisaaclabStandEnv(OceanisaaclabWalkEnv):
    cfg: OceanisaaclabStandEnvCfg

    def __init__(self, cfg: OceanisaaclabStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 站立姿态参考库（躯干命令 4-DOF → 腿角）；脖子头部沿用父类 _neck_head_map
        self._stand_pose = StandPose(self.cfg.stand_pose_path, self.device)
        neutral_torso = torch.zeros(1, 4, device=self.device)
        neutral_leg = self._stand_pose.sample(neutral_torso)
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
            self._stand_pose.sample_base_pos_pf(neutral_torso),
            walk_neutral["base_pos_pf"],
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise RuntimeError("Stand/walk zero-command path-frame torso positions do not match.")
        self._torso_commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._torso_command_scale = torch.tensor(
            self.cfg.torso_command_scale, dtype=torch.float, device=self.device
        ).unsqueeze(0)
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
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        ref_leg = self._stand_pose.sample(self._torso_commands)  # (N,10) 站立腿角
        ref_base_pos_pf = self._stand_pose.sample_base_pos_pf(self._torso_commands)
        ref_base_yaw_pf = self._stand_pose.sample_base_yaw_pf(self._torso_commands)
        neck_ref = self._neck_head_map.sample(self._head_commands)  # (N,4)

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
        in_contact = (self._feet_current_contact_time() > 0.0).float()

        # 命令：躯干 (h, pitch, yaw, roll)
        pitch_cmd = self._torso_commands[:, 1]
        roll_cmd = self._torso_commands[:, 3]

        # 1) 躯干 path 系 xy：跟随离线运动学参考。q=0 时躯干相对双脚
        # link 原点中心保留真实前向偏移，不能强制成 path-frame 原点。
        pos_err = torch.sum(torch.square(pos_pf - ref_base_pos_pf), dim=1)
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
        rew_joint_pos = cfg.rew_w_leg_joint_pos * torch.sum(torch.square(joint_pos - ref_leg), dim=1)
        rew_joint_vel = cfg.rew_w_leg_joint_vel * torch.sum(torch.square(joint_vel), dim=1)
        # 6) 双脚接触（站立参考接触恒 [1,1]）
        rew_contact = cfg.rew_w_contact_match * torch.sum(in_contact, dim=1)
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
        rew_action_rate = cfg.rew_w_action_rate * torch.sum(
            torch.square(self._actions[:, self._leg_action_slice]
                         - self._previous_actions[:, self._leg_action_slice]), dim=1
        )
        rew_action_acc = cfg.rew_w_action_acc * torch.sum(
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
        n = len(env_ids)

        # 站立不平移：loco 命令置 0（path frame 走站立收敛分支）。
        self._commands[env_ids] = 0.0
        self._is_standing[env_ids] = True
        self._phase[env_ids] = 0.0

        # 采样 g_perp；共享零命令 env 同时把 torso/head 命令清零，构成真正的
        # stand/walk 公共切换状态，而不是“腿零命令但头仍随机”。
        zero = self._resample_stand_controls(env_ids)

        # stand_rsi_prob 决定是否从命令对应参考状态出生；其余 env 从公共 neutral
        # 状态出生并学习姿态过渡。零命令无论 RSI 抽样结果如何都保持 URDF q=0 精确姿态。
        rsi = torch.rand(n, device=self.device) < self.cfg.stand_rsi_prob
        init_torso_commands = torch.zeros(n, 4, device=self.device)
        init_head_commands = torch.zeros(n, 4, device=self.device)
        init_torso_commands[rsi] = self._torso_commands[env_ids[rsi]]
        init_head_commands[rsi] = self._head_commands[env_ids[rsi]]

        leg_ref = self._stand_pose.sample(init_torso_commands)
        noisy_rsi = rsi & ~zero
        if torch.any(noisy_rsi) and self.cfg.stand_rsi_joint_pos_noise > 0.0:
            leg_ref[noisy_rsi] += torch.empty_like(leg_ref[noisy_rsi]).uniform_(
                -self.cfg.stand_rsi_joint_pos_noise,
                self.cfg.stand_rsi_joint_pos_noise,
            )

        # Convert the initialized pose through the same bounded action mapping used at runtime.
        # Writing the representable targets back to simulation makes q, action history, FOH and
        # LPF state exactly describe one state even at command-grid extremes.
        leg_actions = torch.clamp(
            (leg_ref - self._default_leg_joint_pos[env_ids]) / self._joint_ranges,
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

        # 参考 xy 来自离线运动学躯干轨迹，不把它解释为或替代 CoP。
        pf_offset = self._stand_pose.sample_base_pos_pf(init_torso_commands)
        init_base_yaw_pf = self._stand_pose.sample_base_yaw_pf(init_torso_commands)
        init_feet_heading = (
            init_torso_commands[:, 2] + self._head_yaw_offset - init_base_yaw_pf
        )
        init_feet_heading = torch.atan2(
            torch.sin(init_feet_heading),
            torch.cos(init_feet_heading),
        )
        base_xy = (
            self.scene.env_origins[env_ids, :2]
            + self.robot.data.default_root_pose.torch[env_ids, :2]
        ).clone()
        cos_y, sin_y = torch.cos(init_feet_heading), torch.sin(init_feet_heading)
        path_xy = base_xy.clone()
        path_xy[:, 0] -= pf_offset[:, 0] * cos_y - pf_offset[:, 1] * sin_y
        path_xy[:, 1] -= pf_offset[:, 0] * sin_y + pf_offset[:, 1] * cos_y
        self._path_frame.reset(env_ids, path_xy, init_feet_heading)
