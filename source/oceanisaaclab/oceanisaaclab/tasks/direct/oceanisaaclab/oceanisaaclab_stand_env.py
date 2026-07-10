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
        self._torso_commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._torso_command_scale = torch.tensor(
            self.cfg.torso_command_scale, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        # 覆盖奖励分项统计：换成站立项集合（加 torso_height，去 gait 无关项名保持一致）
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "torso_pos_xy",
                "torso_orient",
                "torso_height",
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
        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        head_yaw = self._head_yaw()
        pos_pf, yaw_pf = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
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

        def assemble(lin_vel_t, ang_vel_t, joint_pos_t, joint_vel_t):
            return torch.cat(
                (
                    pos_pf * self.cfg.pos_pf_scale,
                    yaw_feat,
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
                assemble(lin_vel, ang_vel, joint_pos, joint_vel),
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
            joint_vel_n = joint_vel + torch.randn_like(joint_vel) * self.cfg.noise_joint_vel
            obs = assemble(lin_vel_n, ang_vel_n, joint_pos_hat, joint_vel_n)
        else:
            obs = critic_obs[:, : self.cfg.observation_space]
        return {"policy": obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # rewards: 静态姿态模仿（躯干位姿/高度 + 速度→0 + 腿/脖子模仿 + 双脚接触 + 正则）
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        ref_leg = self._stand_pose.sample(self._torso_commands)  # (N,10) 站立腿角
        neck_ref = self._neck_head_map.sample(self._head_commands)  # (N,4)

        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        base_z = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        head_yaw = self._head_yaw()
        pos_pf, _ = self._path_frame.base_in_path_frame(base_xy, head_yaw)
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        joint_acc = self.robot.data.joint_acc.torch[:, self._leg_dof_idx]
        neck_joint_pos = self.robot.data.joint_pos.torch[:, self._neck_dof_idx]
        neck_joint_vel = self.robot.data.joint_vel.torch[:, self._neck_dof_idx]
        in_contact = (self._feet_current_contact_time() > 0.0).float()

        # 命令：躯干 (h, pitch, yaw, roll)
        h_cmd = self._torso_commands[:, 0]
        pitch_cmd = self._torso_commands[:, 1]
        yaw_cmd = self._torso_commands[:, 2]
        roll_cmd = self._torso_commands[:, 3]

        # 1) 躯干 path 系 xy：站立在双脚中心（参考 sway=0）
        pos_err = torch.sum(torch.square(pos_pf), dim=1)
        rew_pos_xy = cfg.rew_w_torso_pos_xy * torch.exp(-cfg.rew_k_torso_pos_xy * pos_err)
        # 2) 躯干朝向跟命令：参考四元数 = (roll_cmd, pitch_cmd, path_yaw + yaw_cmd − offset)
        base_yaw_ref = self._path_frame.yaw + yaw_cmd - self._head_yaw_offset
        quat_ref = quat_from_euler_xyz(roll_cmd, pitch_cmd, base_yaw_ref)
        orient_err = quat_error_magnitude(self.robot.data.root_quat_w.torch, quat_ref)
        rew_orient = cfg.rew_w_torso_orient * torch.exp(-cfg.rew_k_torso_orient * torch.square(orient_err))
        # 3) 躯干高度跟命令：base_height + h_cmd
        height_ref = self._stand_pose.base_height + h_cmd
        rew_height = cfg.rew_w_torso_height * torch.exp(
            -cfg.rew_k_torso_height * torch.square(base_z - height_ref)
        )
        # 4) 线速度 → 0（惩罚水平/竖直移动）
        rew_lin_xy = cfg.rew_w_lin_vel_xy * torch.exp(-cfg.rew_k_lin_vel * torch.sum(torch.square(lin_vel[:, :2]), dim=1))
        rew_lin_z = cfg.rew_w_lin_vel_z * torch.exp(-cfg.rew_k_lin_vel * torch.square(lin_vel[:, 2]))
        # 5) 角速度 → 0
        rew_ang_xy = cfg.rew_w_ang_vel_xy * torch.exp(-cfg.rew_k_ang_vel * torch.sum(torch.square(ang_vel[:, :2]), dim=1))
        rew_ang_z = cfg.rew_w_ang_vel_z * torch.exp(-cfg.rew_k_ang_vel * torch.square(ang_vel[:, 2]))
        # 6) 腿关节角模仿站立参考 / 关节速度 → 0
        rew_joint_pos = cfg.rew_w_leg_joint_pos * torch.sum(torch.square(joint_pos - ref_leg), dim=1)
        rew_joint_vel = cfg.rew_w_leg_joint_vel * torch.sum(torch.square(joint_vel), dim=1)
        # 7) 双脚接触（站立参考接触恒 [1,1]）
        rew_contact = cfg.rew_w_contact_match * torch.sum(in_contact, dim=1)
        # 8) 正则：力矩 / 关节加速度 / 腿动作率 / 腿动作加速度
        rew_torque = cfg.rew_w_torque * torch.sum(torch.square(self._applied_leg_torque), dim=1)
        rew_joint_acc = cfg.rew_w_joint_acc * torch.sum(torch.square(joint_acc), dim=1)
        rew_action_rate = cfg.rew_w_action_rate * torch.sum(
            torch.square(self._actions[:, self._leg_action_slice]
                         - self._previous_actions[:, self._leg_action_slice]), dim=1
        )
        rew_action_acc = cfg.rew_w_action_acc * torch.sum(
            torch.square(self._actions[:, self._leg_action_slice]
                         - 2.0 * self._previous_actions[:, self._leg_action_slice]
                         + self._prev_prev_actions[:, self._leg_action_slice]), dim=1
        )
        # 9) 脖子模仿头部命令参考角 + 速度惩罚 + 脖子动作率/加速度
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
        # 10) 存活
        rew_survival = cfg.rew_w_survival * (1.0 - self.reset_terminated.float())

        reward_terms = {
            "torso_pos_xy": rew_pos_xy,
            "torso_orient": rew_orient,
            "torso_height": rew_height,
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
    # reset: 复用行走 reset（执行器随机化/扰动/path frame/头命令），覆盖站立特有项：
    # loco 命令置 0、采样躯干命令、从标称站姿(default)出生（覆盖行走的 gait RSI）。
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_stand_pose"):
            return  # 父类构造期首次 reset：本类缓冲尚未建立
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else \
                torch.tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)

        # 站立不平移：loco 命令置 0（path frame 走站立收敛分支）
        self._commands[env_ids] = 0.0
        # 采样躯干命令 g_perp 的 (h, pitch, yaw, roll)；一部分 env 命令全零（标称直立）
        self._torso_commands[env_ids, 0] = sample_uniform(*self.cfg.torso_command_h_range, (n,), self.device)
        self._torso_commands[env_ids, 1] = sample_uniform(*self.cfg.torso_command_pitch_range, (n,), self.device)
        self._torso_commands[env_ids, 2] = sample_uniform(*self.cfg.torso_command_yaw_range, (n,), self.device)
        self._torso_commands[env_ids, 3] = sample_uniform(*self.cfg.torso_command_roll_range, (n,), self.device)
        zero = torch.rand(n, device=self.device) < self.cfg.stand_zero_command_prob
        self._torso_commands[env_ids[zero]] = 0.0

        # 从标称站姿出生：覆盖行走 gait RSI 写入的腿角/速度/base 高度，改回 default 站姿、
        # 正立、零速；脖子出生到头部命令对应参考角。策略从稳定站姿学着达到躯干命令姿态。
        neck_ref = self._neck_head_map.sample(self._head_commands[env_ids])
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._neck_dof_idx] = neck_ref
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] = root_pose[:, :3] + self.scene.env_origins[env_ids]
        root_vel = torch.zeros_like(self.robot.data.default_root_vel.torch[env_ids])
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # 动作管线缓冲对齐标称站姿 / 脖子目标对齐头命令参考角
        self._target_joint_pos[env_ids] = self._default_leg_joint_pos[env_ids]
        self._prev_target_joint_pos[env_ids] = self._default_leg_joint_pos[env_ids]
        self._filtered_joint_target[env_ids] = self._default_leg_joint_pos[env_ids]
        self._neck_target[env_ids] = neck_ref
        # path frame 复位到出生位姿（站立收敛分支，无 sway 偏移）
        self._reset_path_frame(env_ids, None)
