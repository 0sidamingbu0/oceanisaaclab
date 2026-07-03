# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""路线 B（BDX 式参考轨迹模仿）训练环境。

继承路线 A 环境的场景/观测/命令采样/推扰/领域随机化等全部基础设施（观测 41 维，
无足底接触量、sim2sim 部署链路一致），仅替换两处核心：

1. ``_get_rewards``：改为参考模仿奖励——按命令与相位从步态库采样参考帧，奖励以
   关节角匹配为主导 + 接触时序匹配 + 基座高度/姿态匹配 + 速度命令跟踪 + 少量正则。
   路线 A 的 21 项手工塑形联动栈（fwd_gate / instability gate / phase_contact /
   swing_contact / single_support / feet_clearance / feet_air_time / stand_still）
   全部删除。
2. ``_reset_idx``：追加参考态初始化（RSI）——大部分 env 直接从随机相位的参考帧
   出发（关节角/基座高度姿态/基座速度），绕过「从静止起步」的最难学习阶段。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from .oceanisaaclab_env import OceanisaaclabEnv
from .oceanisaaclab_walk_env_cfg import OceanisaaclabWalkEnvCfg
from .reference_gait import ReferenceGait


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
        # 覆盖父类的奖励分项统计（模仿范式的分项集合完全不同）
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "terminated",
                "imit_joint_pos",
                "imit_joint_vel",
                "imit_contact",
                "imit_height",
                "imit_orient",
                "track_lin_vel",
                "track_ang_vel",
                "action_rate",
                "feet_slide",
            ]
        }

    # ------------------------------------------------------------------
    # rewards: reference imitation
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ref = self._reference_gait.sample(self._commands, self._gait_phase())

        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]
        base_height = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        proj_g = self.robot.data.projected_gravity_b.torch
        root_lin_vel_b = self.robot.data.root_lin_vel_b.torch
        root_ang_vel_b = self.robot.data.root_ang_vel_b.torch
        in_contact = (
            self.contact_sensor.data.current_contact_time.torch[:, self._feet_contact_ids] > 0.0
        ).float()
        feet_lin_vel = self.robot.data.body_lin_vel_w.torch[:, self._feet_body_ids, :2]

        # 1) 关节角匹配（主导项）：参考帧关节角为绝对角（default_joint_pos 恒 0）
        joint_pos_err = torch.sum(torch.square(joint_pos - ref["joint_pos"]), dim=1)
        rew_imit_joint_pos = self.cfg.rew_scale_imit_joint_pos * torch.exp(
            -joint_pos_err / self.cfg.imit_joint_pos_sigma
        )
        # 2) 关节速度匹配（低权重：参考速度来自有限差分）
        joint_vel_err = torch.sum(torch.square(joint_vel - ref["joint_vel"]), dim=1)
        rew_imit_joint_vel = self.cfg.rew_scale_imit_joint_vel * torch.exp(
            -joint_vel_err / self.cfg.imit_joint_vel_sigma
        )
        # 3) 接触时序匹配：参考 schedule 是（可能插值出的分数）0/1，按逐脚一致度打分
        contact_match = 1.0 - torch.mean(torch.abs(in_contact - ref["feet_contact"]), dim=1)
        rew_imit_contact = self.cfg.rew_scale_imit_contact * contact_match
        # 4) 基座高度匹配
        height_err = torch.square(base_height - ref["base_height"])
        rew_imit_height = self.cfg.rew_scale_imit_height * torch.exp(-height_err / self.cfg.imit_height_sigma)
        # 5) 基座姿态匹配（proj_g 对参考前倾姿态；行走前倾/站立竖直全部由参考决定）
        orient_err = torch.sum(torch.square(proj_g - ref["proj_g"]), dim=1)
        rew_imit_orient = self.cfg.rew_scale_imit_orient * torch.exp(-orient_err / self.cfg.imit_orient_sigma)
        # 6) 速度命令跟踪：头部系约定与路线 A 一致（head-forward = fsign * body-x）
        act_planar = self.cfg.forward_vx_sign * root_lin_vel_b[:, :2]
        lin_vel_err = torch.sum(torch.square(self._commands[:, :2] - act_planar), dim=1)
        rew_track_lin_vel = self.cfg.rew_scale_walk_track_lin_vel * torch.exp(
            -lin_vel_err / self.cfg.walk_lin_vel_track_sigma
        )
        ang_vel_err = torch.square(self._commands[:, 2] - root_ang_vel_b[:, 2])
        rew_track_ang_vel = self.cfg.rew_scale_walk_track_ang_vel * torch.exp(
            -ang_vel_err / self.cfg.walk_ang_vel_track_sigma
        )
        # 7) 正则项：存活 / 摔倒 / 动作变化率 / 接地滑移
        rew_alive = self.cfg.rew_scale_walk_alive * (1.0 - self.reset_terminated.float())
        rew_terminated = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_action_rate = self.cfg.rew_scale_walk_action_rate * torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        slide = torch.sum(torch.sum(torch.square(feet_lin_vel), dim=2) * in_contact, dim=1)
        rew_feet_slide = self.cfg.rew_scale_walk_feet_slide * slide

        reward_terms = {
            "alive": rew_alive,
            "terminated": rew_terminated,
            "imit_joint_pos": rew_imit_joint_pos,
            "imit_joint_vel": rew_imit_joint_vel,
            "imit_contact": rew_imit_contact,
            "imit_height": rew_imit_height,
            "imit_orient": rew_imit_orient,
            "track_lin_vel": rew_track_lin_vel,
            "track_ang_vel": rew_track_ang_vel,
            "action_rate": rew_action_rate,
            "feet_slide": rew_feet_slide,
        }
        total_reward = torch.stack(list(reward_terms.values()), dim=0).sum(dim=0)
        for key, value in reward_terms.items():
            self._episode_sums[key] += value
        return total_reward

    # ------------------------------------------------------------------
    # reset: reference-state initialization (RSI)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        # 父类完成命令采样、相位偏置采样、默认状态写入与统计上报
        super()._reset_idx(env_ids)
        if not hasattr(self, "_reference_gait"):
            return  # __init__ 里父类构造期间的首次全量 reset：步态库尚未加载

        rsi = torch.rand(len(env_ids), device=self.device) < self.cfg.rsi_prob
        if not torch.any(rsi):
            return
        rsi_ids = env_ids[rsi]
        # episode_length_buf 已被父类清 0 → 当前相位 = 相位偏置
        phase = self._gait_phase_offset[rsi_ids]
        ref = self._reference_gait.sample(self._commands[rsi_ids], phase)

        # 关节状态：参考帧关节角（加小噪声）+ 参考关节速度；脖子保持默认
        joint_pos = self.robot.data.default_joint_pos.torch[rsi_ids].clone()
        leg_pos = ref["joint_pos"] + torch.empty_like(ref["joint_pos"]).uniform_(
            -self.cfg.rsi_joint_pos_noise, self.cfg.rsi_joint_pos_noise
        )
        soft_limits = self.robot.data.soft_joint_pos_limits.torch[rsi_ids][:, self._leg_dof_idx]
        joint_pos[:, self._leg_dof_idx] = torch.clamp(leg_pos, soft_limits[..., 0], soft_limits[..., 1])
        joint_vel = self.robot.data.default_joint_vel.torch[rsi_ids].clone()
        joint_vel[:, self._leg_dof_idx] = ref["joint_vel"]

        # 基座位姿：参考高度；朝向沿用 default_root_pose（与非 RSI env / 路线 A spawn 一致）。
        # 注意：不要手工覆盖 root_pose[:,3:7] 四元数——诊断实测该手写 (cos,0,sin,0) 会把 90%
        # 的 RSI env 一进 reset 就翻成 >49° 倾倒(proj_g_z→+1，绕X翻 180°)，随即被 not_upright
        # 判定成片摔倒。根因是这里 root_pose 四元数的存储顺序与手写假设不一致（cfg
        # init rot=(0,0,0,1) 即单位四元数 → 该管线按 xyzw 存），且手写还丢弃了 default 朝向。
        # 参考 base_pitch 仅 ±5°，对 RSI 可忽略（前倾姿态由 imit_orient 奖励在 episode 中学到），
        # 直接保留 default 朝向即可保证正立且方向正确。
        root_pose = self.robot.data.default_root_pose.torch[rsi_ids].clone()
        root_pose[:, :3] = self.scene.env_origins[rsi_ids]
        root_pose[:, 2] += ref["base_height"]
        # 基座速度：参考 body 系线速度（reset 时 yaw=0，body≈world）+ 命令 yaw 角速度
        root_vel = self.robot.data.default_root_vel.torch[rsi_ids].clone()
        root_vel[:, :3] = ref["lin_vel_b"]
        root_vel[:, 3:5] = 0.0
        root_vel[:, 5] = self._commands[rsi_ids, 2]

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=rsi_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=rsi_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=rsi_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=rsi_ids)
