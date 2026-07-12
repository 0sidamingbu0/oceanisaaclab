# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path frame（BDX 论文第 V-A 节 / Fig.4）——参考运动的移动锚定坐标系。

每个 env 维护一个平面坐标系状态 (x, y, yaw)，其中 +x 轴 = 头部前向：

- **行走**（非零命令）：按 path 系速度命令 (v_x, v_y, ω_z) 逐步积分推进；
- **站立**（零命令）：一阶低通收敛到双脚中心位置与双脚平均朝向
  （论文：converges towards the average position and heading of the feet）；
- **最大偏差投影**：path frame 与躯干实际位置/朝向的偏差超阈值时把 path frame
  拉回（论文：f_t is projected to a maximum distance from the current torso
  state），防止摔倒/受扰后参考跑飞、策略被"追不上的参考"惩罚死。

躯干的 path 系 xy 位置与相对 yaw 是策略观测（论文公式 (8) 的 p_P、θ_P），
躯干 path 系位置/朝向模仿是奖励主项（表 I 前两行）——速度跟踪由此隐式实现：
path frame 按命令速度前进，躯干必须贴住它才有奖励。

该动力学在 RL 训练与部署 runtime 必须逐行为一致实现（论文 V-D 节明确要求）。
纯 torch 张量运算、无 isaaclab 依赖，便于部署侧 numpy 复刻。
"""

from __future__ import annotations

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """把角度环绕到 (-π, π]。"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class PathFrame:
    """每 env 一个平面 path frame (x, y, yaw)，yaw 的 +x 轴 = 头部前向。"""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        stand_time_constant: float = 1.0,
        max_pos_deviation: float = 0.25,
        max_yaw_deviation: float = 0.6,
    ):
        self.device = torch.device(device)
        self.pos = torch.zeros(num_envs, 2, device=self.device)  # 世界系 xy
        self.yaw = torch.zeros(num_envs, device=self.device)  # 头部前向朝向
        self.stand_time_constant = stand_time_constant
        self.max_pos_deviation = max_pos_deviation
        self.max_yaw_deviation = max_yaw_deviation

    def reset(self, env_ids: torch.Tensor, base_pos_xy: torch.Tensor, head_yaw: torch.Tensor) -> None:
        """reset 时把 path frame 初始化到躯干出生位姿（论文：起始于躯干状态）。"""
        self.pos[env_ids] = base_pos_xy
        self.yaw[env_ids] = head_yaw

    def step(
        self,
        dt: float,
        commands: torch.Tensor,
        moving: torch.Tensor,
        base_pos_xy: torch.Tensor,
        head_yaw: torch.Tensor,
        feet_center_xy: torch.Tensor,
        feet_heading_yaw: torch.Tensor,
    ) -> None:
        """推进一个控制步。

        Args:
            dt: 控制步长 [s]。
            commands: (N,3) path 系速度命令 (v_x 头前向, v_y 头左向, ω_z)。
            moving: (N,) bool，非零命令 mask（行走积分 / 站立收敛分支）。
            base_pos_xy: (N,2) 躯干世界系 xy。
            head_yaw: (N,) 躯干头部前向朝向（世界系）。
            feet_center_xy: (N,2) 双脚中心世界系 xy（站立收敛目标）。
            feet_heading_yaw: (N,) 双脚平均 heading（世界系，已校准左右 link-frame 偏置）。
        """
        # 行走：按命令积分。命令是 path 系的 → 旋转到世界系再积分。
        cos_y, sin_y = torch.cos(self.yaw), torch.sin(self.yaw)
        dx_w = commands[:, 0] * cos_y - commands[:, 1] * sin_y
        dy_w = commands[:, 0] * sin_y + commands[:, 1] * cos_y
        walk_pos = self.pos + torch.stack((dx_w, dy_w), dim=1) * dt
        walk_yaw = self.yaw + commands[:, 2] * dt
        # 站立：一阶低通收敛到双脚中心 + 双脚平均 heading（论文 V-A）。
        alpha = min(1.0, dt / self.stand_time_constant)
        stand_pos = self.pos + alpha * (feet_center_xy - self.pos)
        stand_yaw = self.yaw + alpha * wrap_angle(feet_heading_yaw - self.yaw)
        m = moving.unsqueeze(1).float()
        self.pos = m * walk_pos + (1.0 - m) * stand_pos
        self.yaw = wrap_angle(moving.float() * walk_yaw + (1.0 - moving.float()) * stand_yaw)
        # 最大偏差投影：把 path frame 拉回躯干附近（位置 + 朝向分别投影）。
        offset = self.pos - base_pos_xy
        dist = torch.norm(offset, dim=1, keepdim=True)
        scale = torch.clamp(self.max_pos_deviation / torch.clamp(dist, min=1e-6), max=1.0)
        self.pos = base_pos_xy + offset * scale
        yaw_err = wrap_angle(self.yaw - head_yaw)
        self.yaw = wrap_angle(
            head_yaw + torch.clamp(yaw_err, -self.max_yaw_deviation, self.max_yaw_deviation)
        )

    def base_in_path_frame(
        self, base_pos_xy: torch.Tensor, head_yaw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """躯干在 path 系中的 xy 位置 (N,2) 与相对 yaw (N,)（观测/奖励用）。"""
        rel = base_pos_xy - self.pos
        cos_y, sin_y = torch.cos(self.yaw), torch.sin(self.yaw)
        x_pf = rel[:, 0] * cos_y + rel[:, 1] * sin_y
        y_pf = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
        return torch.stack((x_pf, y_pf), dim=1), wrap_angle(head_yaw - self.yaw)
