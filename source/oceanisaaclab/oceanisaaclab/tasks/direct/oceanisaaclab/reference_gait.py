# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""参考步态库运行时采样器（路线 B：BDX 式参考轨迹模仿）。

加载 ``scripts/gen_reference_gait.py`` 生成的 npz 步态库，按 (vx, vy, wz) 命令做
三线性插值、按步态相位做环形线性插值，向训练环境提供逐帧参考量（关节角/关节速度/
接触时序/基座姿态与速度）。全部张量运算、无 python 循环，适配数千并行 env。

约定与训练环境一致：
- 命令为头部系（vx 头前向、vy 头左向、wz 逆时针），库内关节角已按
  forward_vx_sign 转换到 base 系，采样端不需要再做符号变换；
- 相位 φ∈[0,1) 与 env ``_gait_phase()`` 相同：右脚 φ∈[0,duty) 支撑、左脚反相；
- 零命令网格点是常量站立帧，命令插值自然给出「命令→0 则步幅→0」的过渡。
"""

from __future__ import annotations

import numpy as np
import torch


class ReferenceGait:
    """npz 步态库的 torch 采样器。"""

    def __init__(self, npz_path: str, device: str | torch.device):
        data = np.load(npz_path, allow_pickle=False)
        self.device = torch.device(device)

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(data[name], dtype=torch.float32, device=self.device)

        # 网格轴（必须单调递增）
        self.vx_grid = tensor("vx_grid")
        self.vy_grid = tensor("vy_grid")
        self.wz_grid = tensor("wz_grid")
        # 相位表 (nx, ny, nz, P, D)
        self.joint_pos = tensor("joint_pos")
        self.joint_vel = tensor("joint_vel")
        self.feet_contact = tensor("feet_contact")
        # 命令常量表 (nx, ny, nz, D) / (nx, ny, nz)
        self.lin_vel_b = tensor("lin_vel_b")
        self.proj_g = tensor("proj_g")
        self.base_height = tensor("base_height")
        self.base_pitch = tensor("base_pitch")
        self.gait_period = float(data["gait_period"])
        self.gait_duty = float(data["gait_duty"])
        self.foot_clearance = (
            float(data["foot_clearance"]) if "foot_clearance" in data.files else 0.0
        )
        self.num_phase = self.joint_pos.shape[3]
        self.joint_names = [str(n) for n in data["joint_names"]]
        # --- BDX 论文完整目标状态 x_t 的新增字段（旧库回退到兼容默认值） ---
        # path 系躯干 xy 轨迹 (nx,ny,nz,P,2)：行走时的左右重心 sway；旧库 → 0
        if "base_pos_pf" in data.files:
            self.base_pos_pf = tensor("base_pos_pf")
        else:
            self.base_pos_pf = torch.zeros(
                (*self.joint_pos.shape[:4], 2), dtype=torch.float32, device=self.device
            )
        # path 系躯干 yaw 振荡 (nx,ny,nz,P)；旧库 → 0
        if "base_yaw_pf" in data.files:
            self.base_yaw_pf = tensor("base_yaw_pf").unsqueeze(-1)  # (nx,ny,nz,P,1)
        else:
            self.base_yaw_pf = torch.zeros(
                (*self.joint_pos.shape[:4], 1), dtype=torch.float32, device=self.device
            )
        # body 系角速度参考 (nx,ny,nz,3)；旧库 → 0（转向参考角速度缺失，仅影响奖励项）
        if "ang_vel_b" in data.files:
            self.ang_vel_b = tensor("ang_vel_b")
        else:
            self.ang_vel_b = torch.zeros_like(self.lin_vel_b)
        # 命令相关相位速率 φ̇ (nx,ny,nz) [1/s]；旧库 → 恒定 1/gait_period
        if "phase_rate" in data.files:
            self.phase_rate = tensor("phase_rate")
        else:
            self.phase_rate = torch.full(
                self.base_height.shape, 1.0 / self.gait_period, dtype=torch.float32, device=self.device
            )
        if "neck_pos" in data.files:
            self.neck_pos = tensor("neck_pos")
            self.neck_vel = tensor("neck_vel")
        else:
            shape = (*self.joint_pos.shape[:4], 4)
            self.neck_pos = torch.zeros(shape, dtype=torch.float32, device=self.device)
            self.neck_vel = torch.zeros_like(self.neck_pos)

    @staticmethod
    def _grid_coords(values: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (下标 i0, 下标 i1, 插值权重 w)，命令超出网格范围时钳制到边界。"""
        clamped = torch.clamp(values, grid[0], grid[-1])
        # searchsorted 找右邻，再回退一格得左邻
        i1 = torch.searchsorted(grid, clamped, right=False).clamp(1, len(grid) - 1)
        i0 = i1 - 1
        w = (clamped - grid[i0]) / (grid[i1] - grid[i0])
        return i0, i1, w

    def sample(self, commands: torch.Tensor, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """按命令 (N,3) 与相位 (N,) 采样参考帧。

        返回 dict：joint_pos/joint_vel (N,10)、feet_contact (N,2)、lin_vel_b/proj_g (N,3)、
        base_height/base_pitch (N,)。
        """
        ix0, ix1, wx = self._grid_coords(commands[:, 0], self.vx_grid)
        iy0, iy1, wy = self._grid_coords(commands[:, 1], self.vy_grid)
        iz0, iz1, wz = self._grid_coords(commands[:, 2], self.wz_grid)
        # 相位环形插值下标
        p = torch.remainder(phase, 1.0) * self.num_phase
        ip0 = p.floor().long() % self.num_phase
        ip1 = (ip0 + 1) % self.num_phase
        wp = (p - p.floor()).unsqueeze(-1)

        def lerp_phase_table(table: torch.Tensor) -> torch.Tensor:
            out = None
            for cx, fx in ((ix0, 1.0 - wx), (ix1, wx)):
                for cy, fy in ((iy0, 1.0 - wy), (iy1, wy)):
                    for cz, fz in ((iz0, 1.0 - wz), (iz1, wz)):
                        weight = (fx * fy * fz).unsqueeze(-1)
                        corner = table[cx, cy, cz]  # (N, P, D)
                        frame = corner[torch.arange(len(p), device=self.device), ip0] * (1.0 - wp)
                        frame = frame + corner[torch.arange(len(p), device=self.device), ip1] * wp
                        out = frame * weight if out is None else out + frame * weight
            return out

        def lerp_table(table: torch.Tensor) -> torch.Tensor:
            expand = table.dim() == 3  # (nx,ny,nz) 标量表
            out = None
            for cx, fx in ((ix0, 1.0 - wx), (ix1, wx)):
                for cy, fy in ((iy0, 1.0 - wy), (iy1, wy)):
                    for cz, fz in ((iz0, 1.0 - wz), (iz1, wz)):
                        weight = fx * fy * fz
                        corner = table[cx, cy, cz]
                        if not expand:
                            weight = weight.unsqueeze(-1)
                        out = corner * weight if out is None else out + corner * weight
            return out

        return {
            "joint_pos": lerp_phase_table(self.joint_pos),
            "joint_vel": lerp_phase_table(self.joint_vel),
            "feet_contact": lerp_phase_table(self.feet_contact),
            "base_pos_pf": lerp_phase_table(self.base_pos_pf),
            "base_yaw_pf": lerp_phase_table(self.base_yaw_pf).squeeze(-1),
            "lin_vel_b": (
                lerp_phase_table(self.lin_vel_b) if self.lin_vel_b.dim() == 5 else lerp_table(self.lin_vel_b)
            ),
            "ang_vel_b": (
                lerp_phase_table(self.ang_vel_b) if self.ang_vel_b.dim() == 5 else lerp_table(self.ang_vel_b)
            ),
            "proj_g": lerp_table(self.proj_g),
            "base_height": lerp_table(self.base_height),
            "base_pitch": lerp_table(self.base_pitch),
            "phase_rate": lerp_table(self.phase_rate),
            "neck_pos": lerp_phase_table(self.neck_pos),
            "neck_vel": lerp_phase_table(self.neck_vel),
        }

    def sample_phase_rate(self, commands: torch.Tensor) -> torch.Tensor:
        """仅采样相位速率 φ̇ (N,)（每步积分相位用，避免整帧采样开销）。"""
        ix0, ix1, wx = self._grid_coords(commands[:, 0], self.vx_grid)
        iy0, iy1, wy = self._grid_coords(commands[:, 1], self.vy_grid)
        iz0, iz1, wz = self._grid_coords(commands[:, 2], self.wz_grid)
        out = None
        for cx, fx in ((ix0, 1.0 - wx), (ix1, wx)):
            for cy, fy in ((iy0, 1.0 - wy), (iy1, wy)):
                for cz, fz in ((iz0, 1.0 - wz), (iz1, wz)):
                    corner = self.phase_rate[cx, cy, cz] * (fx * fy * fz)
                    out = corner if out is None else out + corner
        return out


class NeckHeadMap:
    """脖子/头部参考映射的 torch 采样器（``scripts/gen_neck_head_map.py`` 生成）。

    头部命令 4-DOF (Δh 头高, pitch 点头, yaw 摇头, roll 歪头) → 4 个脖子关节参考角
    (neck_n1..n4)。因 base 固定时脖子与脚 IK 解耦，头姿不进步态库网格（会维度爆炸），
    单独存这张 4D 表，运行时四线性插值给出脖子模仿奖励的目标角。纯张量、无循环。
    """

    def __init__(self, npz_path: str, device: str | torch.device):
        data = np.load(npz_path, allow_pickle=False)
        self.device = torch.device(device)

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(data[name], dtype=torch.float32, device=self.device)

        self.dh_grid = tensor("dh_grid")
        self.pitch_grid = tensor("pitch_grid")
        self.yaw_grid = tensor("yaw_grid")
        self.roll_grid = tensor("roll_grid")
        self.neck_pos = tensor("neck_pos")  # (nh, npitch, nyaw, nroll, 4)
        self.neck_joint_names = [str(n) for n in data["neck_joint_names"]]
        self.num_neck = self.neck_pos.shape[-1]

    def sample(self, head_cmd: torch.Tensor) -> torch.Tensor:
        """按头部命令 (N,4)=(Δh, pitch, yaw, roll) 四线性插值返回脖子参考角 (N,4)。"""
        ih0, ih1, wh = ReferenceGait._grid_coords(head_cmd[:, 0], self.dh_grid)
        ip0, ip1, wp = ReferenceGait._grid_coords(head_cmd[:, 1], self.pitch_grid)
        iy0, iy1, wy = ReferenceGait._grid_coords(head_cmd[:, 2], self.yaw_grid)
        ir0, ir1, wr = ReferenceGait._grid_coords(head_cmd[:, 3], self.roll_grid)
        out = None
        for ch, fh in ((ih0, 1.0 - wh), (ih1, wh)):
            for cp, fp in ((ip0, 1.0 - wp), (ip1, wp)):
                for cy, fy in ((iy0, 1.0 - wy), (iy1, wy)):
                    for cr, fr in ((ir0, 1.0 - wr), (ir1, wr)):
                        weight = (fh * fp * fy * fr).unsqueeze(-1)
                        corner = self.neck_pos[ch, cp, cy, cr]  # (N, 4)
                        out = corner * weight if out is None else out + corner * weight
        return out


class StandPose:
    """站立姿态参考的 torch 采样器（``scripts/gen_stand_pose.py`` 生成）。

    perpetual 站立策略：躯干命令 4-DOF (h_torso 高度偏移, pitch, yaw, roll) → 10 个腿关节
    参考角。无相位、脚不迈步（双脚固定地面）。头部脖子角由 NeckHeadMap 另行提供（解耦）。
    四线性插值，纯张量、无循环。
    """

    def __init__(self, npz_path: str, device: str | torch.device):
        data = np.load(npz_path, allow_pickle=False)
        self.device = torch.device(device)

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(data[name], dtype=torch.float32, device=self.device)

        self.h_grid = tensor("torso_h_grid")
        self.pitch_grid = tensor("torso_pitch_grid")
        self.yaw_grid = tensor("torso_yaw_grid")
        self.roll_grid = tensor("torso_roll_grid")
        self.joint_pos = tensor("joint_pos")  # (nh, npitch, nyaw, nroll, 10)
        self.joint_names = [str(n) for n in data["joint_names"]]
        self.base_height = float(data["base_height"])

    def sample(self, torso_cmd: torch.Tensor) -> torch.Tensor:
        """按躯干命令 (N,4)=(h_torso, pitch, yaw, roll) 四线性插值返回腿关节角 (N,10)。"""
        ih0, ih1, wh = ReferenceGait._grid_coords(torso_cmd[:, 0], self.h_grid)
        ip0, ip1, wp = ReferenceGait._grid_coords(torso_cmd[:, 1], self.pitch_grid)
        iy0, iy1, wy = ReferenceGait._grid_coords(torso_cmd[:, 2], self.yaw_grid)
        ir0, ir1, wr = ReferenceGait._grid_coords(torso_cmd[:, 3], self.roll_grid)
        out = None
        for ch, fh in ((ih0, 1.0 - wh), (ih1, wh)):
            for cp, fp in ((ip0, 1.0 - wp), (ip1, wp)):
                for cy, fy in ((iy0, 1.0 - wy), (iy1, wy)):
                    for cr, fr in ((ir0, 1.0 - wr), (ir1, wr)):
                        weight = (fh * fp * fy * fr).unsqueeze(-1)
                        corner = self.joint_pos[ch, cp, cy, cr]  # (N, 10)
                        out = corner * weight if out is None else out + corner * weight
        return out
