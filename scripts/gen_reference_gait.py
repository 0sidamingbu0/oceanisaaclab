# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""参考步态库离线生成器（路线 B：BDX 式参考轨迹模仿）。

对照迪士尼 BDX 复刻（Open Duck / AWD go_bdx）的 Placo ZMP 参考步态方案：他们用 ZMP
walk-pattern generator 离线生成按 (dx, dy, dtheta) 参数化的周期步态库，训练时按命令
插值取参考帧，奖励以关节角匹配为主导。本脚本用纯 numpy 复刻同样的产物：

1. 从 ocean.urdf 解析双腿 5-DOF 运动链（r1 髋 yaw / r2 髋 roll / r3 髋 pitch /
   r4 膝 / r5 踝 pitch），建立数值 FK；
2. 在 (vx, vy, wz) 命令网格上，按「支撑脚世界系静止 + 摆动脚余弦摆动 + 抛物线抬脚 +
   躯干前倾 + 侧向重心 sway」的运动学步态公式生成脚底轨迹（头部前向约定与训练环境
   forward_vx_sign=-1 完全一致：URDF base_link +x 指尾部）；
3. 用阻尼最小二乘数值 IK（位置 + 姿态 6 维任务）解出每帧 10 个腿关节角，软限位内钳制；
4. 输出 npz 步态库：joint_pos / joint_vel / feet_contact / base 参考量，训练环境按
   命令三线性插值 + 相位环形插值采样。

运行（纯 numpy，系统 python 或 isaaclab python 均可）：

    python scripts/gen_reference_gait.py \
        --urdf source/oceanisaaclab/assets/urdf/ocean.urdf \
        --out source/oceanisaaclab/assets/gaits/reference_gait.npz
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 常量：与训练环境 cfg 保持一致（改这里必须同步改 env cfg）
# ---------------------------------------------------------------------------
FORWARD_VX_SIGN = -1.0  # 头部前向 = forward_vx_sign * base_x（URDF +x 指尾部）
GAIT_PERIOD = 0.6  # [s] 完整步态周期（与 cfg.gait_cycle_period 一致）
GAIT_DUTY = 0.5  # 单脚接触占空比（与 cfg.gait_duty_factor 一致）
NUM_PHASE_SAMPLES = 48  # 每周期相位采样帧数
BASE_HEIGHT_REF = 0.36  # [m] 参考步态基座高度（微蹲，留 IK 上下工作空间）
FOOT_ORIGIN_OFFSET = 0.067  # [m] leg_[lr]5_link 原点到脚底的高度（与 cfg 一致）
FOOT_CLEARANCE = 0.035  # [m] 摆动脚抬脚高度（changelog 建议 0.03~0.04）
WALK_LEAN_ANGLE = 0.087  # [rad] ≈5° 满速前进时躯干前倾角（与 cfg.walk_lean_angle 一致）
LATERAL_SWAY = 0.015  # [m] 基座向支撑脚侧的侧向 sway 幅度（ZMP 步态的简化重心转移）
SOFT_LIMIT_FACTOR = 0.9  # 关节软限位系数（与 cfg.soft_joint_pos_limit_factor 一致）

# 命令网格（覆盖训练命令范围 vx±0.25 / vy±0.15 / wz±0.8）
VX_GRID = np.array([-0.25, -0.125, 0.0, 0.125, 0.25])
VY_GRID = np.array([-0.15, 0.0, 0.15])
WZ_GRID = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

LEG_JOINT_NAMES = [f"leg_{side}{i}_joint" for side in "rl" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# URDF 运动链解析 + FK
# ---------------------------------------------------------------------------
def _rpy_to_rot(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rot_z @ rot_y @ rot_x


class LegChain:
    """单腿 5 关节串联链（base_link -> leg_[lr]5_link），数值 FK + DLS IK。"""

    def __init__(self, urdf_path: Path, side: str):
        tree = ET.parse(urdf_path)
        self.origins: list[np.ndarray] = []  # 4x4 固定变换
        self.axes: list[np.ndarray] = []
        self.lower = np.zeros(5)
        self.upper = np.zeros(5)
        for i in range(1, 6):
            joint = next(j for j in tree.getroot().findall("joint") if j.get("name") == f"leg_{side}{i}_joint")
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz").split()])
            rpy = [float(v) for v in origin.get("rpy").split()]
            transform = np.eye(4)
            transform[:3, :3] = _rpy_to_rot(rpy)
            transform[:3, 3] = xyz
            self.origins.append(transform)
            axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
            self.axes.append(axis / np.linalg.norm(axis))
            limit = joint.find("limit")
            self.lower[i - 1] = float(limit.get("lower"))
            self.upper[i - 1] = float(limit.get("upper"))
        # 软限位：围绕中点收缩到 0.9 倍半行程（与 Isaac Lab soft_joint_pos_limit_factor 一致）
        mid = 0.5 * (self.lower + self.upper)
        half = 0.5 * (self.upper - self.lower) * SOFT_LIMIT_FACTOR
        self.soft_lower = mid - half
        self.soft_upper = mid + half

    def fk(self, q: np.ndarray) -> np.ndarray:
        """base_link 系下 leg_[lr]5_link 的 4x4 位姿。"""
        transform = np.eye(4)
        for i in range(5):
            joint_rot = np.eye(4)
            axis = self.axes[i]
            c, s = np.cos(q[i]), np.sin(q[i])
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            joint_rot[:3, :3] = np.eye(3) + s * skew + (1 - c) * (skew @ skew)
            transform = transform @ self.origins[i] @ joint_rot
        return transform

    def ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        pos_weight: float = 1.0,
        # 姿态权重取低值：本腿无踝滚转，侧向 sway/转向时脚底无法严格保持水平，
        # 高权重会把位置精度牺牲到 ~15mm；0.05 时位置 ~2mm 且脚底仍近似水平（>0.99）。
        rot_weight: float = 0.05,
        iterations: int = 120,
        damping: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        """阻尼最小二乘 IK（6 维任务：位置 3 + 姿态 3），返回 (关节角, 残余位置误差)。"""
        q = q_init.copy()
        eps = 1e-5
        for _ in range(iterations):
            pose = self.fk(q)
            pos_err = target_pos - pose[:3, 3]
            rot_cur = pose[:3, :3]
            rot_err = 0.5 * sum(np.cross(rot_cur[:, k], target_rot[:, k]) for k in range(3))
            err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])
            if np.linalg.norm(pos_err) < 1e-5 and np.linalg.norm(rot_err) < 1e-4:
                break
            jac = np.zeros((6, 5))
            for j in range(5):
                dq = q.copy()
                dq[j] += eps
                pose_d = self.fk(dq)
                dpos = (pose_d[:3, 3] - pose[:3, 3]) / eps
                drot = 0.5 * sum(np.cross(rot_cur[:, k], (pose_d[:3, :3][:, k] - rot_cur[:, k]) / eps) for k in range(3))
                jac[:3, j] = pos_weight * dpos
                jac[3:, j] = rot_weight * drot
            jjt = jac @ jac.T + damping * np.eye(6)
            q = q + jac.T @ np.linalg.solve(jjt, err)
            q = np.clip(q, self.soft_lower, self.soft_upper)
        final = self.fk(q)
        return q, float(np.linalg.norm(target_pos - final[:3, 3]))


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# 步态公式：脚底轨迹（世界/航向系） → base 系目标
# ---------------------------------------------------------------------------
def _foot_cycle(phase: float, duty: float) -> tuple[float, float, bool]:
    """单脚步态周期参数。

    返回 (xi, lift, in_contact)：
      xi ∈ [-0.5, 0.5]  相对步长参数。支撑相从 +0.5 线性走到 -0.5（脚世界系静止、
        基座前移的运动学等效）；摆动相用余弦缓动从 -0.5 拉回 +0.5。
      lift  摆动相抬脚高度归一化（sin 抛物线，支撑相为 0）。
    """
    if phase < duty:  # stance
        s = phase / duty
        return 0.5 - s, 0.0, True
    s = (phase - duty) / (1.0 - duty)  # swing
    s_smooth = 0.5 * (1.0 - np.cos(np.pi * s))
    return -0.5 + s_smooth, float(np.sin(np.pi * s)), False


def generate_command_gait(
    chains: dict[str, LegChain],
    anchors: dict[str, np.ndarray],
    rot0: dict[str, np.ndarray],
    vx: float,
    vy: float,
    wz: float,
) -> dict[str, np.ndarray]:
    """生成单个 (vx, vy, wz) 命令的整周期参考帧。

    vx / vy / wz 为头部系命令（vx 头前向、vy 头左向、wz 逆时针）。转到 base 系：
    v_base = FORWARD_VX_SIGN * (vx, vy)。
    """
    standing = max(abs(vx), abs(vy)) < 1e-6 and abs(wz) < 1e-6
    v_base = FORWARD_VX_SIGN * np.array([vx, vy])
    stance_time = GAIT_PERIOD * GAIT_DUTY
    step_disp = v_base * stance_time  # 支撑相内基座平移 → 脚相对基座反向平移
    step_yaw = wz * stance_time  # 支撑相内基座偏航 → 脚相对基座反向旋转
    # 满前速时前倾 WALK_LEAN_ANGLE，随 vx 成比例（后退命令对称后倾）
    vx_max = float(np.max(np.abs(VX_GRID)))
    lean = 0.0 if standing else WALK_LEAN_ANGLE * (vx / vx_max)
    rot_lean = _rot_y(lean)  # p_base = rot_lean @ (p_world - base_pos)（R_wb = Ry(-lean)）

    num = 1 if standing else NUM_PHASE_SAMPLES
    joint_pos = np.zeros((NUM_PHASE_SAMPLES, 10))
    feet_contact = np.zeros((NUM_PHASE_SAMPLES, 2))
    max_pos_err = 0.0
    q_warm = {side: np.zeros(5) for side in "rl"}
    for k in range(num):
        phase = k / NUM_PHASE_SAMPLES
        frame_q = np.zeros(10)
        frame_contact = np.zeros(2)
        for col, side in enumerate("rl"):
            # 环境约定：右脚相位 = φ，左脚相位 = φ+0.5；右脚 φ∈[0,duty) 支撑
            foot_phase = phase if side == "r" else (phase + 0.5) % 1.0
            if standing:
                xi, lift, contact = 0.0, 0.0, True
            else:
                xi, lift, contact = _foot_cycle(foot_phase, GAIT_DUTY)
            anchor = anchors[side]
            # 世界（航向）系脚原点目标：绕基座旋转 + 平移 + 抬脚
            planar = _rot_z(step_yaw * xi)[:2, :2] @ anchor[:2] + step_disp * xi
            # 侧向 sway：右脚支撑(φ∈[0,0.5))时基座移向 +y（右侧）→ 脚目标相对基座 -y
            sway = 0.0 if standing else -LATERAL_SWAY * np.sin(2.0 * np.pi * phase)
            p_world = np.array([planar[0], planar[1] + sway, FOOT_ORIGIN_OFFSET + FOOT_CLEARANCE * lift])
            p_base = rot_lean @ (p_world - np.array([0.0, 0.0, BASE_HEIGHT_REF]))
            rot_target = rot_lean @ _rot_z(step_yaw * xi) @ rot0[side]
            q, pos_err = chains[side].ik(p_base, rot_target, q_warm[side])
            q_warm[side] = q
            frame_q[col * 5 : col * 5 + 5] = q
            frame_contact[col] = float(contact)
            max_pos_err = max(max_pos_err, pos_err)
        joint_pos[k] = frame_q
        feet_contact[k] = frame_contact
    if standing:  # 站立帧对所有相位常量复制
        joint_pos[:] = joint_pos[0]
        feet_contact[:] = feet_contact[0]

    dt = GAIT_PERIOD / NUM_PHASE_SAMPLES
    joint_vel = (np.roll(joint_pos, -1, axis=0) - np.roll(joint_pos, 1, axis=0)) / (2.0 * dt)
    if standing:
        joint_vel[:] = 0.0
    # base 参考量（body 系）：v_body = Ry(lean) @ (v_base, 0)；proj_g = Ry(lean) @ (0,0,-1)
    lin_vel_b = rot_lean @ np.array([v_base[0], v_base[1], 0.0])
    proj_g = rot_lean @ np.array([0.0, 0.0, -1.0])
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "feet_contact": feet_contact,
        "lin_vel_b": lin_vel_b,
        "proj_g": proj_g,
        "base_height": BASE_HEIGHT_REF,
        "base_pitch": -lean,  # R_wb = Ry(base_pitch)
        "max_pos_err": max_pos_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--urdf", type=Path, default=repo_root / "source/oceanisaaclab/assets/urdf/ocean.urdf")
    parser.add_argument("--out", type=Path, default=repo_root / "source/oceanisaaclab/assets/gaits/reference_gait.npz")
    args = parser.parse_args()

    chains = {side: LegChain(args.urdf, side) for side in "rl"}
    # 站立锚点：q=0 FK 的脚原点平面位置（自然站位），姿态取 q=0 的脚系姿态（脚底水平）
    anchors, rot0 = {}, {}
    for side in "rl":
        pose0 = chains[side].fk(np.zeros(5))
        anchors[side] = pose0[:3, 3].copy()
        rot0[side] = pose0[:3, :3].copy()
        print(f"[{side}] stance anchor (base frame, q=0): {np.round(pose0[:3, 3], 4)}")

    nx, ny, nz = len(VX_GRID), len(VY_GRID), len(WZ_GRID)
    joint_pos = np.zeros((nx, ny, nz, NUM_PHASE_SAMPLES, 10), dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    feet_contact = np.zeros((nx, ny, nz, NUM_PHASE_SAMPLES, 2), dtype=np.float32)
    lin_vel_b = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    proj_g = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    base_height = np.zeros((nx, ny, nz), dtype=np.float32)
    base_pitch = np.zeros((nx, ny, nz), dtype=np.float32)

    worst_err = 0.0
    for ix, vx in enumerate(VX_GRID):
        for iy, vy in enumerate(VY_GRID):
            for iz, wz in enumerate(WZ_GRID):
                frames = generate_command_gait(chains, anchors, rot0, float(vx), float(vy), float(wz))
                joint_pos[ix, iy, iz] = frames["joint_pos"]
                joint_vel[ix, iy, iz] = frames["joint_vel"]
                feet_contact[ix, iy, iz] = frames["feet_contact"]
                lin_vel_b[ix, iy, iz] = frames["lin_vel_b"]
                proj_g[ix, iy, iz] = frames["proj_g"]
                base_height[ix, iy, iz] = frames["base_height"]
                base_pitch[ix, iy, iz] = frames["base_pitch"]
                worst_err = max(worst_err, frames["max_pos_err"])
                print(
                    f"cmd(vx={vx:+.3f}, vy={vy:+.2f}, wz={wz:+.1f})  "
                    f"max IK pos err = {frames['max_pos_err'] * 1000:.2f} mm"
                )
    print(f"\nworst IK position error over library: {worst_err * 1000:.2f} mm")
    if worst_err > 0.005:
        print("WARNING: IK residual above 5 mm — check workspace / target heights.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        feet_contact=feet_contact,
        lin_vel_b=lin_vel_b,
        proj_g=proj_g,
        base_height=base_height,
        base_pitch=base_pitch,
        vx_grid=VX_GRID.astype(np.float32),
        vy_grid=VY_GRID.astype(np.float32),
        wz_grid=WZ_GRID.astype(np.float32),
        gait_period=np.float32(GAIT_PERIOD),
        gait_duty=np.float32(GAIT_DUTY),
        forward_vx_sign=np.float32(FORWARD_VX_SIGN),
        foot_clearance=np.float32(FOOT_CLEARANCE),
        joint_names=np.array(LEG_JOINT_NAMES),
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"saved gait library: {args.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
