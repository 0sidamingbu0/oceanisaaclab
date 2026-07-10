# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""参考步态库离线生成器（路线 B：BDX 式参考轨迹模仿，placo 版）。

对照迪士尼 BDX 复刻（Open Duck / AWD go_bdx）的 Placo 参考步态方案：离线生成按
(dx, dy, dtheta) 参数化的周期步态库，训练时按命令插值取参考帧，奖励以关节角匹配
为主导。本脚本用 placo（Rhoban 的 pinocchio 运动学求解器，Open Duck 同款）实现：

1. placo.RobotWrapper 加载 ocean.urdf（mesh 路径自动重写到 assets/meshes）；
2. 在 (vx, vy, wz) 命令网格上，按「支撑脚世界系静止 + 摆动脚余弦摆动 + 抛物线抬脚 +
   躯干前倾 + 侧向重心 sway」的运动学步态公式生成脚底轨迹（头部前向约定与训练环境
   forward_vx_sign=-1 完全一致：URDF base_link +x 指尾部）；
3. placo KinematicsSolver（双脚 FrameTask：位置 1.0 / 姿态 0.05 权重）解出每帧
   10 个腿关节角，关节软限位（0.9 因子）由求解器强制满足；
4. 输出 npz 步态库：joint_pos / joint_vel / feet_contact / base 参考量，训练环境按
   命令三线性插值 + 相位环形插值采样（格式与旧版完全一致）。

基座高度约定（重要）：URDF 全零关节姿态即 BDX 标准站立姿态（屈膝微蹲，腿部有伸直
角度预留），默认基座高度直接取全零姿态 FK 的站立高度（≈0.385 m），**不再额外压低**。

可调步态参数全部集中在 GaitParams（步幅由 vx × 支撑时长推出，抬腿高度 / 周期 /
占空比 / 侧向 sway / 前倾角等均可命令行覆盖），见 --help。

依赖：pip install placo meshcat

运行（生成 + 每个命令回放一遍 meshcat 可视化确认）：

    python scripts/gen_reference_gait.py --viz

只看某个命令（如满速前进）、循环 5 遍：

    python scripts/gen_reference_gait.py --viz --viz-cmd 0.25,0,0 --viz-loops 5

调参示例（抬腿 5cm、周期 0.7s）：

    python scripts/gen_reference_gait.py --foot-clearance 0.05 --gait-period 0.7

注意：--gait-period 必须与 env cfg.gait_cycle_period 一致，否则训练环境启动时报错。
生成的库可用 scripts/play_reference_gait.py 播放确认。
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import placo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "source/oceanisaaclab/assets/urdf/ocean.urdf"
DEFAULT_MESH_DIR = REPO_ROOT / "source/oceanisaaclab/assets/meshes"
DEFAULT_OUT = REPO_ROOT / "source/oceanisaaclab/assets/gaits/reference_gait.npz"

# 头部前向 = forward_vx_sign * base_x（URDF base_link +x 指尾部），与训练环境一致
FORWARD_VX_SIGN = -1.0

LEG_JOINT_NAMES = [f"leg_{side}{i}_joint" for side in "rl" for i in range(1, 6)]
NECK_JOINT_NAMES = [f"neck_n{i}_joint" for i in range(1, 5)]


# ---------------------------------------------------------------------------
# 步态参数：想调什么改这里（或用同名命令行参数覆盖）
# ---------------------------------------------------------------------------
@dataclass
class GaitParams:
    """参考步态的全部可调参数。

    步幅（step length）不是独立参数，由命令速度推出：
        单步步幅 = vx × gait_period × gait_duty
    满速 vx=0.25、周期 0.6s、占空比 0.5 时步幅 = 7.5 cm。要更大步幅就调大
    vx_max（需同步扩大训练命令范围）或加长 gait_period。
    """

    gait_period: float = 0.6
    """[s] 完整步态周期（左右脚各迈一步）。必须与 env cfg.gait_cycle_period 一致。"""

    gait_period_fast: float = 0.48
    """[s] 满速命令下的步态周期（BDX 论文：相位速率 φ̇ 随命令变化，走得快步频高）。
    <=0 表示与 gait_period 相同（恒定步频）。每个命令的周期按速度占比在
    gait_period（零速）与 gait_period_fast（满速）之间线性插值，库里存 phase_rate
    (=1/周期) 表，训练环境按命令插值积分相位。"""

    gait_duty: float = 0.6
    """单脚接触占空比。0.6 在左右反相步态中产生每周期 20% 的双支撑窗口。"""

    num_phase_samples: int = 48
    """每周期相位采样帧数（48 帧 @0.6s ≈ 80Hz）。"""

    base_height: float = -1.0
    """[m] 参考步态基座高度。<=0 表示自动取 URDF 全零姿态（BDX 标准屈膝站立）的
    FK 站立高度（≈0.385 m），不额外下蹲——腿部伸直余量留给摆动/迈步。"""

    foot_clearance: float = 0.035
    """[m] 摆动脚抬腿高度（抛物线峰值，脚底离地间隙）。"""

    lateral_sway: float = 0.015
    """[m] 基座向支撑脚侧的侧向重心转移幅度（ZMP 重心 sway 的简化）。"""

    walk_lean_angle: float = 0.087
    """[rad] ≈5°。满速前进时躯干前倾角，随 vx 比例缩放（后退对称后倾）。
    与 cfg.walk_lean_angle 一致。"""

    foot_origin_offset: float = 0.067
    """[m] leg_[lr]5_link 原点到脚底的高度（与 cfg.foot_origin_offset 一致）。"""

    soft_limit_factor: float = 0.9
    """关节软限位系数（与 cfg.soft_joint_pos_limit_factor 一致），求解器内强制。"""

    vx_max: float = 0.25
    """[m/s] 命令网格 vx 上限（5 点对称网格；与训练命令范围一致）。"""

    vy_max: float = 0.15
    """[m/s] 命令网格 vy 上限（3 点对称网格）。"""

    wz_max: float = 0.8
    """[rad/s] 命令网格 wz 上限（5 点对称网格）。"""

    foot_rot_weight: float = 0.0025
    """脚底姿态任务权重（位置权重恒 1.0）。本腿无踝滚转，侧向 sway/转向时脚底无法
    严格保持水平，姿态权重过高会牺牲位置精度（0.05 时 ~9mm）；0.0025 时位置 <1mm
    且脚底仍近似水平（z 轴对齐 >0.997）。"""

    ik_iterations: int = 40
    """每帧 placo 求解迭代次数（warm-start 下 40 次足以收敛到亚毫米）。"""

    def vx_grid(self) -> np.ndarray:
        return np.linspace(-self.vx_max, self.vx_max, 5)

    def vy_grid(self) -> np.ndarray:
        return np.linspace(-self.vy_max, self.vy_max, 3)

    def wz_grid(self) -> np.ndarray:
        return np.linspace(-self.wz_max, self.wz_max, 5)

    def describe(self) -> str:
        lines = ["gait parameters:"]
        for f in fields(self):
            lines.append(f"  {f.name:20s} = {getattr(self, f.name)}")
        step = self.vx_max * self.gait_period * self.gait_duty
        lines.append(f"  {'step_length_max':20s} = {step:.4f}  [m] (= vx_max * period * duty)")
        return "\n".join(lines)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# placo 机器人加载 + IK 求解器
# ---------------------------------------------------------------------------
def load_placo_robot(urdf_path: Path, mesh_dir: Path) -> placo.RobotWrapper:
    """加载 ocean.urdf：把 package://ocean_description/meshes 重写到本仓库 meshes 目录。"""
    text = urdf_path.read_text()
    text = text.replace("package://ocean_description/meshes", str(mesh_dir.resolve()))
    tmp = Path(tempfile.mkdtemp(prefix="ocean_placo_")) / urdf_path.name
    tmp.write_text(text)
    robot = placo.RobotWrapper(str(tmp), placo.Flags.ignore_collisions)
    robot.update_kinematics()
    return robot


class PlacoGaitIK:
    """双脚 FrameTask 的 placo 运动学求解器。

    基座固定在世界原点（mask_fbase），脚目标在 base 系表达——与训练库的约定一致：
    p_base = Ry(lean) @ (p_world - [0, 0, base_height])。
    """

    FOOT_FRAMES = {"r": "leg_r5_link", "l": "leg_l5_link"}

    def __init__(self, robot: placo.RobotWrapper, params: GaitParams):
        self.robot = robot
        self.params = params
        # 关节软限位：围绕中点收缩到 soft_limit_factor 倍半行程，由求解器强制
        for name in LEG_JOINT_NAMES:
            lower, upper = robot.get_joint_limits(name)
            mid = 0.5 * (lower + upper)
            half = 0.5 * (upper - lower) * params.soft_limit_factor
            robot.set_joint_limits(name, mid - half, mid + half)
        robot.set_T_world_frame("base_link", np.eye(4))
        robot.update_kinematics()

        self.solver = placo.KinematicsSolver(robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        # 脖子保持 0（低权重正则，避免无任务关节漂移）
        neck_task = self.solver.add_joints_task()
        neck_task.set_joints({name: 0.0 for name in NECK_JOINT_NAMES})
        neck_task.configure("neck", "soft", 1e-2)
        # 双脚 6D 任务。姿态权重取低值（见 GaitParams.foot_rot_weight 说明）。
        self.foot_tasks = {}
        for side, frame in self.FOOT_FRAMES.items():
            task = self.solver.add_frame_task(frame, robot.get_T_world_frame(frame))
            task.configure(f"foot_{side}", "soft", 1.0, params.foot_rot_weight)
            self.foot_tasks[side] = task

    def reset_zero(self) -> None:
        """腿关节回 0、基座回世界原点（每个命令求解前的 warm-start 起点；
        可视化回放会移动基座，这里必须复位）。"""
        for name in LEG_JOINT_NAMES:
            self.robot.set_joint(name, 0.0)
        self.robot.set_T_world_frame("base_link", np.eye(4))
        self.robot.update_kinematics()

    def anchors(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """q=0（BDX 标准站姿）时双脚原点在 base 系的位置与姿态。"""
        self.reset_zero()
        pos, rot = {}, {}
        for side, frame in self.FOOT_FRAMES.items():
            pose = self.robot.get_T_world_frame(frame)
            pos[side] = pose[:3, 3].copy()
            rot[side] = pose[:3, :3].copy()
        return pos, rot

    def solve(self, targets: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """求解双脚 4x4 目标位姿，返回（10 关节角 [r1..r5, l1..l5]，最大位置残差）。"""
        for side, task in self.foot_tasks.items():
            task.T_world_frame = targets[side]
        for _ in range(self.params.ik_iterations):
            self.solver.solve(True)
            self.robot.update_kinematics()
        q = np.array([self.robot.get_joint(name) for name in LEG_JOINT_NAMES])
        err = max(
            float(np.linalg.norm(self.robot.get_T_world_frame(frame)[:3, 3] - targets[side][:3, 3]))
            for side, frame in self.FOOT_FRAMES.items()
        )
        return q, err


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
    ik: PlacoGaitIK,
    params: GaitParams,
    base_height: float,
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
    # 命令相关步态周期（BDX 论文：φ̇ 随命令变化）。速度占比取三轴归一化最大值，
    # 周期在 gait_period（零速）与 gait_period_fast（满速）间线性插值。
    period_fast = params.gait_period_fast if params.gait_period_fast > 0.0 else params.gait_period
    speed_frac = min(
        1.0, max(abs(vx) / params.vx_max, abs(vy) / params.vy_max, abs(wz) / params.wz_max)
    )
    cmd_period = params.gait_period + (period_fast - params.gait_period) * speed_frac
    stance_time = cmd_period * params.gait_duty
    step_disp = v_base * stance_time  # 支撑相内基座平移 → 脚相对基座反向平移
    step_yaw = wz * stance_time  # 支撑相内基座偏航 → 脚相对基座反向旋转
    # 满前速时前倾 walk_lean_angle，随 vx 成比例（后退命令对称后倾）
    lean = 0.0 if standing else params.walk_lean_angle * (vx / params.vx_max)
    rot_lean = _rot_y(lean)  # p_base = rot_lean @ (p_world - base_pos)（R_wb = Ry(-lean)）

    n_phase = params.num_phase_samples
    num = 1 if standing else n_phase
    joint_pos = np.zeros((n_phase, 10))
    feet_contact = np.zeros((n_phase, 2))
    max_pos_err = 0.0
    ik.reset_zero()
    for k in range(num):
        phase = k / n_phase
        targets = {}
        frame_contact = np.zeros(2)
        for col, side in enumerate("rl"):
            # 环境约定：右脚相位 = φ，左脚相位 = φ+0.5；右脚 φ∈[0,duty) 支撑
            foot_phase = phase if side == "r" else (phase + 0.5) % 1.0
            if standing:
                xi, lift, contact = 0.0, 0.0, True
            else:
                xi, lift, contact = _foot_cycle(foot_phase, params.gait_duty)
            anchor = anchors[side]
            # 世界（航向）系脚原点目标：绕基座旋转 + 平移 + 抬脚
            planar = _rot_z(step_yaw * xi)[:2, :2] @ anchor[:2] + step_disp * xi
            # 侧向 sway：右脚支撑(φ∈[0,0.5))时基座移向 +y（右侧）→ 脚目标相对基座 -y
            sway = 0.0 if standing else -params.lateral_sway * np.sin(2.0 * np.pi * phase)
            p_world = np.array(
                [planar[0], planar[1] + sway, params.foot_origin_offset + params.foot_clearance * lift]
            )
            target = np.eye(4)
            target[:3, 3] = rot_lean @ (p_world - np.array([0.0, 0.0, base_height]))
            target[:3, :3] = rot_lean @ _rot_z(step_yaw * xi) @ rot0[side]
            targets[side] = target
            frame_contact[col] = float(contact)
        q, pos_err = ik.solve(targets)
        joint_pos[k] = q
        feet_contact[k] = frame_contact
        max_pos_err = max(max_pos_err, pos_err)
    if standing:  # 站立帧对所有相位常量复制
        joint_pos[:] = joint_pos[0]
        feet_contact[:] = feet_contact[0]

    dt = cmd_period / n_phase
    joint_vel = (np.roll(joint_pos, -1, axis=0) - np.roll(joint_pos, 1, axis=0)) / (2.0 * dt)
    if standing:
        joint_vel[:] = 0.0
    # base 参考量（body 系）：v_body = Ry(lean) @ (v_base, 0)；proj_g = Ry(lean) @ (0,0,-1)
    phases = np.arange(n_phase) / n_phase
    sway_rate_pf = np.zeros(n_phase) if standing else (
        FORWARD_VX_SIGN * params.lateral_sway * 2.0 * np.pi / cmd_period
        * np.cos(2.0 * np.pi * phases)
    )
    lin_vel_b = np.zeros((n_phase, 3))
    for k in range(n_phase):
        # base_pos_pf is expressed in head/path coordinates; convert its derivative
        # to the URDF base frame before applying the constant torso lean.
        v_with_sway = v_base + np.array([0.0, FORWARD_VX_SIGN * sway_rate_pf[k]])
        lin_vel_b[k] = rot_lean @ np.array([v_with_sway[0], v_with_sway[1], 0.0])
    proj_g = rot_lean @ np.array([0.0, 0.0, -1.0])
    # body 系角速度：稳态转向时 ω_world=(0,0,wz)，ω_b = R_bw @ ω_w = rot_lean @ (0,0,wz)
    ang_vel_b = np.repeat(
        (rot_lean @ np.array([0.0, 0.0, wz]))[None, :], n_phase, axis=0
    )
    # path 系躯干 xy 轨迹（BDX 论文公式 (1) 的 p_t，path frame 坐标，+x=头前向）：
    # 生成器把脚目标在 base 系 y 方向平移 -sway(φ) ⇒ 躯干相对双脚中心（=path frame）
    # 在 base 系 +y 偏移 +sway(φ)。head-left 轴 = FORWARD_VX_SIGN * base_y = -base_y，
    # 故 path 系 y = -sway(φ) = +lateral_sway*sin(2πφ) 取负。x 方向本步态无前后振荡 → 0。
    base_pos_pf = np.zeros((n_phase, 2))
    if not standing:
        base_pos_pf[:, 1] = FORWARD_VX_SIGN * params.lateral_sway * np.sin(2.0 * np.pi * phases)
    # path 系躯干偏航（相对 path frame 朝向的 yaw 振荡）：本步态不建模 → 0
    base_yaw_pf = np.zeros(n_phase)
    # Nominal expressive head motion from the periodic reference. User head
    # commands are added as offsets at runtime, matching paper Eq. (6).
    neck_pos = np.zeros((n_phase, 4))
    if not standing:
        wave = np.sin(2.0 * np.pi * phases)
        wave_quadrature = np.cos(2.0 * np.pi * phases)
        neck_pos[:, 0] = 0.035 * wave_quadrature
        neck_pos[:, 1] = -0.035 * wave_quadrature
        neck_pos[:, 2] = 0.045 * wave
        neck_pos[:, 3] = 0.035 * wave
    neck_vel = (np.roll(neck_pos, -1, axis=0) - np.roll(neck_pos, 1, axis=0)) / (2.0 * dt)
    if standing:
        neck_vel[:] = 0.0
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "feet_contact": feet_contact,
        "lin_vel_b": lin_vel_b,
        "ang_vel_b": ang_vel_b,
        "proj_g": proj_g,
        "base_pos_pf": base_pos_pf,
        "base_yaw_pf": base_yaw_pf,
        "base_height": base_height,
        "base_pitch": -lean,  # R_wb = Ry(base_pitch)
        "phase_rate": 1.0 / cmd_period,
        "neck_pos": neck_pos,
        "neck_vel": neck_vel,
        "max_pos_err": max_pos_err,
    }


# ---------------------------------------------------------------------------
# meshcat 可视化回放
# ---------------------------------------------------------------------------
def replay_command(
    robot: placo.RobotWrapper,
    viz,
    frames: dict[str, np.ndarray],
    params: GaitParams,
    loops: int = 1,
) -> None:
    """在 meshcat 里按真实时间回放一个命令的整周期（基座抬到参考高度 + 前倾）。"""
    from placo_utils.visualization import robot_frame_viz

    dt = params.gait_period / params.num_phase_samples
    T_base = np.eye(4)
    T_base[:3, :3] = _rot_y(frames["base_pitch"])
    T_base[2, 3] = frames["base_height"]
    for _ in range(loops):
        for k in range(params.num_phase_samples):
            for j, name in enumerate(LEG_JOINT_NAMES):
                robot.set_joint(name, float(frames["joint_pos"][k, j]))
            robot.set_T_world_frame("base_link", T_base)
            robot.update_kinematics()
            viz.display(robot.state.q)
            robot_frame_viz(robot, "leg_r5_link")
            robot_frame_viz(robot, "leg_l5_link")
            time.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    defaults = GaitParams()
    parser.add_argument("--gait-period", type=float, default=defaults.gait_period, help="[s] 步态周期")
    parser.add_argument(
        "--gait-period-fast",
        type=float,
        default=defaults.gait_period_fast,
        help="[s] 满速命令步态周期（<=0 = 与 --gait-period 相同，恒定步频）",
    )
    parser.add_argument("--gait-duty", type=float, default=defaults.gait_duty, help="支撑占空比")
    parser.add_argument("--num-phase-samples", type=int, default=defaults.num_phase_samples)
    parser.add_argument(
        "--base-height",
        type=float,
        default=defaults.base_height,
        help="[m] 基座高度；<=0 = 自动取 URDF 全零姿态站立高度（默认，不下蹲）",
    )
    parser.add_argument("--foot-clearance", type=float, default=defaults.foot_clearance, help="[m] 抬腿高度")
    parser.add_argument("--lateral-sway", type=float, default=defaults.lateral_sway, help="[m] 侧向重心 sway")
    parser.add_argument("--walk-lean-angle", type=float, default=defaults.walk_lean_angle, help="[rad] 满速前倾角")
    parser.add_argument("--vx-max", type=float, default=defaults.vx_max, help="[m/s] vx 网格上限")
    parser.add_argument("--vy-max", type=float, default=defaults.vy_max, help="[m/s] vy 网格上限")
    parser.add_argument("--wz-max", type=float, default=defaults.wz_max, help="[rad/s] wz 网格上限")
    parser.add_argument("--viz", action="store_true", help="生成时用 meshcat 逐命令回放确认")
    parser.add_argument(
        "--viz-cmd",
        type=str,
        default=None,
        help='只回放最接近 "vx,vy,wz" 的网格命令（如 "0.25,0,0"），其余命令静默生成',
    )
    parser.add_argument("--viz-loops", type=int, default=1, help="每个命令回放遍数")
    args = parser.parse_args()

    params = GaitParams(
        gait_period=args.gait_period,
        gait_period_fast=args.gait_period_fast,
        gait_duty=args.gait_duty,
        num_phase_samples=args.num_phase_samples,
        base_height=args.base_height,
        foot_clearance=args.foot_clearance,
        lateral_sway=args.lateral_sway,
        walk_lean_angle=args.walk_lean_angle,
        vx_max=args.vx_max,
        vy_max=args.vy_max,
        wz_max=args.wz_max,
    )

    robot = load_placo_robot(args.urdf, args.mesh_dir)
    ik = PlacoGaitIK(robot, params)
    anchors, rot0 = ik.anchors()
    # 基座高度：默认 = 全零姿态（BDX 标准屈膝站立）时脚底落地的站立高度
    standing_height = float(np.mean([params.foot_origin_offset - anchors[s][2] for s in "rl"]))
    base_height = params.base_height if params.base_height > 0.0 else standing_height
    params.base_height = base_height
    print(params.describe())
    print(f"standing height at q=0 (BDX stance): {standing_height:.4f} m -> base_height = {base_height:.4f} m")
    for side in "rl":
        print(f"[{side}] stance anchor (base frame, q=0): {np.round(anchors[side], 4)}")

    viz = None
    viz_cmd = None
    if args.viz:
        from placo_utils.visualization import robot_viz

        viz = robot_viz(robot, "ocean")
        if args.viz_cmd is not None:
            viz_cmd = np.array([float(v) for v in args.viz_cmd.split(",")])
        print("meshcat 已启动（URL 见上方日志），逐命令回放中…")

    vx_grid, vy_grid, wz_grid = params.vx_grid(), params.vy_grid(), params.wz_grid()
    nx, ny, nz = len(vx_grid), len(vy_grid), len(wz_grid)
    n_phase = params.num_phase_samples
    joint_pos = np.zeros((nx, ny, nz, n_phase, 10), dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    feet_contact = np.zeros((nx, ny, nz, n_phase, 2), dtype=np.float32)
    base_pos_pf = np.zeros((nx, ny, nz, n_phase, 2), dtype=np.float32)
    base_yaw_pf = np.zeros((nx, ny, nz, n_phase), dtype=np.float32)
    lin_vel_b = np.zeros((nx, ny, nz, n_phase, 3), dtype=np.float32)
    ang_vel_b = np.zeros_like(lin_vel_b)
    neck_pos = np.zeros((nx, ny, nz, n_phase, 4), dtype=np.float32)
    neck_vel = np.zeros_like(neck_pos)
    proj_g = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    base_height_tab = np.zeros((nx, ny, nz), dtype=np.float32)
    base_pitch = np.zeros((nx, ny, nz), dtype=np.float32)
    phase_rate = np.zeros((nx, ny, nz), dtype=np.float32)

    worst_err = 0.0
    for ix, vx in enumerate(vx_grid):
        for iy, vy in enumerate(vy_grid):
            for iz, wz in enumerate(wz_grid):
                frames = generate_command_gait(
                    ik, params, base_height, anchors, rot0, float(vx), float(vy), float(wz)
                )
                joint_pos[ix, iy, iz] = frames["joint_pos"]
                joint_vel[ix, iy, iz] = frames["joint_vel"]
                feet_contact[ix, iy, iz] = frames["feet_contact"]
                base_pos_pf[ix, iy, iz] = frames["base_pos_pf"]
                base_yaw_pf[ix, iy, iz] = frames["base_yaw_pf"]
                lin_vel_b[ix, iy, iz] = frames["lin_vel_b"]
                ang_vel_b[ix, iy, iz] = frames["ang_vel_b"]
                proj_g[ix, iy, iz] = frames["proj_g"]
                base_height_tab[ix, iy, iz] = frames["base_height"]
                base_pitch[ix, iy, iz] = frames["base_pitch"]
                phase_rate[ix, iy, iz] = frames["phase_rate"]
                neck_pos[ix, iy, iz] = frames["neck_pos"]
                neck_vel[ix, iy, iz] = frames["neck_vel"]
                worst_err = max(worst_err, frames["max_pos_err"])
                step_len = abs(vx) * params.gait_period * params.gait_duty
                print(
                    f"cmd(vx={vx:+.3f}, vy={vy:+.2f}, wz={wz:+.1f})  "
                    f"step={step_len * 100:.1f}cm  max IK pos err = {frames['max_pos_err'] * 1000:.2f} mm"
                )
                if viz is not None and (
                    viz_cmd is None or np.allclose(viz_cmd, [vx, vy, wz], atol=1e-6)
                ):
                    replay_command(robot, viz, frames, params, loops=args.viz_loops)
    print(f"\nworst IK position error over library: {worst_err * 1000:.2f} mm")
    if worst_err > 0.005:
        print("WARNING: IK residual above 5 mm — check workspace / target heights.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        feet_contact=feet_contact,
        base_pos_pf=base_pos_pf,
        base_yaw_pf=base_yaw_pf,
        lin_vel_b=lin_vel_b,
        ang_vel_b=ang_vel_b,
        proj_g=proj_g,
        base_height=base_height_tab,
        base_pitch=base_pitch,
        phase_rate=phase_rate,
        neck_pos=neck_pos,
        neck_vel=neck_vel,
        vx_grid=vx_grid.astype(np.float32),
        vy_grid=vy_grid.astype(np.float32),
        wz_grid=wz_grid.astype(np.float32),
        gait_period=np.float32(params.gait_period),
        gait_duty=np.float32(params.gait_duty),
        forward_vx_sign=np.float32(FORWARD_VX_SIGN),
        foot_clearance=np.float32(params.foot_clearance),
        joint_names=np.array(LEG_JOINT_NAMES),
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"saved gait library: {args.out} ({size_kb:.0f} KB)")
    print("确认轨迹：python scripts/play_reference_gait.py --vx 0.25")


if __name__ == "__main__":
    main()
