# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""脖子/头部参考映射离线生成器（路线 B 论文复刻：加头部命令）。

论文头部命令 g 里的 (Δh_head, Δθ_head) 是相对标称头姿的偏移。本机脖子 4 关节
（neck_n1..n4）可实现 4-DOF 头姿：头高 Δh（矢状面 n1/n2 对）+ 点头 pitch + 摇头 yaw
+ 歪头 roll。因为 base 固定时脖子在 base 之上、与脚 IK 完全解耦，头部命令**不做成
步态库的额外网格维度**（会维度爆炸），而是单独在 4-DOF 头命令网格上用 placo 解出
对应的 4 个脖子关节角，存成一张小映射表。训练/部署运行时按头命令四线性插值取脖子
参考角（reference_gait.py 的 NeckHeadMap），作为脖子关节模仿奖励的目标。

约定（与 gen_reference_gait.py / 训练环境一致）：
- 复用 gen_reference_gait.load_placo_robot（URDF mesh 路径重写）与 NECK_JOINT_NAMES；
- base_link 固定在世界原点（mask_fbase）；头姿目标在 base 系表达；
- 头前向 = base -x（URDF base_link +x 指尾部，forward_vx_sign=-1）；
- 头部命令 4 轴：
    Δh    [m]   头部 link（neck_n4_link）原点相对标称站姿的竖直偏移（+ 抬高）；
    pitch [rad] 头绕自身 y 轴点头（+ 低头/抬头，符号见下）；
    yaw   [rad] 头绕自身 z 轴摇头（+ 左转，与 wz 左正同向）；
    roll  [rad] 头绕自身 x 轴歪头。
  姿态偏移在头局部系右乘：R_target = R0 @ Rz(yaw) @ Ry(pitch) @ Rx(roll)。

依赖：pip install placo（与 gen_reference_gait.py 相同环境）。

运行：
    ./_isaaclab/isaaclab.sh -p scripts/gen_neck_head_map.py
    ./_isaaclab/isaaclab.sh -p scripts/gen_neck_head_map.py --dh-max 0.05 --pitch-max 0.5 \
        --yaw-max 1.0 --roll-max 0.6 --grid 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import placo

# 复用 gen_reference_gait 的加载器与约定（同目录脚本）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_reference_gait import (  # noqa: E402
    DEFAULT_MESH_DIR,
    DEFAULT_URDF,
    LEG_JOINT_NAMES,
    NECK_JOINT_NAMES,
    REPO_ROOT,
    _rot_y,
    _rot_z,
    load_placo_robot,
)

DEFAULT_OUT = REPO_ROOT / "source/oceanisaaclab/assets/gaits/neck_head_map.npz"

HEAD_FRAME = "neck_n4_link"  # 脖子链顶端 = 头（URDF 中 neck_n4_link 质量 ~1.05kg）


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@dataclass
class HeadParams:
    """头部命令网格参数（4-DOF：Δh / pitch / yaw / roll）。范围取脖子关节可达内的保守值。"""

    dh_max: float = 0.02
    """[m] 头高偏移网格上限（±对称，5 点）。本机脖子头高权限弱（与 pitch 共用 n1/n2），
    压到 ±2cm 保证可达、配合连续 warm-start 保持映射单调。"""

    pitch_max: float = 0.5
    """[rad] 点头网格上限（±对称）。n1/n2 矢状面对，与头高耦合，由 IK 最小二乘协调。"""

    yaw_max: float = 1.0
    """[rad] 摇头网格上限（±对称）。n3 独立 yaw，URDF 限 ±1.571。"""

    roll_max: float = 0.6
    """[rad] 歪头网格上限（±对称）。n4 独立 roll，URDF 限 ±0.785。"""

    grid: int = 5
    """每轴网格点数（奇数含 0/标称姿态）。5⁴=625 个姿态，placo 解很便宜。"""

    soft_limit_factor: float = 0.9
    """脖子关节软限位系数（与腿一致，求解器内强制，避免映射跑到硬限位）。"""

    ik_iterations: int = 150
    """每个头姿 placo 迭代次数（warm-start 下足够收敛；dh 与 pitch 共用 n1/n2 对，
    多迭代帮助收敛到抬头/craning 解而非局部对称解）。"""

    joint_reg_weight: float = 1e-3
    """脖子关节零位正则权重（解 4 关节 vs 6D 头姿是超定最小二乘，微正则稳定冗余）。"""

    def dh_grid(self) -> np.ndarray:
        return np.linspace(-self.dh_max, self.dh_max, self.grid)

    def pitch_grid(self) -> np.ndarray:
        return np.linspace(-self.pitch_max, self.pitch_max, self.grid)

    def yaw_grid(self) -> np.ndarray:
        return np.linspace(-self.yaw_max, self.yaw_max, self.grid)

    def roll_grid(self) -> np.ndarray:
        return np.linspace(-self.roll_max, self.roll_max, self.grid)

    def describe(self) -> str:
        lines = ["head map parameters:"]
        for f in fields(self):
            lines.append(f"  {f.name:20s} = {getattr(self, f.name)}")
        return "\n".join(lines)


class PlacoNeckIK:
    """base 固定 + 头部 6D frame task 的 placo 求解器，解 4 个脖子关节角。

    与 PlacoGaitIK 同约定：base_link 固定在世界原点。脚不加任务（脖子解与脚解耦，
    脚由 gen_reference_gait 单独生成）。头姿目标在 base 系表达。
    """

    def __init__(self, robot: placo.RobotWrapper, params: HeadParams):
        self.robot = robot
        self.params = params
        # 脖子软限位（围绕中点收缩到 soft_limit_factor 倍半行程）
        for name in NECK_JOINT_NAMES:
            lower, upper = robot.get_joint_limits(name)
            mid = 0.5 * (lower + upper)
            half = 0.5 * (upper - lower) * params.soft_limit_factor
            robot.set_joint_limits(name, mid - half, mid + half)
        for name in NECK_JOINT_NAMES:
            robot.set_joint(name, 0.0)
        robot.set_T_world_frame("base_link", np.eye(4))
        robot.update_kinematics()

        self.solver = placo.KinematicsSolver(robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        # 头姿目标：姿态(3D) + 高度(仅 z)= 4 约束，恰好匹配 4 个脖子关节。
        # 不约束头 x,y 位置——脖子是 4 连杆链，转头/点头时头 link 原点必然平移，
        # 若把 x,y 也钉死会与姿态矛盾（实测残差 >140mm）。只钉 z（高度）+ 姿态。
        self.T0 = robot.get_T_world_frame(HEAD_FRAME).copy()
        self.pos_task = self.solver.add_position_task(HEAD_FRAME, self.T0[:3, 3].copy())
        self.pos_task.mask.set_axises("z")  # 仅约束竖直高度
        self.pos_task.configure("head_h", "soft", 1.0)
        self.orient_task = self.solver.add_orientation_task(HEAD_FRAME, self.T0[:3, :3].copy())
        self.orient_task.configure("head_rot", "soft", 1.0)
        # 零位正则：稳定冗余、让 0 命令收敛到标称站姿
        reg = self.solver.add_joints_task()
        reg.set_joints({name: 0.0 for name in NECK_JOINT_NAMES})
        reg.configure("neck_reg", "soft", params.joint_reg_weight)

    def nominal_pose(self) -> np.ndarray:
        return self.T0

    def solve(self, dh: float, pitch: float, yaw: float, roll: float) -> tuple[np.ndarray, float]:
        """解一个头部命令 → 返回（4 个脖子关节角 [n1..n4]，头高 z 残差 [m]）。

        姿态偏移在 **base 系左乘**（不是头局部系）：URDF 脖子有效轴在 base 系为
        n3≈+z(yaw)、n1/n2≈y(pitch)、n4≈x(roll)，头 link 局部系相对 base 转了 ~90°，
        若在头局部系加偏移会把 yaw/pitch/roll 映射到错误关节。base 系左乘后一一对应：
        yaw 绕 base z(转头)、pitch 绕 base y(点头)、roll 绕 base x(歪头)。
        """
        target_pos = self.T0[:3, 3] + np.array([0.0, 0.0, dh])
        target_R = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll) @ self.T0[:3, :3]
        self.pos_task.target_world = target_pos
        self.orient_task.R_world_frame = target_R
        for _ in range(self.params.ik_iterations):
            self.solver.solve(True)
            self.robot.update_kinematics()
        q = np.array([self.robot.get_joint(name) for name in NECK_JOINT_NAMES])
        z_now = float(self.robot.get_T_world_frame(HEAD_FRAME)[2, 3])
        err = abs(z_now - target_pos[2])
        return q, err


def _viz_sweep(ik: "PlacoNeckIK", params: HeadParams, loops: int, hold: float) -> None:
    """meshcat 里逐轴扫掠头部命令(其余轴归零),实时显示脖子解出的姿态。

    脚保持全零默认站姿,只动脖子 4 关节;每个头命令用 placo 现场求解(与库同一 IK)。
    依赖 meshcat（pip install meshcat）。打开日志里打印的 URL 观看。
    """
    import time

    from placo_utils.visualization import robot_frame_viz, robot_viz

    robot = ik.robot
    for name in LEG_JOINT_NAMES:
        robot.set_joint(name, 0.0)
    viz = robot_viz(robot)

    def show(dh: float, pitch: float, yaw: float, roll: float, label: str) -> None:
        q, _ = ik.solve(dh, pitch, yaw, roll)
        for name in LEG_JOINT_NAMES:
            robot.set_joint(name, 0.0)
        robot.update_kinematics()
        viz.display(robot.state.q)
        robot_frame_viz(robot, HEAD_FRAME)
        print(f"  [viz] {label:20s} neck[n1..n4]={np.round(q, 3).tolist()}")
        time.sleep(hold)

    # 每轴：0 → +max → −max → 0，来回扫；其余轴归零
    axes = [
        ("dh", params.dh_grid(), 0),
        ("pitch", params.pitch_grid(), 1),
        ("yaw", params.yaw_grid(), 2),
        ("roll", params.roll_grid(), 3),
    ]
    print(f"meshcat viz: 逐轴扫掠 {loops} 遍（脚固定默认站姿，只动脖子）")
    for _ in range(loops):
        for name, grid, idx in axes:
            fine = np.concatenate([
                np.linspace(0.0, grid[-1], 12),
                np.linspace(grid[-1], grid[0], 24),
                np.linspace(grid[0], 0.0, 12),
            ])
            for v in fine:
                cmd = [0.0, 0.0, 0.0, 0.0]
                cmd[idx] = float(v)
                show(*cmd, f"{name}={v:+.3f}")


def generate(params: HeadParams, urdf: Path, mesh_dir: Path) -> dict:
    robot = load_placo_robot(urdf, mesh_dir)
    ik = PlacoNeckIK(robot, params)

    dh_grid = params.dh_grid()
    pitch_grid = params.pitch_grid()
    yaw_grid = params.yaw_grid()
    roll_grid = params.roll_grid()
    nh, npi, ny, nr = len(dh_grid), len(pitch_grid), len(yaw_grid), len(roll_grid)
    neck_pos = np.zeros((nh, npi, ny, nr, 4), dtype=np.float32)

    worst = 0.0
    total = nh * npi * ny * nr
    done = 0
    # 4 维蛇行(boustrophedon)遍历 + 连续 warm-start(不复位到零):相邻求解点只差
    # 一个网格步,placo 从上一解 warm-start → 始终停在同一 IK 分枝上,消除 dh/pitch
    # 共用 n1/n2 造成的分枝跳变与非单调。零命令的精确解仍是零角度,不受 warm-start 影响。
    ik.robot.update_kinematics()
    for ih in range(nh):
        ip_range = range(npi) if ih % 2 == 0 else range(npi - 1, -1, -1)
        for ip in ip_range:
            iy_range = range(ny) if (ih + ip) % 2 == 0 else range(ny - 1, -1, -1)
            for iy in iy_range:
                ir_range = range(nr) if (ih + ip + iy) % 2 == 0 else range(nr - 1, -1, -1)
                for ir in ir_range:
                    q, err = ik.solve(
                        float(dh_grid[ih]),
                        float(pitch_grid[ip]),
                        float(yaw_grid[iy]),
                        float(roll_grid[ir]),
                    )
                    neck_pos[ih, ip, iy, ir] = q
                    worst = max(worst, err)
                    done += 1
        print(f"  [{done}/{total}] dh={dh_grid[ih]:+.3f} done, worst z err so far {worst*1000:.2f} mm")

    print(f"worst head-height residual: {worst*1000:.3f} mm")
    return dict(
        dh_grid=dh_grid.astype(np.float32),
        pitch_grid=pitch_grid.astype(np.float32),
        yaw_grid=yaw_grid.astype(np.float32),
        roll_grid=roll_grid.astype(np.float32),
        neck_pos=neck_pos,
        neck_joint_names=np.array(NECK_JOINT_NAMES),
        nominal_head_pose=ik.nominal_pose().astype(np.float32),
    )


def main() -> None:
    defaults = HeadParams()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dh-max", type=float, default=defaults.dh_max)
    parser.add_argument("--pitch-max", type=float, default=defaults.pitch_max)
    parser.add_argument("--yaw-max", type=float, default=defaults.yaw_max)
    parser.add_argument("--roll-max", type=float, default=defaults.roll_max)
    parser.add_argument("--grid", type=int, default=defaults.grid)
    parser.add_argument("--viz", action="store_true", help="生成后用 meshcat 逐轴扫掠回放确认头姿")
    parser.add_argument("--viz-only", action="store_true", help="只可视化、不重新生成/保存 npz")
    parser.add_argument("--viz-loops", type=int, default=2, help="扫掠遍数")
    parser.add_argument("--viz-hold", type=float, default=0.03, help="[s] 每帧停留时间")
    args = parser.parse_args()

    params = HeadParams(
        dh_max=args.dh_max,
        pitch_max=args.pitch_max,
        yaw_max=args.yaw_max,
        roll_max=args.roll_max,
        grid=args.grid,
    )
    print(params.describe())
    if not args.viz_only:
        data = generate(params, args.urdf, args.mesh_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
        print(f"saved neck-head map → {args.out}  (grid {args.grid}^4 = {args.grid**4} poses)")
    if args.viz or args.viz_only:
        robot = load_placo_robot(args.urdf, args.mesh_dir)
        ik = PlacoNeckIK(robot, params)
        _viz_sweep(ik, params, loops=args.viz_loops, hold=args.viz_hold)


if __name__ == "__main__":
    main()
