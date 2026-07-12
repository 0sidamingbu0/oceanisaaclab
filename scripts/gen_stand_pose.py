# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""站立姿态参考库离线生成器（路线 B 论文复刻：perpetual 站立策略）。

论文把运动分成独立策略（divide-and-conquer）：periodic 行走策略（已实现）+ perpetual
站立策略。站立策略命令 g_perp = (Δh_head, Δθ_head, h_torso, θ_torso)（式 5）——躯干的
高度/朝向 + 头部的高度/朝向，**无相位**，脚不迈步。本脚本在躯干命令 4-DOF 网格
(h_torso, pitch, yaw, roll) 上用 placo 解静态站姿的 10 个腿关节角：双脚固定在地面站立
锚点、躯干处于命令的高度与朝向。头部命令与脚 IK 解耦（脖子在 base 之上），复用
gen_neck_head_map.py 的 neck_head_map.npz，不进本网格。

约定（与 gen_reference_gait.py / 训练环境一致）：
- 复用 gen_reference_gait 的 load_placo_robot / PlacoGaitIK / GaitParams / LEG_JOINT_NAMES；
- base_link 固定世界原点、脚目标在 base 系表达：p_base = R_bw @ (p_world − [0,0,h_torso])，
  其中 R_bw = R_torso^T（R_torso = Rz(yaw)Ry(pitch)Rx(roll) 为躯干世界朝向）；
- 脚世界锚点 = q=0 标称站立时脚原点的世界 xy + foot_origin_offset（脚底贴地）；
- 躯干高度 h_torso = 标称站高 base_height + Δh_torso 命令偏移。

依赖：pip install placo（与 gen_reference_gait.py 相同环境，用系统 python3 跑）。

运行：
    python3 scripts/gen_stand_pose.py
    python3 scripts/gen_stand_pose.py --viz-only     # meshcat 扫掠可视化
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_reference_gait import (  # noqa: E402
    DEFAULT_MESH_DIR,
    DEFAULT_URDF,
    FORWARD_VX_SIGN,
    LEG_JOINT_NAMES,
    REPO_ROOT,
    GaitParams,
    PlacoGaitIK,
    _rot_y,
    _rot_z,
    load_placo_robot,
)

DEFAULT_OUT = REPO_ROOT / "source/oceanisaaclab/assets/gaits/stand_pose.npz"


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


@dataclass
class StandParams:
    """站立躯干命令网格参数（4-DOF：Δh_torso / pitch / yaw / roll）。范围取站得稳的保守值。"""

    dh_max: float = 0.04
    """[m] 躯干高度偏移网格半幅（蹲下为负、升高为正；升高受腿伸直限制，见 dh_up_max）。"""

    dh_up_max: float = 0.01
    """[m] 升高方向单独上限（腿接近伸直，抬高余量小）。网格用 [-dh_max, dh_up_max]。"""

    pitch_max: float = 0.17
    """[rad] 躯干前后倾网格半幅。"""

    yaw_max: float = 0.24
    """[rad] 躯干原地偏航网格半幅（脚固定，靠髋 yaw 扭转，范围保守）。"""

    roll_max: float = 0.09
    """[rad] 躯干侧倾网格半幅（单足无踝 roll，侧倾靠髋，范围保守）。"""

    grid: int = 5
    """每轴网格点数。高度轴显式包含 0；默认 5⁴=625 个姿态。"""

    ik_iterations: int = 150
    """每个姿态 placo 迭代次数（连续 warm-start 下足够收敛）。"""

    def dh_grid(self) -> np.ndarray:
        if self.grid < 3:
            raise ValueError("stand pose grid must contain at least 3 points")
        # 高度范围不对称，直接对 [-0.04, 0.01] 做偶数间隔不会可靠命中 0。分别在
        # 负/正半轴采样，保证 exact zero command 对应一个真实网格点。
        n_negative = self.grid // 2
        n_positive = self.grid - n_negative - 1
        negative = np.linspace(-self.dh_max, 0.0, n_negative + 1)[:-1]
        positive = np.linspace(0.0, self.dh_up_max, n_positive + 1)[1:]
        return np.concatenate((negative, np.array([0.0]), positive))

    def pitch_grid(self) -> np.ndarray:
        return np.linspace(-self.pitch_max, self.pitch_max, self.grid)

    def yaw_grid(self) -> np.ndarray:
        return np.linspace(-self.yaw_max, self.yaw_max, self.grid)

    def roll_grid(self) -> np.ndarray:
        return np.linspace(-self.roll_max, self.roll_max, self.grid)

    def describe(self) -> str:
        lines = ["stand pose parameters:"]
        for f in fields(self):
            lines.append(f"  {f.name:20s} = {getattr(self, f.name)}")
        return "\n".join(lines)


def _foot_target(anchor_pos: np.ndarray, rot0: np.ndarray, R_bw: np.ndarray,
                 base_height: float, foot_origin_offset: float) -> np.ndarray:
    """站立单脚 base 系 6D 目标：脚固定地面锚点，躯干处于命令朝向 R_wb=R_bw^T、高度 base_height。"""
    # 脚世界锚点：站立 xy 保持 q=0 位置，脚原点 z = foot_origin_offset（脚底贴地）
    p_world = np.array([anchor_pos[0], anchor_pos[1], foot_origin_offset])
    target = np.eye(4)
    target[:3, 3] = R_bw @ (p_world - np.array([0.0, 0.0, base_height]))
    target[:3, :3] = R_bw @ rot0
    return target


def generate(params: StandParams, urdf: Path, mesh_dir: Path) -> dict:
    robot = load_placo_robot(urdf, mesh_dir)
    gait_params = GaitParams()  # 复用脚 task / 软限位 / foot_origin_offset 约定
    ik = PlacoGaitIK(robot, gait_params)
    anchors, rot0 = ik.anchors()  # q=0 时双脚在 base 系（=world，base 在原点）的位姿
    # 标称站高：脚底贴地 → base_height = foot_origin_offset − 脚原点 z（base 系，负值）
    # 与 walking reference 使用同一 FK 定义：两脚 q=0 锚点高度的平均值。
    base_height = float(np.mean([
        gait_params.foot_origin_offset - anchors[side][2] for side in "rl"
    ]))

    dh_grid = params.dh_grid()
    pitch_grid = params.pitch_grid()
    yaw_grid = params.yaw_grid()
    roll_grid = params.roll_grid()
    nh, npi, ny, nr = len(dh_grid), len(pitch_grid), len(yaw_grid), len(roll_grid)
    joint_pos = np.zeros((nh, npi, ny, nr, 10), dtype=np.float32)
    # Torso world xy is fixed at the base origin. The path frame converges to the solved
    # feet center/heading, so the torso reference generally has a non-zero forward offset.
    # This is a kinematic reference, not a CoP/ZMP estimate.
    base_pos_pf = np.zeros((nh, npi, ny, nr, 2), dtype=np.float32)
    # 两只脚的 URDF link frame 朝向不同（右脚约 π、左脚约 0）。保存 q=0
    # 世界 yaw，运行时先消去各自 link-frame 偏置，再求双脚平均 heading。
    foot_yaw_neutral = np.array(
        [np.arctan2(rot0[side][1, 0], rot0[side][0, 0]) for side in "rl"],
        dtype=np.float32,
    )
    base_yaw_pf = np.zeros((nh, npi, ny, nr), dtype=np.float32)
    head_yaw_offset = 0.0 if FORWARD_VX_SIGN > 0.0 else np.pi

    def solved_base_reference(
        height: float, pitch: float, yaw: float, roll: float
    ) -> tuple[np.ndarray, float]:
        """Torso xy/yaw in the calibrated path frame of the solved feet."""
        R_wb = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        foot_positions = []
        foot_headings = []
        for foot_id, side in enumerate("rl"):
            # Placo keeps base_link at identity; convert the solved base-frame foot
            # pose to the commanded torso world frame before path-frame calibration.
            T_bf = robot.get_T_world_frame(ik.FOOT_FRAMES[side])
            foot_positions.append(np.array([0.0, 0.0, height]) + R_wb @ T_bf[:3, 3])
            R_bf = T_bf[:3, :3]
            R_wf = R_wb @ R_bf
            raw_yaw = float(np.arctan2(R_wf[1, 0], R_wf[0, 0]))
            relative_yaw = _wrap_angle(raw_yaw - float(foot_yaw_neutral[foot_id]))
            foot_headings.append(relative_yaw + head_yaw_offset)
        feet_heading = float(np.arctan2(
            np.mean(np.sin(foot_headings)),
            np.mean(np.cos(foot_headings)),
        ))
        feet_center = np.mean(foot_positions, axis=0)
        rel_xy = -feet_center[:2]
        cos_y, sin_y = np.cos(feet_heading), np.sin(feet_heading)
        torso_pos_pf = np.array([
            rel_xy[0] * cos_y + rel_xy[1] * sin_y,
            -rel_xy[0] * sin_y + rel_xy[1] * cos_y,
        ])
        head_yaw = yaw + head_yaw_offset
        return torso_pos_pf, _wrap_angle(head_yaw - feet_heading)

    def solve_pose(dh: float, pitch: float, yaw: float, roll: float) -> tuple[np.ndarray, float]:
        R_wb = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)  # 躯干世界朝向
        R_bw = R_wb.T
        h = base_height + dh
        targets = {
            side: _foot_target(anchors[side], rot0[side], R_bw, h, gait_params.foot_origin_offset)
            for side in "rl"
        }
        return ik.solve(targets)

    worst = 0.0
    total = nh * npi * ny * nr
    done = 0
    # 4 维蛇行 + 连续 warm-start（与 neck-head 生成器同思路：相邻姿态只差一步，
    # placo 从上一解 warm-start，保证腿角映射平滑单调、无 IK 分枝跳变）。
    ik.reset_zero()
    for ih in range(nh):
        ip_range = range(npi) if ih % 2 == 0 else range(npi - 1, -1, -1)
        for ip in ip_range:
            iy_range = range(ny) if (ih + ip) % 2 == 0 else range(ny - 1, -1, -1)
            for iy in iy_range:
                ir_range = range(nr) if (ih + ip + iy) % 2 == 0 else range(nr - 1, -1, -1)
                for ir in ir_range:
                    q, err = solve_pose(
                        float(dh_grid[ih]), float(pitch_grid[ip]),
                        float(yaw_grid[iy]), float(roll_grid[ir]),
                    )
                    joint_pos[ih, ip, iy, ir] = q
                    pose_pos_pf, pose_yaw_pf = solved_base_reference(
                        base_height + float(dh_grid[ih]),
                        float(pitch_grid[ip]),
                        float(yaw_grid[iy]),
                        float(roll_grid[ir]),
                    )
                    base_pos_pf[ih, ip, iy, ir] = pose_pos_pf
                    base_yaw_pf[ih, ip, iy, ir] = pose_yaw_pf
                    worst = max(worst, err)
                    done += 1
        print(f"  [{done}/{total}] dh={dh_grid[ih]:+.3f} done, worst foot err so far {worst*1000:.2f} mm")

    print(f"worst foot position residual: {worst*1000:.3f} mm")
    if worst > 0.005:
        print("WARNING: stand IK residual above 5 mm at command-grid edges; review command limits.")
    zero_idx = (
        int(np.flatnonzero(np.isclose(dh_grid, 0.0))[0]),
        int(np.flatnonzero(np.isclose(pitch_grid, 0.0))[0]),
        int(np.flatnonzero(np.isclose(yaw_grid, 0.0))[0]),
        int(np.flatnonzero(np.isclose(roll_grid, 0.0))[0]),
    )
    # Shared stand/walk handoff state: the URDF q=0 pose is the canonical zero-speed
    # stance. Do not retain solver residuals at the exact neutral command.
    joint_pos[zero_idx] = 0.0
    neutral_feet_center = np.mean([anchors[side][:2] for side in "rl"], axis=0)
    neutral_rel = -neutral_feet_center
    cos_y, sin_y = np.cos(head_yaw_offset), np.sin(head_yaw_offset)
    base_pos_pf[zero_idx] = np.array([
        neutral_rel[0] * cos_y + neutral_rel[1] * sin_y,
        -neutral_rel[0] * sin_y + neutral_rel[1] * cos_y,
    ])
    base_yaw_pf[zero_idx] = 0.0
    print(f"nominal base_height = {base_height:.4f} m")
    return dict(
        torso_h_grid=dh_grid.astype(np.float32),
        torso_pitch_grid=pitch_grid.astype(np.float32),
        torso_yaw_grid=yaw_grid.astype(np.float32),
        torso_roll_grid=roll_grid.astype(np.float32),
        joint_pos=joint_pos,
        base_pos_pf=base_pos_pf,
        base_yaw_pf=base_yaw_pf,
        foot_yaw_neutral=foot_yaw_neutral,
        joint_names=np.array(LEG_JOINT_NAMES),
        base_height=np.float32(base_height),
        foot_origin_offset=np.float32(gait_params.foot_origin_offset),
    )


def _viz_sweep(params: StandParams, urdf: Path, mesh_dir: Path, loops: int, hold: float) -> None:
    """meshcat 里逐轴扫掠躯干命令(其余轴归零)，实时显示解出的站姿。用系统 python3 + meshcat。"""
    import time

    from placo_utils.visualization import robot_viz

    robot = load_placo_robot(urdf, mesh_dir)
    gait_params = GaitParams()
    ik = PlacoGaitIK(robot, gait_params)
    anchors, rot0 = ik.anchors()
    base_height = gait_params.foot_origin_offset - float(min(anchors["r"][2], anchors["l"][2]))
    viz = robot_viz(robot)

    def show(dh, pitch, yaw, roll, label):
        R_bw = (_rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)).T
        h = base_height + dh
        targets = {
            side: _foot_target(anchors[side], rot0[side], R_bw, h, gait_params.foot_origin_offset)
            for side in "rl"
        }
        q, _ = ik.solve(targets)
        viz.display(robot.state.q)
        print(f"  [viz] {label:22s} legq={np.round(q, 3).tolist()}")
        time.sleep(hold)

    axes = [
        ("h_torso", params.dh_grid(), 0),
        ("pitch", params.pitch_grid(), 1),
        ("yaw", params.yaw_grid(), 2),
        ("roll", params.roll_grid(), 3),
    ]
    print(f"meshcat viz: 逐轴扫掠 {loops} 遍（双脚固定地面，只动躯干姿态）")
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


def main() -> None:
    defaults = StandParams()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dh-max", type=float, default=defaults.dh_max)
    parser.add_argument("--dh-up-max", type=float, default=defaults.dh_up_max)
    parser.add_argument("--pitch-max", type=float, default=defaults.pitch_max)
    parser.add_argument("--yaw-max", type=float, default=defaults.yaw_max)
    parser.add_argument("--roll-max", type=float, default=defaults.roll_max)
    parser.add_argument("--grid", type=int, default=defaults.grid)
    parser.add_argument("--viz", action="store_true", help="生成后 meshcat 逐轴扫掠回放确认")
    parser.add_argument("--viz-only", action="store_true", help="只可视化、不重新生成/保存")
    parser.add_argument("--viz-loops", type=int, default=2)
    parser.add_argument("--viz-hold", type=float, default=0.03)
    args = parser.parse_args()

    params = StandParams(
        dh_max=args.dh_max, dh_up_max=args.dh_up_max, pitch_max=args.pitch_max,
        yaw_max=args.yaw_max, roll_max=args.roll_max, grid=args.grid,
    )
    print(params.describe())
    if not args.viz_only:
        data = generate(params, args.urdf, args.mesh_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
        print(f"saved stand pose lib → {args.out}  (grid {args.grid}^4 = {args.grid**4} poses)")
    if args.viz or args.viz_only:
        _viz_sweep(params, args.urdf, args.mesh_dir, args.viz_loops, args.viz_hold)


if __name__ == "__main__":
    main()
