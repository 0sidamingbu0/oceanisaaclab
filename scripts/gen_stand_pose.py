# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""站立姿态参考库离线生成器（路线 B 论文复刻：perpetual 站立策略）。

论文把运动分成独立策略（divide-and-conquer）：periodic 行走策略（已实现）+ perpetual
站立策略。站立策略命令 g_perp = (Δh_head, Δθ_head, h_torso, θ_torso)（式 5）——躯干的
高度/朝向 + 头部的高度/朝向，**无相位**，脚不迈步。本脚本按论文 Section V-A 生成
head+torso 联合静态参考：双脚固定在地面锚点，调整躯干平移使全身 CoM 投影保持在 neutral
CoM 标定目标（该目标位于/接近双脚支撑中心），并相应求解 10 个腿关节角。

直接保存 5^8 个联合网格既慢，训练时 256 角点插值也会显著拖慢采样。因此资产采用
factorized 近似表示：5^4 torso 平衡姿态 + 5^4 head 全身 CoM 偏移，并在每个 torso 节点保存静态平衡
和腿 IK 对基座平移的局部雅可比。运行时 StandPose 同时输入 torso4/head4，组合出腿角与
躯干 path-frame xy；头部四个关节仍由同一个 neck_head_map.npz 给出。

约定（与 gen_reference_gait.py / 训练环境一致）：
- 复用 gen_reference_gait 的 load_placo_robot / PlacoGaitIK / GaitParams / LEG_JOINT_NAMES；
- base_link 固定世界原点、脚目标在 base 系表达：p_base = R_bw @ (p_world − [0,0,h_torso])，
  其中 R_bw = R_torso^T（R_torso = Rz(yaw)Ry(pitch)Rx(roll) 为躯干世界朝向）；
- 脚世界锚点 = q=0 标称站立时脚原点的世界 xy + foot_origin_offset（脚底贴地）；
- 躯干高度 h_torso = 标称站高 base_height + Δh_torso 命令偏移。
- 精确零命令保持 URDF q=0，与 walking 零速参考完全一致；足底几何中心和 neutral CoM 的
  约 1--2 mm 标定差作为允许误差，不为追求数值零误差破坏策略切换姿态。

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
DEFAULT_NECK_MAP = REPO_ROOT / "source/oceanisaaclab/assets/gaits/neck_head_map.npz"


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

    balance_iterations: int = 8
    """每个 torso 节点的静态 CoM 平衡迭代上限。"""

    balance_tolerance: float = 2.0e-5
    """[m] torso 网格节点的静态 CoM 平衡收敛阈值。"""

    balance_gain: float = 1.25
    """CoM 残差到躯干 xy 修正的固定点增益；本机数值雅可比约为 0.8I。"""

    jacobian_epsilon: float = 1.0e-3
    """[m] 计算平衡/腿 IK 局部雅可比的中心差分步长。"""

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


def _foot_target(
    anchor_pos: np.ndarray,
    rot0: np.ndarray,
    R_bw: np.ndarray,
    base_pos: np.ndarray,
    foot_origin_offset: float,
) -> np.ndarray:
    """站立单脚 base 系 6D 目标：脚固定地面锚点，躯干处于命令朝向 R_wb=R_bw^T、高度 base_height。"""
    # 脚世界锚点：站立 xy 保持 q=0 位置，脚原点 z = foot_origin_offset（脚底贴地）
    p_world = np.array([anchor_pos[0], anchor_pos[1], foot_origin_offset])
    target = np.eye(4)
    target[:3, 3] = R_bw @ (p_world - base_pos)
    target[:3, :3] = R_bw @ rot0
    return target


def _load_neck_map(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {
        "dh_grid", "pitch_grid", "yaw_grid", "roll_grid", "neck_pos", "neck_joint_names"
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"neck map is missing fields: {sorted(missing)}")
    names = [str(name) for name in data["neck_joint_names"]]
    if names != [f"neck_n{i}_joint" for i in range(1, 5)]:
        raise ValueError(f"unexpected neck joint order in {path}: {names}")
    return {name: np.asarray(data[name]) for name in data.files}


def _sole_support_center(robot, mesh_dir: Path, low_layer: float = 0.002) -> np.ndarray:
    """Estimate the neutral double-support center from the two sole collision meshes.

    Each foot is filtered relative to its own lowest layer because the exported left/right
    meshes have an approximately 10 mm vertical offset. Only xy is used; the result is a
    calibration/validation value, not a runtime contact estimate.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("gen_stand_pose.py requires trimesh to calibrate the soles") from exc

    low_points = []
    for side in "rl":
        mesh_path = mesh_dir / f"leg_{side}5_link.STL"
        vertices = np.asarray(trimesh.load_mesh(mesh_path, process=False).vertices)
        T_bf = robot.get_T_world_frame(f"leg_{side}5_link")
        points_b = vertices @ T_bf[:3, :3].T + T_bf[:3, 3]
        low = points_b[:, 2] <= float(points_b[:, 2].min()) + low_layer
        if not np.any(low):
            raise RuntimeError(f"no sole vertices selected from {mesh_path}")
        low_points.append(points_b[low, :2])
    support = np.concatenate(low_points, axis=0)
    return 0.5 * (support.min(axis=0) + support.max(axis=0))


def _set_joints(robot, names: list[str], values: np.ndarray) -> None:
    for name, value in zip(names, values, strict=True):
        robot.set_joint(name, float(value))


def generate(params: StandParams, urdf: Path, mesh_dir: Path, neck_map_path: Path) -> dict:
    robot = load_placo_robot(urdf, mesh_dir)
    gait_params = GaitParams()  # 复用脚 task / 软限位 / foot_origin_offset 约定
    gait_params.ik_iterations = params.ik_iterations
    ik = PlacoGaitIK(robot, gait_params)
    anchors, rot0 = ik.anchors()  # q=0 时双脚在 base 系（=world，base 在原点）的位姿
    # 标称站高：脚底贴地 → base_height = foot_origin_offset − 脚原点 z（base 系，负值）
    # 与 walking reference 使用同一 FK 定义：两脚 q=0 锚点高度的平均值。
    base_height = float(np.mean([
        gait_params.foot_origin_offset - anchors[side][2] for side in "rl"
    ]))
    neck_map = _load_neck_map(neck_map_path)
    head_grids = [
        neck_map["dh_grid"], neck_map["pitch_grid"],
        neck_map["yaw_grid"], neck_map["roll_grid"],
    ]
    neck_pos = np.asarray(neck_map["neck_pos"], dtype=np.float64)
    head_zero_idx = tuple(
        int(np.flatnonzero(np.isclose(grid, 0.0))[0]) for grid in head_grids
    )
    neutral_neck = neck_pos[head_zero_idx]

    # The q=0 full-body CoM is already within about 1--2 mm of the collision-sole bbox
    # center. Use it as the balance target so the exact shared stand/walk neutral remains
    # untouched, while retaining the sole center as an auditable calibration value.
    ik.reset_zero()
    _set_joints(robot, [f"neck_n{i}_joint" for i in range(1, 5)], neutral_neck)
    robot.update_kinematics()
    neutral_com_b = np.asarray(robot.com_world(), dtype=np.float64)
    total_mass = float(robot.total_mass())
    sole_support_center_b = _sole_support_center(robot, mesh_dir)
    balance_target_xy = neutral_com_b[:2].copy()
    print(
        "neutral full-body CoM xy = "
        f"{np.round(neutral_com_b[:2] * 1000.0, 3).tolist()} mm; "
        "sole support center xy = "
        f"{np.round(sole_support_center_b * 1000.0, 3).tolist()} mm"
    )
    if np.linalg.norm(neutral_com_b[:2] - sole_support_center_b) > 0.005:
        raise RuntimeError(
            "neutral CoM is more than 5 mm from the calibrated sole support center; "
            "review sole frames before generating a reference"
        )

    # Head command -> change of the *full-body* CoM in base coordinates. This is
    # independent of the leg configuration because the neck is a separate branch.
    head_com_offset_b = np.zeros((*neck_pos.shape[:-1], 3), dtype=np.float32)
    for index in np.ndindex(neck_pos.shape[:-1]):
        ik.reset_zero()
        _set_joints(robot, [f"neck_n{i}_joint" for i in range(1, 5)], neck_pos[index])
        robot.update_kinematics()
        head_com_offset_b[index] = np.asarray(robot.com_world()) - neutral_com_b
    head_com_offset_b[head_zero_idx] = 0.0

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
    base_xy_w = np.zeros((nh, npi, ny, nr, 2), dtype=np.float32)
    # Torso-node Jacobians map a head-induced full-body CoM change in base coordinates
    # to the additional balanced leg angles / torso path-frame xy.
    head_joint_pos_jac = np.zeros((nh, npi, ny, nr, 10, 3), dtype=np.float32)
    head_base_pos_pf_jac = np.zeros((nh, npi, ny, nr, 2, 3), dtype=np.float32)
    head_base_xy_w_jac = np.zeros((nh, npi, ny, nr, 2, 3), dtype=np.float32)
    # 两只脚的 URDF link frame 朝向不同（右脚约 π、左脚约 0）。保存 q=0
    # 世界 yaw，运行时先消去各自 link-frame 偏置，再求双脚平均 heading。
    foot_yaw_neutral = np.array(
        [np.arctan2(rot0[side][1, 0], rot0[side][0, 0]) for side in "rl"],
        dtype=np.float32,
    )
    base_yaw_pf = np.zeros((nh, npi, ny, nr), dtype=np.float32)
    head_yaw_offset = 0.0 if FORWARD_VX_SIGN > 0.0 else np.pi

    def solved_base_reference(
        base_xy: np.ndarray, height: float, pitch: float, yaw: float, roll: float
    ) -> tuple[np.ndarray, float]:
        """Torso xy/yaw in the calibrated path frame of the solved feet."""
        R_wb = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        foot_positions = []
        foot_headings = []
        for foot_id, side in enumerate("rl"):
            # Placo keeps base_link at identity; convert the solved base-frame foot
            # pose to the commanded torso world frame before path-frame calibration.
            T_bf = robot.get_T_world_frame(ik.FOOT_FRAMES[side])
            foot_positions.append(np.array([base_xy[0], base_xy[1], height]) + R_wb @ T_bf[:3, 3])
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
        rel_xy = base_xy - feet_center[:2]
        cos_y, sin_y = np.cos(feet_heading), np.sin(feet_heading)
        torso_pos_pf = np.array([
            rel_xy[0] * cos_y + rel_xy[1] * sin_y,
            -rel_xy[0] * sin_y + rel_xy[1] * cos_y,
        ])
        head_yaw = yaw + head_yaw_offset
        return torso_pos_pf, _wrap_angle(head_yaw - feet_heading)

    def solve_pose(
        dh: float, pitch: float, yaw: float, roll: float, base_xy: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray]:
        R_wb = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)  # 躯干世界朝向
        R_bw = R_wb.T
        h = base_height + dh
        base_pos = np.array([base_xy[0], base_xy[1], h])
        targets = {
            side: _foot_target(
                anchors[side], rot0[side], R_bw, base_pos, gait_params.foot_origin_offset
            )
            for side in "rl"
        }
        q, foot_err = ik.solve(targets)
        _set_joints(robot, [f"neck_n{i}_joint" for i in range(1, 5)], neutral_neck)
        robot.update_kinematics()
        com_w = base_pos + R_wb @ np.asarray(robot.com_world())
        return q, foot_err, com_w[:2] - balance_target_xy

    def solve_balanced_pose(
        dh: float, pitch: float, yaw: float, roll: float, initial_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        base_xy = np.asarray(initial_xy, dtype=np.float64).copy()
        for _ in range(params.balance_iterations):
            q, foot_err, balance_err = solve_pose(dh, pitch, yaw, roll, base_xy)
            if np.linalg.norm(balance_err) <= params.balance_tolerance:
                break
            base_xy -= params.balance_gain * balance_err
        else:
            q, foot_err, balance_err = solve_pose(dh, pitch, yaw, roll, base_xy)
        return q, base_xy, foot_err, float(np.linalg.norm(balance_err))

    def local_head_jacobians(
        dh: float,
        pitch: float,
        yaw: float,
        roll: float,
        base_xy: np.ndarray,
        feet_heading: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eps = params.jacobian_epsilon
        d_balance_d_xy = np.zeros((2, 2), dtype=np.float64)
        d_q_d_xy = np.zeros((10, 2), dtype=np.float64)
        for axis in range(2):
            delta = np.zeros(2)
            delta[axis] = eps
            q_plus, _, f_plus = solve_pose(dh, pitch, yaw, roll, base_xy + delta)
            q_minus, _, f_minus = solve_pose(dh, pitch, yaw, roll, base_xy - delta)
            d_balance_d_xy[:, axis] = (f_plus - f_minus) / (2.0 * eps)
            d_q_d_xy[:, axis] = (q_plus - q_minus) / (2.0 * eps)
        # Restore the exact center solution for base reference extraction and warm-start.
        solve_pose(dh, pitch, yaw, roll, base_xy)
        R_wb = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        # Head table stores a full-body CoM delta in base coordinates. Rotate it into world
        # xy, then solve the linearized static-balance equation for torso translation.
        d_base_xy_d_head_com = -np.linalg.solve(d_balance_d_xy, R_wb[:2, :])
        d_joint_d_head_com = d_q_d_xy @ d_base_xy_d_head_com
        c, s = np.cos(feet_heading), np.sin(feet_heading)
        R_pf_w = np.array([[c, s], [-s, c]])
        d_base_pf_d_head_com = R_pf_w @ d_base_xy_d_head_com
        # Guard against a singular IK/balance node leaking extreme values into training.
        if not np.all(np.isfinite(d_joint_d_head_com)) or not np.all(
            np.isfinite(d_base_pf_d_head_com)
        ):
            raise RuntimeError("non-finite head balance Jacobian")
        return d_joint_d_head_com, d_base_pf_d_head_com, d_base_xy_d_head_com

    worst = 0.0
    worst_balance = 0.0
    total = nh * npi * ny * nr
    done = 0
    # 4 维蛇行 + 连续 warm-start（与 neck-head 生成器同思路：相邻姿态只差一步，
    # placo 从上一解 warm-start，保证腿角映射平滑单调、无 IK 分枝跳变）。
    ik.reset_zero()
    warm_base_xy = np.zeros(2)
    for ih in range(nh):
        ip_range = range(npi) if ih % 2 == 0 else range(npi - 1, -1, -1)
        for ip in ip_range:
            iy_range = range(ny) if (ih + ip) % 2 == 0 else range(ny - 1, -1, -1)
            for iy in iy_range:
                ir_range = range(nr) if (ih + ip + iy) % 2 == 0 else range(nr - 1, -1, -1)
                for ir in ir_range:
                    q, pose_base_xy, err, balance_err = solve_balanced_pose(
                        float(dh_grid[ih]), float(pitch_grid[ip]),
                        float(yaw_grid[iy]), float(roll_grid[ir]),
                        warm_base_xy,
                    )
                    warm_base_xy = pose_base_xy
                    joint_pos[ih, ip, iy, ir] = q
                    base_xy_w[ih, ip, iy, ir] = pose_base_xy
                    pose_pos_pf, pose_yaw_pf = solved_base_reference(
                        pose_base_xy,
                        base_height + float(dh_grid[ih]),
                        float(pitch_grid[ip]),
                        float(yaw_grid[iy]),
                        float(roll_grid[ir]),
                    )
                    base_pos_pf[ih, ip, iy, ir] = pose_pos_pf
                    base_yaw_pf[ih, ip, iy, ir] = pose_yaw_pf
                    joint_jac, base_jac, base_xy_jac = local_head_jacobians(
                        float(dh_grid[ih]), float(pitch_grid[ip]),
                        float(yaw_grid[iy]), float(roll_grid[ir]),
                        pose_base_xy, float(
                            float(yaw_grid[iy]) + head_yaw_offset - pose_yaw_pf
                        ),
                    )
                    head_joint_pos_jac[ih, ip, iy, ir] = joint_jac
                    head_base_pos_pf_jac[ih, ip, iy, ir] = base_jac
                    head_base_xy_w_jac[ih, ip, iy, ir] = base_xy_jac
                    worst = max(worst, err)
                    worst_balance = max(worst_balance, balance_err)
                    done += 1
        print(f"  [{done}/{total}] dh={dh_grid[ih]:+.3f} done, worst foot err so far {worst*1000:.2f} mm")

    print(f"worst foot position residual: {worst*1000:.3f} mm")
    print(f"worst neutral-head CoM residual: {worst_balance*1000:.3f} mm")
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
    base_xy_w[zero_idx] = 0.0
    neutral_feet_center = np.mean([anchors[side][:2] for side in "rl"], axis=0)
    neutral_rel = -neutral_feet_center
    cos_y, sin_y = np.cos(head_yaw_offset), np.sin(head_yaw_offset)
    base_pos_pf[zero_idx] = np.array([
        neutral_rel[0] * cos_y + neutral_rel[1] * sin_y,
        -neutral_rel[0] * sin_y + neutral_rel[1] * cos_y,
    ])
    base_yaw_pf[zero_idx] = 0.0

    # Validate the factorization at every torso node and the 16 corners of the head
    # command hypercube (plus neutral). This exercises combined commands without solving
    # or storing the full 5^8 table.
    head_corner_indices = [head_zero_idx]
    head_corner_indices.extend(
        tuple(index)
        for index in np.ndindex(*(2,) * 4)
    )
    head_corner_indices = [
        tuple(
            head_zero_idx[axis]
            if index == head_zero_idx
            else (0 if index[axis] == 0 else len(head_grids[axis]) - 1)
            for axis in range(4)
        )
        for index in head_corner_indices
    ]
    # Deduplicate neutral if a future one-point head axis is used.
    head_corner_indices = list(dict.fromkeys(head_corner_indices))
    worst_combined_com = 0.0
    worst_combined_foot = 0.0
    worst_combined_label = None
    for torso_index in np.ndindex(joint_pos.shape[:-1]):
        ih, ip, iy, ir = torso_index
        torso_values = (
            float(dh_grid[ih]), float(pitch_grid[ip]),
            float(yaw_grid[iy]), float(roll_grid[ir]),
        )
        dh, pitch, yaw, roll = torso_values
        R_wb = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        for head_index in head_corner_indices:
            head_delta = np.asarray(head_com_offset_b[head_index], dtype=np.float64)
            q = np.asarray(joint_pos[torso_index], dtype=np.float64) + np.asarray(
                head_joint_pos_jac[torso_index], dtype=np.float64
            ) @ head_delta
            pose_xy = np.asarray(base_xy_w[torso_index], dtype=np.float64) + np.asarray(
                head_base_xy_w_jac[torso_index], dtype=np.float64
            ) @ head_delta
            robot.set_T_world_frame("base_link", np.eye(4))
            _set_joints(robot, LEG_JOINT_NAMES, q)
            _set_joints(
                robot,
                [f"neck_n{i}_joint" for i in range(1, 5)],
                neck_pos[head_index],
            )
            robot.update_kinematics()
            pose = np.array([pose_xy[0], pose_xy[1], base_height + dh])
            com_w = pose + R_wb @ np.asarray(robot.com_world())
            com_err = float(np.linalg.norm(com_w[:2] - balance_target_xy))
            foot_err = 0.0
            for side in "rl":
                foot_b = robot.get_T_world_frame(ik.FOOT_FRAMES[side])[:3, 3]
                foot_w = pose + R_wb @ foot_b
                target_w = np.array([
                    anchors[side][0], anchors[side][1], gait_params.foot_origin_offset
                ])
                foot_err = max(foot_err, float(np.linalg.norm(foot_w - target_w)))
            if max(com_err, foot_err) > max(worst_combined_com, worst_combined_foot):
                worst_combined_label = (torso_index, head_index)
            worst_combined_com = max(worst_combined_com, com_err)
            worst_combined_foot = max(worst_combined_foot, foot_err)
    print(
        "factorized head+torso validation: "
        f"max CoM error={worst_combined_com * 1000.0:.3f} mm, "
        f"max foot error={worst_combined_foot * 1000.0:.3f} mm "
        f"at {worst_combined_label}"
    )
    if worst_combined_com > 0.005 or worst_combined_foot > 0.005:
        print(
            "WARNING: factorized combined-reference error exceeds 5 mm; reduce command "
            "ranges or use a denser/nonlinear balance representation."
        )
    print(f"nominal base_height = {base_height:.4f} m")
    return dict(
        torso_h_grid=dh_grid.astype(np.float32),
        torso_pitch_grid=pitch_grid.astype(np.float32),
        torso_yaw_grid=yaw_grid.astype(np.float32),
        torso_roll_grid=roll_grid.astype(np.float32),
        joint_pos=joint_pos,
        base_pos_pf=base_pos_pf,
        base_xy_w=base_xy_w,
        base_yaw_pf=base_yaw_pf,
        head_dh_grid=np.asarray(head_grids[0], dtype=np.float32),
        head_pitch_grid=np.asarray(head_grids[1], dtype=np.float32),
        head_yaw_grid=np.asarray(head_grids[2], dtype=np.float32),
        head_roll_grid=np.asarray(head_grids[3], dtype=np.float32),
        head_com_offset_b=head_com_offset_b,
        head_joint_pos_jac=head_joint_pos_jac,
        head_base_pos_pf_jac=head_base_pos_pf_jac,
        head_base_xy_w_jac=head_base_xy_w_jac,
        balance_target_xy=balance_target_xy.astype(np.float32),
        sole_support_center_b=sole_support_center_b.astype(np.float32),
        neutral_com_b=neutral_com_b.astype(np.float32),
        total_mass=np.float32(total_mass),
        balance_method=np.array("factorized_full_body_com_v1"),
        foot_yaw_neutral=foot_yaw_neutral,
        joint_names=np.array(LEG_JOINT_NAMES),
        base_height=np.float32(base_height),
        foot_origin_offset=np.float32(gait_params.foot_origin_offset),
    )


def _sample_numpy4(table: np.ndarray, grids: list[np.ndarray], command: np.ndarray) -> np.ndarray:
    coordinates = []
    for value, grid in zip(command, grids, strict=True):
        value = float(np.clip(value, grid[0], grid[-1]))
        upper = int(np.searchsorted(grid, value, side="right"))
        upper = min(max(upper, 1), len(grid) - 1)
        lower = upper - 1
        width = float(grid[upper] - grid[lower])
        fraction = 0.0 if width <= 0.0 else (value - float(grid[lower])) / width
        coordinates.append((lower, upper, fraction))
    result = np.zeros(table.shape[4:], dtype=np.float64)
    for corner in np.ndindex(*(2,) * 4):
        index = []
        weight = 1.0
        for axis, high in enumerate(corner):
            lower, upper, fraction = coordinates[axis]
            index.append(upper if high else lower)
            weight *= fraction if high else 1.0 - fraction
        result += weight * np.asarray(table[tuple(index)], dtype=np.float64)
    return result


def _viz_sweep(
    stand_path: Path,
    neck_map_path: Path,
    urdf: Path,
    mesh_dir: Path,
    loops: int,
    hold: float,
) -> None:
    """Play the saved coupled reference while sweeping all torso and head axes."""
    import time

    from placo_utils.visualization import robot_viz

    stand = np.load(stand_path, allow_pickle=False)
    neck = np.load(neck_map_path, allow_pickle=False)
    robot = load_placo_robot(urdf, mesh_dir)
    gait_params = GaitParams()
    ik = PlacoGaitIK(robot, gait_params)
    anchors, _ = ik.anchors()
    base_height = float(stand["base_height"])
    torso_grids = [
        stand["torso_h_grid"], stand["torso_pitch_grid"],
        stand["torso_yaw_grid"], stand["torso_roll_grid"],
    ]
    head_grids = [
        neck["dh_grid"], neck["pitch_grid"], neck["yaw_grid"], neck["roll_grid"],
    ]
    balance_target = (
        np.asarray(stand["balance_target_xy"], dtype=np.float64)
        if "balance_target_xy" in stand.files
        else np.asarray(robot.com_world()[:2], dtype=np.float64)
    )
    viz = robot_viz(robot)

    def show(torso_cmd: np.ndarray, head_cmd: np.ndarray, label: str) -> None:
        leg_q = _sample_numpy4(stand["joint_pos"], torso_grids, torso_cmd)
        base_xy = (
            _sample_numpy4(stand["base_xy_w"], torso_grids, torso_cmd)
            if "base_xy_w" in stand.files
            else np.zeros(2)
        )
        if "head_com_offset_b" in stand.files:
            stand_head_grids = [
                stand["head_dh_grid"], stand["head_pitch_grid"],
                stand["head_yaw_grid"], stand["head_roll_grid"],
            ]
            head_com = _sample_numpy4(
                stand["head_com_offset_b"], stand_head_grids, head_cmd
            )
            leg_q += _sample_numpy4(
                stand["head_joint_pos_jac"], torso_grids, torso_cmd
            ) @ head_com
            if "head_base_xy_w_jac" in stand.files:
                base_xy += _sample_numpy4(
                    stand["head_base_xy_w_jac"], torso_grids, torso_cmd
                ) @ head_com
        neck_q = _sample_numpy4(neck["neck_pos"], head_grids, head_cmd)
        _set_joints(robot, LEG_JOINT_NAMES, leg_q)
        _set_joints(robot, [f"neck_n{i}_joint" for i in range(1, 5)], neck_q)
        dh, pitch, yaw, roll = torso_cmd
        T_wb = np.eye(4)
        T_wb[:3, :3] = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        T_wb[:3, 3] = [base_xy[0], base_xy[1], base_height + dh]
        robot.set_T_world_frame("base_link", T_wb)
        robot.update_kinematics()
        viz.display(robot.state.q)
        com_error_mm = 1000.0 * (np.asarray(robot.com_world()[:2]) - balance_target)
        foot_error = 0.0
        for side in "rl":
            foot_w = robot.get_T_world_frame(ik.FOOT_FRAMES[side])[:3, 3]
            target_w = np.array([
                anchors[side][0], anchors[side][1], gait_params.foot_origin_offset
            ])
            foot_error = max(foot_error, float(np.linalg.norm(foot_w - target_w)))
        print(
            f"  [viz] {label:24s} CoM_err_xy={np.round(com_error_mm, 2).tolist()} mm "
            f"foot_err={foot_error * 1000.0:.2f} mm"
        )
        time.sleep(hold)

    axes = [
        ("torso_h", torso_grids[0], "torso", 0),
        ("torso_pitch", torso_grids[1], "torso", 1),
        ("torso_yaw", torso_grids[2], "torso", 2),
        ("torso_roll", torso_grids[3], "torso", 3),
        ("head_h", head_grids[0], "head", 0),
        ("head_pitch", head_grids[1], "head", 1),
        ("head_yaw", head_grids[2], "head", 2),
        ("head_roll", head_grids[3], "head", 3),
    ]
    print(f"meshcat viz: 逐轴扫掠 {loops} 遍 head+torso 联合站立参考")
    for _ in range(loops):
        for name, grid, group, index in axes:
            fine = np.concatenate([
                np.linspace(0.0, grid[-1], 12),
                np.linspace(grid[-1], grid[0], 24),
                np.linspace(grid[0], 0.0, 12),
            ])
            for value in fine:
                torso_cmd = np.zeros(4)
                head_cmd = np.zeros(4)
                (torso_cmd if group == "torso" else head_cmd)[index] = float(value)
                show(torso_cmd, head_cmd, f"{name}={value:+.3f}")


def main() -> None:
    defaults = StandParams()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--neck-map", type=Path, default=DEFAULT_NECK_MAP)
    parser.add_argument("--dh-max", type=float, default=defaults.dh_max)
    parser.add_argument("--dh-up-max", type=float, default=defaults.dh_up_max)
    parser.add_argument("--pitch-max", type=float, default=defaults.pitch_max)
    parser.add_argument("--yaw-max", type=float, default=defaults.yaw_max)
    parser.add_argument("--roll-max", type=float, default=defaults.roll_max)
    parser.add_argument("--grid", type=int, default=defaults.grid)
    parser.add_argument("--ik-iterations", type=int, default=defaults.ik_iterations)
    parser.add_argument("--balance-iterations", type=int, default=defaults.balance_iterations)
    parser.add_argument("--viz", action="store_true", help="生成后 meshcat 逐轴扫掠回放确认")
    parser.add_argument("--viz-only", action="store_true", help="只可视化、不重新生成/保存")
    parser.add_argument("--viz-loops", type=int, default=2)
    parser.add_argument("--viz-hold", type=float, default=0.03)
    args = parser.parse_args()

    params = StandParams(
        dh_max=args.dh_max, dh_up_max=args.dh_up_max, pitch_max=args.pitch_max,
        yaw_max=args.yaw_max, roll_max=args.roll_max, grid=args.grid,
        ik_iterations=args.ik_iterations, balance_iterations=args.balance_iterations,
    )
    print(params.describe())
    if not args.viz_only:
        data = generate(params, args.urdf, args.mesh_dir, args.neck_map)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
        print(
            f"saved coupled stand pose lib → {args.out}  "
            f"(torso {args.grid}^4 + factorized head {data['head_com_offset_b'].shape[:-1]})"
        )
    if args.viz or args.viz_only:
        _viz_sweep(
            args.out, args.neck_map, args.urdf, args.mesh_dir,
            args.viz_loops, args.viz_hold,
        )


if __name__ == "__main__":
    main()
