# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""参考步态库播放器：用 meshcat 可视化回放 gen_reference_gait.py 生成的 npz 库。

对任意 (vx, vy, wz) 命令做与训练环境完全相同的三线性插值 + 相位环形插值采样
（reference_gait.py 的 numpy 版），机器人基座按命令速度在世界系里真实行进
（位置积分 vx/vy、偏航积分 wz），可以直观确认步幅 / 抬腿高度 / 接触时序 / 前倾
是否符合预期。绿/红小球标记双脚参考接触状态（绿 = 支撑，红 = 摆动）。

依赖：pip install placo meshcat

用法示例：

    # 满速前进（打开日志里打印的 meshcat URL 观看）
    python scripts/play_reference_gait.py --vx 0.25

    # 网格点之间的插值命令（前进 + 左移 + 左转）
    python scripts/play_reference_gait.py --vx 0.15 --vy 0.1 --wz 0.4

    # 原地回放（基座不行进）、0.25 倍速慢放
    python scripts/play_reference_gait.py --vx 0.25 --in-place --speed 0.25

    # 只打印库的参数信息
    python scripts/play_reference_gait.py --info
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gen_reference_gait import (
    DEFAULT_MESH_DIR,
    DEFAULT_OUT,
    DEFAULT_URDF,
    FORWARD_VX_SIGN,
    LEG_JOINT_NAMES,
    _rot_y,
    _rot_z,
    load_placo_robot,
)


class ReferenceGaitNumpy:
    """npz 步态库的 numpy 采样器（训练端 reference_gait.py 三线性插值的单命令版）。"""

    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=False)
        self.joint_pos = data["joint_pos"]
        self.joint_vel = data["joint_vel"]
        self.feet_contact = data["feet_contact"]
        self.base_height = data["base_height"]
        self.base_pitch = data["base_pitch"]
        self.grids = (data["vx_grid"], data["vy_grid"], data["wz_grid"])
        self.gait_period = float(data["gait_period"])
        self.gait_duty = float(data["gait_duty"])
        self.foot_clearance = float(data["foot_clearance"])
        self.num_phase = self.joint_pos.shape[3]
        self.joint_names = [str(n) for n in data["joint_names"]]

    def describe(self) -> str:
        vx, vy, wz = self.grids
        return "\n".join(
            [
                "reference gait library:",
                f"  gait_period    = {self.gait_period} s",
                f"  gait_duty      = {self.gait_duty}",
                f"  foot_clearance = {self.foot_clearance} m",
                f"  base_height    = {float(self.base_height[0, 0, 0]):.4f} m (standing)",
                f"  phase samples  = {self.num_phase}",
                f"  vx grid        = {vx}",
                f"  vy grid        = {vy}",
                f"  wz grid        = {wz}",
                f"  step_length(vx)= vx * {self.gait_period * self.gait_duty:.2f} m",
                f"  joint order    = {self.joint_names}",
            ]
        )

    @staticmethod
    def _coords(value: float, grid: np.ndarray) -> tuple[int, int, float]:
        value = float(np.clip(value, grid[0], grid[-1]))
        i1 = int(np.clip(np.searchsorted(grid, value), 1, len(grid) - 1))
        i0 = i1 - 1
        w = (value - grid[i0]) / (grid[i1] - grid[i0])
        return i0, i1, w

    def sample(self, cmd: np.ndarray, phase: float) -> dict[str, np.ndarray | float]:
        """按命令 (vx, vy, wz) 与相位 φ∈[0,1) 采样一帧参考量（与训练端插值一致）。"""
        coords = [self._coords(cmd[a], self.grids[a]) for a in range(3)]
        p = (phase % 1.0) * self.num_phase
        ip0 = int(p) % self.num_phase
        ip1 = (ip0 + 1) % self.num_phase
        wp = p - int(p)

        def lerp(table: np.ndarray, with_phase: bool) -> np.ndarray:
            out = 0.0
            for cx, fx in ((coords[0][0], 1 - coords[0][2]), (coords[0][1], coords[0][2])):
                for cy, fy in ((coords[1][0], 1 - coords[1][2]), (coords[1][1], coords[1][2])):
                    for cz, fz in ((coords[2][0], 1 - coords[2][2]), (coords[2][1], coords[2][2])):
                        corner = table[cx, cy, cz]
                        frame = corner[ip0] * (1 - wp) + corner[ip1] * wp if with_phase else corner
                        out = out + fx * fy * fz * frame
            return out

        return {
            "joint_pos": lerp(self.joint_pos, True),
            "feet_contact": lerp(self.feet_contact, True),
            "base_height": float(lerp(self.base_height, False)),
            "base_pitch": float(lerp(self.base_pitch, False)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_OUT, help="步态库 npz 路径")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--vx", type=float, default=0.0, help="[m/s] 头前向速度命令")
    parser.add_argument("--vy", type=float, default=0.0, help="[m/s] 头左向速度命令")
    parser.add_argument("--wz", type=float, default=0.0, help="[rad/s] 逆时针偏航命令")
    parser.add_argument("--speed", type=float, default=1.0, help="回放倍速（0.25 = 慢放 4 倍）")
    parser.add_argument("--duration", type=float, default=0.0, help="[s] 回放时长；<=0 = 无限循环")
    parser.add_argument("--in-place", action="store_true", help="基座固定原地回放（不积分行进）")
    parser.add_argument("--info", action="store_true", help="只打印库参数信息后退出")
    args = parser.parse_args()

    gait = ReferenceGaitNumpy(args.npz)
    print(gait.describe())
    if args.info:
        return

    cmd = np.array([args.vx, args.vy, args.wz])
    print(f"\nplaying cmd (vx={cmd[0]:+.3f}, vy={cmd[1]:+.2f}, wz={cmd[2]:+.2f})  "
          f"step≈{abs(cmd[0]) * gait.gait_period * gait.gait_duty * 100:.1f} cm  Ctrl-C 退出")

    from placo_utils.visualization import point_viz, robot_frame_viz, robot_viz

    robot = load_placo_robot(args.urdf, args.mesh_dir)
    viz = robot_viz(robot, "ocean")

    dt = gait.gait_period / gait.num_phase
    v_base = FORWARD_VX_SIGN * cmd[:2]  # 命令是头部系，base +x 指尾部
    pos_w = np.zeros(2)
    yaw = 0.0
    t = 0.0
    try:
        while args.duration <= 0.0 or t < args.duration:
            frame = gait.sample(cmd, t / gait.gait_period)
            for j, name in enumerate(LEG_JOINT_NAMES):
                robot.set_joint(name, float(frame["joint_pos"][j]))
            T = np.eye(4)
            T[:3, :3] = _rot_z(yaw) @ _rot_y(frame["base_pitch"])
            T[:2, 3] = pos_w
            T[2, 3] = frame["base_height"]
            robot.set_T_world_frame("base_link", T)
            robot.update_kinematics()
            viz.display(robot.state.q)
            for col, foot in enumerate(("leg_r5_link", "leg_l5_link")):
                robot_frame_viz(robot, foot)
                contact = frame["feet_contact"][col] > 0.5
                p = robot.get_T_world_frame(foot)[:3, 3] + np.array([0.0, 0.0, 0.05])
                point_viz(f"contact_{foot}", p, radius=0.012, color=0x00FF00 if contact else 0xFF0000)
            if not args.in_place:
                pos_w = pos_w + _rot_z(yaw)[:2, :2] @ v_base * dt
                yaw += cmd[2] * dt
            t += dt
            time.sleep(dt / max(args.speed, 1e-3))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
