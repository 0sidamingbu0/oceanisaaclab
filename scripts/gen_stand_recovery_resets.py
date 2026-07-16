# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate stable, non-canonical standing reset states for recovery training.

The generated poses are initial states only. The standing reference remains the neutral
``stand_pose.npz`` pose, so the policy must take a corrective step instead of treating a
narrow or staggered stance as a new target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gen_reference_gait import (
    DEFAULT_MESH_DIR,
    DEFAULT_URDF,
    LEG_JOINT_NAMES,
    REPO_ROOT,
    GaitParams,
    PlacoGaitIK,
    _rot_z,
    load_placo_robot,
)


DEFAULT_OUT = REPO_ROOT / "source/oceanisaaclab/assets/gaits/stand_recovery_resets.npz"
CATEGORY_NAMES = (
    "single_narrow",
    "symmetric_narrow",
    "single_wide",
    "staggered",
    "single_yaw",
)
CATEGORY_PROBABILITIES = np.array([0.50, 0.20, 0.15, 0.10, 0.05], dtype=np.float64)


@dataclass
class RecoveryResetParams:
    narrow_max: float = 0.050
    wide_max: float = 0.040
    stagger_max: float = 0.050
    yaw_max: float = 0.080
    ik_iterations: int = 150
    balance_iterations: int = 12
    balance_gain: float = 1.25
    balance_tolerance: float = 2.0e-5
    max_foot_error: float = 0.003


def _case_specs(params: RecoveryResetParams) -> list[dict]:
    fractions = (0.30, 0.50, 0.70, 1.00)
    cases: list[dict] = []

    def add(category: int, fraction: float, shifts=None, yaws=None) -> None:
        cases.append(
            {
                "category": category,
                "fraction": fraction,
                "shifts": shifts or {},
                "yaws": yaws or {},
            }
        )

    for fraction in fractions:
        narrow = params.narrow_max * fraction
        add(0, fraction, {"r": np.array([0.0, -narrow])})
        add(0, fraction, {"l": np.array([0.0, narrow])})
        add(
            1,
            fraction,
            {
                "r": np.array([0.0, -0.5 * narrow]),
                "l": np.array([0.0, 0.5 * narrow]),
            },
        )

        wide = params.wide_max * fraction
        add(2, fraction, {"r": np.array([0.0, wide])})
        add(2, fraction, {"l": np.array([0.0, -wide])})

        stagger = params.stagger_max * fraction
        for side in "rl":
            add(3, fraction, {side: np.array([stagger, 0.0])})
            add(3, fraction, {side: np.array([-stagger, 0.0])})

        yaw = params.yaw_max * fraction
        for side in "rl":
            add(4, fraction, yaws={side: yaw})
            add(4, fraction, yaws={side: -yaw})
    return cases


def generate(params: RecoveryResetParams, urdf: Path, mesh_dir: Path) -> dict[str, np.ndarray]:
    robot = load_placo_robot(urdf, mesh_dir)
    gait_params = GaitParams()
    gait_params.ik_iterations = params.ik_iterations
    ik = PlacoGaitIK(robot, gait_params)
    anchors, neutral_rotations = ik.anchors()
    base_height = float(
        np.mean(
            [gait_params.foot_origin_offset - anchors[side][2] for side in "rl"]
        )
    )

    ik.reset_zero()
    robot.update_kinematics()
    neutral_feet_center = np.mean([anchors[side][:2] for side in "rl"], axis=0)
    neutral_com_from_feet = np.asarray(robot.com_world())[:2] - neutral_feet_center
    head_yaw_offset = np.pi
    nominal_heading = head_yaw_offset
    nominal_delta = anchors["l"][:2] - anchors["r"][:2]
    nominal_foot_relative_xy = np.array(
        [
            np.cos(nominal_heading) * nominal_delta[0]
            + np.sin(nominal_heading) * nominal_delta[1],
            -np.sin(nominal_heading) * nominal_delta[0]
            + np.cos(nominal_heading) * nominal_delta[1],
        ]
    )
    neutral_foot_yaw = {
        side: float(
            np.arctan2(
                neutral_rotations[side][1, 0], neutral_rotations[side][0, 0]
            )
        )
        for side in "rl"
    }

    rows = []
    for case in _case_specs(params):
        foot_xy = {
            side: anchors[side][:2]
            + np.asarray(case["shifts"].get(side, np.zeros(2)), dtype=np.float64)
            for side in "rl"
        }
        target_com_xy = np.mean([foot_xy[side] for side in "rl"], axis=0)
        target_com_xy += neutral_com_from_feet
        base_xy = np.mean(
            [np.asarray(case["shifts"].get(side, np.zeros(2))) for side in "rl"],
            axis=0,
        )
        ik.reset_zero()

        for _ in range(params.balance_iterations):
            targets = {}
            for side in "rl":
                target = np.eye(4)
                target[:3, 3] = np.array(
                    [foot_xy[side][0] - base_xy[0], foot_xy[side][1] - base_xy[1], anchors[side][2]]
                )
                yaw_offset = float(case["yaws"].get(side, 0.0))
                target[:3, :3] = _rot_z(yaw_offset) @ neutral_rotations[side]
                targets[side] = target
            joint_pos, foot_error = ik.solve(targets)
            robot.update_kinematics()
            com_error = base_xy + np.asarray(robot.com_world())[:2] - target_com_xy
            if np.linalg.norm(com_error) <= params.balance_tolerance:
                break
            base_xy -= params.balance_gain * com_error

        if foot_error > params.max_foot_error:
            raise RuntimeError(
                f"{CATEGORY_NAMES[case['category']]} IK foot error {foot_error:.6f}m exceeds "
                f"{params.max_foot_error:.6f}m"
            )
        if np.linalg.norm(com_error) > params.balance_tolerance:
            raise RuntimeError(
                f"{CATEGORY_NAMES[case['category']]} balance error did not converge: "
                f"{np.linalg.norm(com_error):.6f}m"
            )

        solved_foot_xy = {}
        solved_foot_heading = {}
        for side in "rl":
            pose = robot.get_T_world_frame(ik.FOOT_FRAMES[side])
            solved_foot_xy[side] = base_xy + pose[:2, 3]
            raw_yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
            solved_foot_heading[side] = float(
                np.arctan2(
                    np.sin(raw_yaw - neutral_foot_yaw[side]),
                    np.cos(raw_yaw - neutral_foot_yaw[side]),
                )
            )
        calibrated_headings = np.array(
            [solved_foot_heading[side] + head_yaw_offset for side in "rl"]
        )
        feet_heading = float(
            np.arctan2(
                np.mean(np.sin(calibrated_headings)),
                np.mean(np.cos(calibrated_headings)),
            )
        )
        feet_center = np.mean([solved_foot_xy[side] for side in "rl"], axis=0)
        foot_delta = solved_foot_xy["l"] - solved_foot_xy["r"]
        cos_h, sin_h = np.cos(feet_heading), np.sin(feet_heading)
        foot_relative_xy = np.array(
            [
                cos_h * foot_delta[0] + sin_h * foot_delta[1],
                -sin_h * foot_delta[0] + cos_h * foot_delta[1],
            ]
        )
        base_delta = base_xy - feet_center
        base_pos_pf = np.array(
            [
                cos_h * base_delta[0] + sin_h * base_delta[1],
                -sin_h * base_delta[0] + cos_h * base_delta[1],
            ]
        )
        rows.append(
            {
                "joint_pos": joint_pos,
                "base_pos_xy": base_xy,
                "base_pos_pf": base_pos_pf,
                "feet_heading_yaw": feet_heading,
                "foot_relative_xy": foot_relative_xy,
                "foot_relative_yaw": float(
                    np.arctan2(
                        np.sin(solved_foot_heading["l"] - solved_foot_heading["r"]),
                        np.cos(solved_foot_heading["l"] - solved_foot_heading["r"]),
                    )
                ),
                "category": case["category"],
                "fraction": case["fraction"],
                "foot_error": foot_error,
                "balance_error": float(np.linalg.norm(com_error)),
            }
        )

    categories = np.array([row["category"] for row in rows], dtype=np.int64)
    sample_weight = np.zeros(len(rows), dtype=np.float64)
    for category, probability in enumerate(CATEGORY_PROBABILITIES):
        selected = categories == category
        sample_weight[selected] = probability / int(np.sum(selected))

    return {
        "joint_pos": np.asarray([row["joint_pos"] for row in rows], dtype=np.float32),
        "base_pos_xy": np.asarray([row["base_pos_xy"] for row in rows], dtype=np.float32),
        "base_pos_pf": np.asarray([row["base_pos_pf"] for row in rows], dtype=np.float32),
        "feet_heading_yaw": np.asarray(
            [row["feet_heading_yaw"] for row in rows], dtype=np.float32
        ),
        "foot_relative_xy": np.asarray(
            [row["foot_relative_xy"] for row in rows], dtype=np.float32
        ),
        "foot_relative_yaw": np.asarray(
            [row["foot_relative_yaw"] for row in rows], dtype=np.float32
        ),
        "category": categories,
        "category_names": np.asarray(CATEGORY_NAMES),
        "curriculum_fraction": np.asarray(
            [row["fraction"] for row in rows], dtype=np.float32
        ),
        "sample_weight": sample_weight.astype(np.float32),
        "foot_error": np.asarray([row["foot_error"] for row in rows], dtype=np.float32),
        "balance_error": np.asarray(
            [row["balance_error"] for row in rows], dtype=np.float32
        ),
        "nominal_foot_relative_xy": nominal_foot_relative_xy.astype(np.float32),
        "nominal_foot_relative_yaw": np.array(0.0, dtype=np.float32),
        "base_height": np.array(base_height, dtype=np.float32),
        "joint_names": np.asarray(LEG_JOINT_NAMES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    params = RecoveryResetParams()
    data = generate(params, args.urdf, args.mesh_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(
        f"saved {len(data['joint_pos'])} recovery reset poses to {args.output}; "
        f"max foot error={1000.0 * float(np.max(data['foot_error'])):.3f}mm; "
        f"max balance error={1000.0 * float(np.max(data['balance_error'])):.3f}mm"
    )


if __name__ == "__main__":
    main()
