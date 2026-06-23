from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from unitree_m8010_interface import UnitreeM8010Interface


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('Install PyYAML to read real robot configs.') from exc
    with path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def quat_to_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation.T @ gravity


def build_observation(
    state,
    previous_action: np.ndarray,
    command: np.ndarray,
    policy_cfg: dict,
) -> torch.Tensor:
    projected_gravity = quat_to_projected_gravity(state.quat)
    command_scale = np.array(policy_cfg['commands_scale'], dtype=np.float32)
    obs = np.concatenate(
        [
            state.gyro * float(policy_cfg['ang_vel_scale']),
            projected_gravity,
            command * command_scale,
            state.joint_pos * float(policy_cfg['dof_pos_scale']),
            state.joint_vel * float(policy_cfg['dof_vel_scale']),
            previous_action,
        ]
    ).astype(np.float32)
    return torch.from_numpy(obs).unsqueeze(0)


def check_tilt(projected_gravity: np.ndarray, max_roll_pitch_rad: float) -> bool:
    min_upright = math.cos(max_roll_pitch_rad)
    return -projected_gravity[2] >= min_upright


def rate_limit_target(
    desired: np.ndarray,
    previous: np.ndarray,
    max_rate: float,
    dt: float,
) -> np.ndarray:
    max_delta = max_rate * dt
    return previous + np.clip(desired - previous, -max_delta, max_delta)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run exported BDX stand policy.')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('bdx_stand_config.yaml'))
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--enable-motors', action='store_true')
    args = parser.parse_args()

    config = load_yaml(args.config)
    policy_path = args.checkpoint or Path(config['policy']['path'])
    control_hz = float(config['policy']['control_hz'])
    inference_hz = float(config['policy']['inference_hz'])
    action_scale = float(config['policy']['action_scale'])
    max_target_rate = float(config['safety']['action_rate_limit_rad_per_s'])
    inference_stride = max(1, round(control_hz / inference_hz))

    interface = UnitreeM8010Interface(config, enable_motors=args.enable_motors)
    interface.connect()

    if policy_path.exists():
        policy = torch.jit.load(str(policy_path), map_location='cpu')
        policy.eval()
    else:
        policy = None
        print(f'[WARN] Policy not found at {policy_path}; running zero-action mock.')

    previous_action = np.zeros(10, dtype=np.float32)
    current_target = np.zeros(10, dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)
    period = 1.0 / control_hz
    step = 0

    try:
        while True:
            start = time.monotonic()
            state = interface.read_state()
            projected_gravity = quat_to_projected_gravity(state.quat)
            if not check_tilt(projected_gravity, config['safety']['max_roll_pitch_rad']):
                raise RuntimeError('Tilt limit exceeded; stopping commands.')

            if step % inference_stride == 0:
                obs = build_observation(
                    state, previous_action, command, config['policy']
                )
                if policy is None:
                    action = np.zeros(10, dtype=np.float32)
                else:
                    with torch.inference_mode():
                        action = policy(obs).squeeze(0).numpy().astype(np.float32)
                action = np.clip(action, -1.0, 1.0)
                previous_action = action
                desired_target = action_scale * action
                current_target = rate_limit_target(
                    desired_target,
                    current_target,
                    max_target_rate,
                    period * inference_stride,
                )

            interface.send_position_targets(current_target)
            step += 1
            elapsed = time.monotonic() - start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        interface.close()


if __name__ == '__main__':
    main()