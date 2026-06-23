from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointConfig:
    name: str
    motor_id: int
    direction: float
    zero_offset: float
    lower: float
    upper: float
    kp: float
    kd: float


@dataclass
class RobotState:
    quat: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray


class UnitreeM8010Interface:
    joint_order = [
        'leg_r1_joint',
        'leg_r2_joint',
        'leg_r3_joint',
        'leg_r4_joint',
        'leg_r5_joint',
        'leg_l1_joint',
        'leg_l2_joint',
        'leg_l3_joint',
        'leg_l4_joint',
        'leg_l5_joint',
    ]

    def __init__(self, config: dict, enable_motors: bool = False):
        self.config = config
        self.enable_motors = enable_motors
        self.joints = self._load_joint_configs(config)
        self._joint_by_name = {joint.name: joint for joint in self.joints}
        self._last_command = np.zeros(len(self.joint_order), dtype=np.float32)

    def connect(self) -> None:
        if self.enable_motors:
            raise NotImplementedError(
                'Fill in the Unitree SDK transport before enabling motors.'
            )

    def close(self) -> None:
        pass

    def read_state(self) -> RobotState:
        return RobotState(
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
            joint_pos=np.zeros(len(self.joint_order), dtype=np.float32),
            joint_vel=np.zeros(len(self.joint_order), dtype=np.float32),
        )

    def send_position_targets(self, joint_targets: np.ndarray) -> None:
        clipped = self.clip_to_limits(joint_targets)
        if self.enable_motors:
            raise NotImplementedError(
                'Map clipped targets to Unitree SDK commands here.'
            )
        self._last_command = clipped

    def clip_to_limits(self, joint_targets: np.ndarray) -> np.ndarray:
        clipped = np.asarray(joint_targets, dtype=np.float32).copy()
        for index, joint_name in enumerate(self.joint_order):
            joint = self._joint_by_name[joint_name]
            clipped[index] = np.clip(clipped[index], joint.lower, joint.upper)
        return clipped

    @staticmethod
    def _load_joint_configs(config: dict) -> list[JointConfig]:
        joints = []
        for bus_cfg in config['buses'].values():
            for joint_cfg in bus_cfg['joints']:
                joints.append(JointConfig(**joint_cfg))
        names = {joint.name for joint in joints}
        missing = set(UnitreeM8010Interface.joint_order) - names
        if missing:
            raise ValueError(f'Missing joint configs: {sorted(missing)}')
        return joints