# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .oceanisaaclab_env_cfg import OceanisaaclabEnvCfg


class OceanisaaclabEnv(DirectRLEnv):
    cfg: OceanisaaclabEnvCfg

    def __init__(self, cfg: OceanisaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._leg_dof_idx = self._find_joint_ids(self.cfg.leg_joint_names)
        self._neck_dof_idx = self._find_joint_ids(self.cfg.neck_joint_names)
        self._base_body_id, _ = self.robot.find_bodies("base_link")
        # feet indices: keep articulation and contact-sensor orderings aligned (r5 then l5)
        self._feet_body_ids, _ = self.robot.find_bodies(["leg_r5_link", "leg_l5_link"], preserve_order=True)
        self._feet_contact_ids, _ = self.contact_sensor.find_sensors(
            ["leg_r5_link", "leg_l5_link"], preserve_order=True
        )

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        # action latency: ring buffer of past raw actions + per-env random delay
        self._action_buf_len = self.cfg.action_latency_steps + 1
        self._action_history = torch.zeros(
            self.num_envs, self._action_buf_len, self.cfg.action_space, device=self.device
        )
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._is_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._push_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._push_torques = torch.zeros_like(self._push_forces)
        self._push_time_left = torch.zeros(self.num_envs, device=self.device)
        self._push_interval_left = torch.zeros(self.num_envs, device=self.device)
        self._command_scale = torch.tensor(
            self.cfg.command_scale, dtype=torch.float, device=self.device
        )

        self._default_leg_joint_pos = self.robot.data.default_joint_pos.torch[:, self._leg_dof_idx].clone()
        self._soft_leg_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[:, self._leg_dof_idx].clone()
        self._default_neck_joint_pos = self.robot.data.default_joint_pos.torch[:, self._neck_dof_idx].clone()
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "terminated",
                "upright",
                "height",
                "ang_vel",
                "track_lin_vel",
                "lateral",
                "joint_pos",
                "joint_vel",
                "action_rate",
                "feet_air_time",
                "feet_slide",
                "stand_still",
            ]
        }

    def _find_joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        joint_ids = []
        for joint_name in joint_names:
            ids, names = self.robot.find_joints(joint_name)
            if len(ids) != 1:
                raise RuntimeError(f"Expected one joint named '{joint_name}', found {names}.")
            joint_ids.append(ids[0])
        return joint_ids

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # The URDF import nests the leg links under base_link, so IsaacLab's
        # activate_contact_sensors BFS stops at base_link and never tags the feet.
        # Manually add the contact-report API to the foot prims on the env_0 source
        # before cloning so all clones inherit it.
        self._activate_feet_contact_sensors("/World/envs/env_0/Robot")
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _activate_feet_contact_sensors(self, robot_prim_path: str) -> None:
        """Add the PhysX contact-report API to the foot rigid-body prims.

        ``activate_contact_sensors`` only tags the top-most rigid body (base_link) for
        this robot's nested link hierarchy, so the feet are tagged explicitly here.
        """
        from pxr import UsdPhysics

        stage = sim_utils.get_current_stage()
        foot_names = ("leg_r5_link", "leg_l5_link")
        tagged = 0
        for prim in stage.Traverse():
            if not prim.GetPath().pathString.startswith(robot_prim_path):
                continue
            if prim.GetName() not in foot_names:
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            applied = prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied:
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
            if "PhysxContactReportAPI" not in applied:
                prim.AddAppliedSchema("PhysxContactReportAPI")
            tagged += 1
        if tagged != len(foot_names):
            raise RuntimeError(
                f"Expected to tag {len(foot_names)} foot prims for contact sensing under "
                f"'{robot_prim_path}', tagged {tagged}. Check foot link names."
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        self._actions = actions.clamp(-1.0, 1.0)
        # apply randomized action latency: shift history and pick a delayed action per env
        if self.cfg.enable_action_latency:
            self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
            self._action_history[:, 0] = self._actions
            delayed_actions = torch.gather(
                self._action_history,
                1,
                self._action_delay.view(-1, 1, 1).expand(-1, 1, self.cfg.action_space),
            ).squeeze(1)
        else:
            delayed_actions = self._actions
        desired_joint_pos = self._default_leg_joint_pos + self.cfg.action_scale * delayed_actions
        self._processed_actions = torch.clamp(
            desired_joint_pos,
            self._soft_leg_joint_pos_limits[:, :, 0],
            self._soft_leg_joint_pos_limits[:, :, 1],
        )
        self._update_random_pushes()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._leg_dof_idx)
        self.robot.set_joint_position_target_index(target=self._default_neck_joint_pos, joint_ids=self._neck_dof_idx)

    def _update_random_pushes(self) -> None:
        if not self.cfg.enable_random_push:
            return

        self._push_time_left = torch.clamp(self._push_time_left - self.step_dt, min=0.0)
        self._push_interval_left = torch.clamp(self._push_interval_left - self.step_dt, min=0.0)
        start_push = (self._push_interval_left <= 0.0) & (self._push_time_left <= 0.0)
        if torch.any(start_push):
            env_ids = torch.nonzero(start_push, as_tuple=False).squeeze(-1)
            angles = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi
            magnitudes = sample_uniform(
                self.cfg.push_force_range[0],
                self.cfg.push_force_range[1],
                (len(env_ids),),
                self.device,
            )
            self._push_forces[env_ids, 0, 0] = magnitudes * torch.cos(angles)
            self._push_forces[env_ids, 0, 1] = magnitudes * torch.sin(angles)
            self._push_forces[env_ids, 0, 2] = 0.0
            self._push_time_left[env_ids] = self.cfg.push_duration_s
            self._push_interval_left[env_ids] = sample_uniform(
                self.cfg.push_interval_s[0],
                self.cfg.push_interval_s[1],
                (len(env_ids),),
                self.device,
            )

        inactive = self._push_time_left <= 0.0
        self._push_forces[inactive] = 0.0
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            self._push_forces,
            self._push_torques,
            body_ids=self._base_body_id,
            is_global=True,
        )

    def _get_observations(self) -> dict:
        ang_vel = self.robot.data.root_ang_vel_b.torch
        proj_g = self.robot.data.projected_gravity_b.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx] - self._default_leg_joint_pos
        joint_vel = self.robot.data.joint_vel.torch[:, self._leg_dof_idx]

        if self.cfg.enable_obs_noise:
            ang_vel = ang_vel + torch.randn_like(ang_vel) * self.cfg.noise_ang_vel
            proj_g = proj_g + torch.randn_like(proj_g) * self.cfg.noise_proj_g
            joint_pos = joint_pos + torch.randn_like(joint_pos) * self.cfg.noise_joint_pos
            joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.noise_joint_vel

        obs = torch.cat(
            (
                ang_vel * self.cfg.ang_vel_scale,
                proj_g,
                self._commands * self._command_scale,
                joint_pos * self.cfg.dof_pos_scale,
                joint_vel * self.cfg.dof_vel_scale,
                self._actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # contact / air-time data for the two feet (contact-sensor index space)
        air_time = self.contact_sensor.data.last_air_time.torch[:, self._feet_contact_ids]
        contact_time = self.contact_sensor.data.current_contact_time.torch[:, self._feet_contact_ids]
        first_contact = self.contact_sensor.compute_first_contact(self.step_dt).torch[:, self._feet_contact_ids]
        in_contact = contact_time > 0.0
        # foot horizontal velocity (articulation index space, aligned order)
        feet_lin_vel = self.robot.data.body_lin_vel_w.torch[:, self._feet_body_ids, :2]

        total_reward, reward_terms = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_track_lin_vel,
            self.cfg.rew_scale_lateral,
            self.cfg.rew_scale_joint_pos,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_feet_air_time,
            self.cfg.rew_scale_feet_slide,
            self.cfg.rew_scale_stand_still,
            self.cfg.target_base_height,
            self.cfg.air_time_target,
            self.cfg.lin_vel_track_sigma,
            self.robot.data.root_pos_w.torch[:, 2],
            self._commands,
            self._is_standing,
            self.robot.data.root_lin_vel_b.torch,
            self.robot.data.projected_gravity_b.torch,
            self.robot.data.root_ang_vel_b.torch,
            self.robot.data.joint_pos.torch[:, self._leg_dof_idx] - self._default_leg_joint_pos,
            self.robot.data.joint_vel.torch[:, self._leg_dof_idx],
            self._actions,
            self._previous_actions,
            air_time,
            first_contact,
            in_contact,
            feet_lin_vel,
            self.reset_terminated,
        )
        for key, value in reward_terms.items():
            self._episode_sums[key] += value
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_too_low = self.robot.data.root_pos_w.torch[:, 2] < self.cfg.min_base_height
        not_upright = -self.robot.data.projected_gravity_b.torch[:, 2] < self.cfg.min_upright_projection
        joint_pos = self.robot.data.joint_pos.torch[:, self._leg_dof_idx]
        lower_limit = self.robot.data.soft_joint_pos_limits.torch[:, self._leg_dof_idx, 0]
        upper_limit = self.robot.data.soft_joint_pos_limits.torch[:, self._leg_dof_idx, 1]
        joint_out_of_bounds = torch.any((joint_pos < lower_limit) | (joint_pos > upper_limit), dim=1)
        return base_too_low | not_upright | joint_out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._default_leg_joint_pos[env_ids]
        self._action_history[env_ids] = 0.0
        if self.cfg.enable_action_latency:
            self._action_delay[env_ids] = torch.randint(
                0, self.cfg.action_latency_steps + 1, (len(env_ids),), device=self.device
            )
        # sample forward velocity command (vx only); a fraction of envs stand still
        self._commands[env_ids] = 0.0
        self._commands[env_ids, 0] = sample_uniform(
            self.cfg.command_vx_range[0],
            self.cfg.command_vx_range[1],
            (len(env_ids),),
            self.device,
        )
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.stand_still_prob
        self._commands[env_ids[standing], 0] = 0.0
        self._is_standing[env_ids] = standing
        self._push_forces[env_ids] = 0.0
        self._push_torques[env_ids] = 0.0
        self._push_time_left[env_ids] = 0.0
        self._push_interval_left[env_ids] = sample_uniform(
            self.cfg.push_interval_s[0],
            self.cfg.push_interval_s[1],
            (len(env_ids),),
            self.device,
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._leg_dof_idx] += sample_uniform(
            -self.cfg.reset_joint_pos_noise,
            self.cfg.reset_joint_pos_noise,
            joint_pos[:, self._leg_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._leg_dof_idx] = torch.clamp(
            joint_pos[:, self._leg_dof_idx],
            self.robot.data.soft_joint_pos_limits.torch[env_ids][:, self._leg_dof_idx, 0],
            self.robot.data.soft_joint_pos_limits.torch[env_ids][:, self._leg_dof_idx, 1],
        )
        joint_pos[:, self._neck_dof_idx] = 0.0
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/fall"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras


def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_height: float,
    rew_scale_ang_vel: float,
    rew_scale_track_lin_vel: float,
    rew_scale_lateral: float,
    rew_scale_joint_pos: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_feet_air_time: float,
    rew_scale_feet_slide: float,
    rew_scale_stand_still: float,
    target_base_height: float,
    air_time_target: float,
    lin_vel_track_sigma: float,
    base_height: torch.Tensor,
    commands: torch.Tensor,
    is_standing: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    joint_pos_error: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    air_time: torch.Tensor,
    first_contact: torch.Tensor,
    in_contact: torch.Tensor,
    feet_lin_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    is_walking = (~is_standing).float()

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_upright = rew_scale_upright * torch.exp(-4.0 * torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1))
    rew_height = rew_scale_height * torch.exp(-20.0 * torch.square(base_height - target_base_height))
    # only penalize roll/pitch angular velocity (keep body steady, do not fight yaw)
    rew_ang_vel = rew_scale_ang_vel * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    # track vx command (exp shaping, positive reward — core walking driver)
    lin_vel_error = torch.square(commands[:, 0] - root_lin_vel_b[:, 0])
    rew_track_lin_vel = rew_scale_track_lin_vel * torch.exp(-lin_vel_error / lin_vel_track_sigma)
    # suppress non-commanded lateral motion (vy) and yaw rotation
    rew_lateral = rew_scale_lateral * (
        torch.square(root_lin_vel_b[:, 1]) + torch.square(root_ang_vel_b[:, 2])
    )
    rew_joint_pos = rew_scale_joint_pos * torch.sum(torch.square(joint_pos_error), dim=1)
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=1)
    rew_action_rate = rew_scale_action_rate * torch.sum(torch.square(actions - previous_actions), dim=1)
    # feet air time: reward time off the ground at each touchdown (walking envs only)
    air_time_reward = torch.sum((air_time - air_time_target) * first_contact.float(), dim=1)
    rew_feet_air_time = rew_scale_feet_air_time * air_time_reward * is_walking
    # penalize horizontal foot slip while in contact
    slide = torch.sum(torch.sum(torch.square(feet_lin_vel), dim=2) * in_contact.float(), dim=1)
    rew_feet_slide = rew_scale_feet_slide * slide
    # standing envs: penalize any foot motion to hold still
    foot_speed_sq = torch.sum(torch.sum(torch.square(feet_lin_vel), dim=2), dim=1)
    rew_stand_still = rew_scale_stand_still * foot_speed_sq * is_standing.float()
    total_reward = (
        rew_alive
        + rew_termination
        + rew_upright
        + rew_height
        + rew_ang_vel
        + rew_track_lin_vel
        + rew_lateral
        + rew_joint_pos
        + rew_joint_vel
        + rew_action_rate
        + rew_feet_air_time
        + rew_feet_slide
        + rew_stand_still
    )
    reward_terms = {
        "alive": rew_alive,
        "terminated": rew_termination,
        "upright": rew_upright,
        "height": rew_height,
        "ang_vel": rew_ang_vel,
        "track_lin_vel": rew_track_lin_vel,
        "lateral": rew_lateral,
        "joint_pos": rew_joint_pos,
        "joint_vel": rew_joint_vel,
        "action_rate": rew_action_rate,
        "feet_air_time": rew_feet_air_time,
        "feet_slide": rew_feet_slide,
        "stand_still": rew_stand_still,
    }
    return total_reward, reward_terms