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
        self._gait_phase_offset = torch.zeros(self.num_envs, device=self.device)
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
                "track_ang_vel",
                "progress",
                "lateral",
                "joint_pos",
                "joint_vel",
                "action_rate",
                "feet_air_time",
                "feet_slide",
                "stand_still",
                "phase_contact",
                "swing_contact",
                "single_support",
                "no_fly",
                "feet_clearance",
                "contact_force_rate",
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

    def _current_push_force_range(self) -> tuple[float, float]:
        """Push magnitude range, linearly ramped over training (curriculum).

        Staged: the ramp does not start until push_curriculum_start_step common steps, so
        the push stays at the gentle floor (cfg.push_force_range) while the policy learns
        to walk first; it then ramps to cfg.push_force_range_max over push_curriculum_steps.
        On resume the counter restarts from 0, so the ramp re-runs from the start range —
        intended, so the recovered policy re-eases into larger pushes rather than being
        slammed.
        """
        lo, hi = self.cfg.push_force_range
        if not self.cfg.enable_push_curriculum:
            return lo, hi
        hi_lo, hi_hi = self.cfg.push_force_range_max
        frac = (self.common_step_counter - self.cfg.push_curriculum_start_step) / float(
            self.cfg.push_curriculum_steps
        )
        frac = max(0.0, min(1.0, frac))
        return lo + (hi_lo - lo) * frac, hi + (hi_hi - hi) * frac

    def _current_command_vx_range(self) -> tuple[float, float]:
        """vx command range, linearly widened over training (curriculum).

        Early training samples a narrow mid-speed band (command_vx_range_start) so the
        gait can bootstrap from a clear "keep walking" signal, then linearly opens to the
        full command_vx_range. Uses the same common_step_counter mechanism as the push
        curriculum; on resume the counter restarts from 0 so the ramp re-runs.
        """
        lo, hi = self.cfg.command_vx_range
        if not self.cfg.enable_command_curriculum:
            return lo, hi
        lo0, hi0 = self.cfg.command_vx_range_start
        frac = min(1.0, self.common_step_counter / float(self.cfg.command_curriculum_steps))
        return lo0 + (lo - lo0) * frac, hi0 + (hi - hi0) * frac

    def _update_random_pushes(self) -> None:
        if not self.cfg.enable_random_push:
            return

        self._push_time_left = torch.clamp(self._push_time_left - self.step_dt, min=0.0)
        self._push_interval_left = torch.clamp(self._push_interval_left - self.step_dt, min=0.0)
        start_push = (self._push_interval_left <= 0.0) & (self._push_time_left <= 0.0)
        if torch.any(start_push):
            env_ids = torch.nonzero(start_push, as_tuple=False).squeeze(-1)
            angles = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi
            lo, hi = self._current_push_force_range()
            magnitudes = sample_uniform(lo, hi, (len(env_ids),), self.device)
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
        phase = self._gait_phase()
        gait_clock = torch.stack((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=1)

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
                gait_clock,
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
        # foot-sole clearance: leg_[lr]5_link origin sits ~foot_origin_offset above the
        # ground when the sole is planted, so subtract it to get the true sole-to-ground
        # gap. Without this the old feet_clearance term was measuring link-origin height
        # (~0.067m at rest > 0.05 target) and its clamp was pinned at 0 the entire run.
        feet_height = (
            self.robot.data.body_pos_w.torch[:, self._feet_body_ids, 2]
            - self.scene.env_origins[:, 2].unsqueeze(1)
            - self.cfg.foot_origin_offset
        )
        # contact-force rate: norm of frame-to-frame change in foot contact force.
        # history shape (N, T, num_sensors, 3); penalizing this directly damps the
        # rapid force jumps that cause sim2real standing jitter/oscillation.
        force_hist = self.contact_sensor.data.net_forces_w_history.torch[:, :, self._feet_contact_ids, :]
        force_diff = force_hist[:, :-1] - force_hist[:, 1:]
        contact_force_rate = torch.sum(torch.norm(force_diff, dim=-1), dim=(1, 2))

        # instability gate (IMU-observable only): torso tilt |proj_g_xy| + tilt-rate |ang_vel_xy|.
        # 0 in steady stance (stepping penalized → planted feet), ->1 when pushed off balance
        # (stepping penalty lifted → free to step and recover). No privileged push signal.
        proj_g = self.robot.data.projected_gravity_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        tilt = torch.norm(proj_g[:, :2], dim=1)
        tilt_rate = torch.norm(ang_vel[:, :2], dim=1)
        gate_arg = (
            (tilt - self.cfg.instability_tilt_thresh) / self.cfg.instability_tilt_thresh
            + (tilt_rate - self.cfg.instability_tilt_rate_thresh) / self.cfg.instability_tilt_rate_thresh
        )
        instability = torch.sigmoid(self.cfg.instability_sharpness * gate_arg)

        total_reward, reward_terms = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_track_lin_vel,
            self.cfg.rew_scale_track_ang_vel,
            self.cfg.rew_scale_progress,
            self.cfg.rew_scale_lateral,
            self.cfg.rew_scale_joint_pos,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_feet_air_time,
            self.cfg.rew_scale_feet_slide,
            self.cfg.rew_scale_stand_still,
            self.cfg.rew_scale_phase_contact,
            self.cfg.rew_scale_swing_contact,
            self.cfg.rew_scale_single_support,
            self.cfg.rew_scale_no_fly,
            self.cfg.rew_scale_feet_clearance,
            self.cfg.rew_scale_contact_force_rate,
            self.cfg.target_base_height,
            self.cfg.air_time_target,
            self.cfg.foot_clearance_target,
            self.cfg.move_command_threshold,
            self.cfg.lin_vel_track_sigma,
            self.cfg.ang_vel_track_sigma,
            self.cfg.gait_duty_factor,
            self.cfg.forward_vx_sign,
            self.robot.data.root_pos_w.torch[:, 2],
            self._commands,
            self._gait_phase(),
            instability,
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
            feet_height,
            contact_force_rate,
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
        # Sample low-speed locomotion commands. A fraction remains zero-command
        # stance; another fraction learns turn-in-place so yaw joystick inputs
        # are inside the training distribution.
        self._commands[env_ids] = 0.0
        vx_lo, vx_hi = self._current_command_vx_range()
        self._commands[env_ids, 0] = sample_uniform(
            vx_lo,
            vx_hi,
            (len(env_ids),),
            self.device,
        )
        non_standing = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.stand_still_prob
        self._commands[env_ids[standing], 0] = 0.0
        non_standing[standing] = False
        turn_in_place = (torch.rand(len(env_ids), device=self.device) < self.cfg.turn_in_place_prob) & non_standing
        self._commands[env_ids[turn_in_place], 0] = 0.0
        moving_or_turning = ~standing
        self._commands[env_ids[moving_or_turning], 2] = sample_uniform(
            self.cfg.command_wz_range[0],
            self.cfg.command_wz_range[1],
            (int(torch.count_nonzero(moving_or_turning).item()),),
            self.device,
        )
        self._is_standing[env_ids] = standing
        self._gait_phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device)
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
        # Reference-state init: give ~half of the vx-commanded envs a starting forward
        # velocity ≈ command (default yaw≈0 so world-x ≈ body-x). Bootstrapping the first
        # step from a standstill is the hardest part of learning to walk; starting some
        # envs already moving removes that barrier and weakens the standing attractor.
        moving = non_standing & ~turn_in_place
        seeded = moving & (torch.rand(len(env_ids), device=self.device) < 0.5)
        vx_seed = self._commands[env_ids, 0] + sample_uniform(
            -0.05, 0.05, (len(env_ids),), self.device
        )
        # head-forward = forward_vx_sign * body-x; at reset yaw≈0 so world-x ≈ body-x
        default_root_vel[seeded, 0] = self.cfg.forward_vx_sign * vx_seed[seeded]

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

    def _gait_phase(self) -> torch.Tensor:
        gait_steps = self.cfg.gait_cycle_period / self.step_dt
        return torch.remainder(self.episode_length_buf.float() / gait_steps + self._gait_phase_offset, 1.0)


def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_height: float,
    rew_scale_ang_vel: float,
    rew_scale_track_lin_vel: float,
    rew_scale_track_ang_vel: float,
    rew_scale_progress: float,
    rew_scale_lateral: float,
    rew_scale_joint_pos: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_feet_air_time: float,
    rew_scale_feet_slide: float,
    rew_scale_stand_still: float,
    rew_scale_phase_contact: float,
    rew_scale_swing_contact: float,
    rew_scale_single_support: float,
    rew_scale_no_fly: float,
    rew_scale_feet_clearance: float,
    rew_scale_contact_force_rate: float,
    target_base_height: float,
    air_time_target: float,
    foot_clearance_target: float,
    move_command_threshold: float,
    lin_vel_track_sigma: float,
    ang_vel_track_sigma: float,
    gait_duty_factor: float,
    forward_vx_sign: float,
    base_height: torch.Tensor,
    commands: torch.Tensor,
    gait_phase: torch.Tensor,
    instability: torch.Tensor,
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
    feet_height: torch.Tensor,
    contact_force_rate: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # instability in [0,1]: 0 steady stance, 1 pushed off balance.
    # Stand-still penalties only apply to stable zero-command samples. Walking and
    # recovery samples must be free to move the feet, otherwise the optimal policy
    # becomes planted-foot leaning or tiny shuffling.
    stability = 1.0 - instability
    command_norm = torch.maximum(torch.abs(commands[:, 0]), torch.abs(commands[:, 2]))
    moving_command = (command_norm > move_command_threshold).float()
    stand_command = 1.0 - moving_command
    stepping_gate = torch.maximum(moving_command, instability)
    stable_stand_gate = stand_command * stability

    # Forward-progress gate. For a forward (vx) command, gait-shaping rewards are scaled by
    # how much of the commanded forward speed is actually realized: fwd_frac in [0,1] is 0 for
    # a robot marching in place (vx≈0) and 1 at/above commanded speed. This kills the
    # "march in place" local optimum — stepping only pays if it produces forward motion.
    # Non-forward commands (turn-in-place / stance) keep gate=1 so their turning/gait rewards
    # are unaffected; and instability lifts the gate so push-recovery stepping is never gated.
    vx_cmd = commands[:, 0]
    # Head-forward speed. URDF base_link +x points to the robot's tail (head faces -x),
    # so measured body-x velocity must be sign-corrected before comparing against the
    # forward command. Without this the policy learns to walk backwards relative to the
    # head (the 07-01 "W walks backwards / S falls forward" bug).
    vx_act = forward_vx_sign * root_lin_vel_b[:, 0]
    forward_command = (vx_cmd > move_command_threshold).float()
    fwd_frac = torch.clamp(vx_act / torch.clamp(vx_cmd, min=1e-3), 0.0, 1.0)
    fwd_gate = torch.where(forward_command > 0.5, fwd_frac, torch.ones_like(fwd_frac))
    progress_pay = torch.maximum(fwd_gate, instability)

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_upright = rew_scale_upright * torch.exp(-4.0 * torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1))
    rew_height = rew_scale_height * torch.exp(-20.0 * torch.square(base_height - target_base_height))
    # only penalize roll/pitch angular velocity (keep body steady, do not fight yaw)
    rew_ang_vel = rew_scale_ang_vel * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    # Track vx/yaw commands. The sim2sim arrow keys map naturally to these
    # command slots, so both must be trained and rewarded.
    lin_vel_error = torch.square(commands[:, 0] - vx_act)
    rew_track_lin_vel = rew_scale_track_lin_vel * torch.exp(-lin_vel_error / lin_vel_track_sigma)
    ang_vel_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_track_ang_vel = rew_scale_track_ang_vel * torch.exp(-ang_vel_error / ang_vel_track_sigma)
    # Linear forward-progress reward. The exp tracking term is nearly flat near vx=0, giving a
    # weak gradient to *start* translating; this linear term pays proportionally to realized
    # forward speed (capped at the command) so there is a constant push to move forward. Zero
    # for stance and turn-in-place, so it cannot be farmed by marching in place.
    rew_progress = rew_scale_progress * fwd_frac * forward_command
    # suppress non-commanded lateral motion (vy). Yaw is handled by tracking.
    rew_lateral = rew_scale_lateral * torch.square(root_lin_vel_b[:, 1])
    rew_joint_pos = rew_scale_joint_pos * torch.sum(torch.square(joint_pos_error), dim=1)
    # Smoothness remains strict for stable standing but is relaxed further while walking.
    motion_smooth_weight = 0.2 + 0.8 * stable_stand_gate
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=1) * motion_smooth_weight
    rew_action_rate = (
        rew_scale_action_rate * torch.sum(torch.square(actions - previous_actions), dim=1) * motion_smooth_weight
    )
    # Feet air time: reward genuine swing steps on landing. clamp(min=0) removes the
    # old negative-reward deadlock where short steps (air_time < target) landed at a
    # penalty, making "step short" worse than "never step". Now stepping only ever
    # adds reward; not stepping simply earns nothing here.
    air_time_reward = torch.sum(torch.clamp(air_time - air_time_target, min=0.0) * first_contact.float(), dim=1)
    rew_feet_air_time = rew_scale_feet_air_time * air_time_reward * stepping_gate * progress_pay
    # Penalize horizontal foot slip while in contact. Keep the penalty active for
    # walking so commanded motion is achieved by lifting/swinging feet, not sliding.
    slide = torch.sum(torch.sum(torch.square(feet_lin_vel), dim=2) * in_contact.float(), dim=1)
    rew_feet_slide = rew_scale_feet_slide * slide * (stand_command * stability + moving_command)
    # Penalize any foot motion only for stable zero-command stance.
    foot_speed_sq = torch.sum(torch.sum(torch.square(feet_lin_vel), dim=2), dim=1)
    rew_stand_still = rew_scale_stand_still * foot_speed_sq * stable_stand_gate
    # Walking contact structure via the gait clock, split into two asymmetric terms so
    # standing on both feet can no longer passively collect the old soft-L1 phase reward:
    #   - support phase (desired contact): reward actual contact
    #   - swing phase (desired lift): hard-penalize staying in contact -> forces lift-off
    contact_count = torch.sum(in_contact.float(), dim=1)
    single_support = contact_count == 1.0
    no_contact = contact_count == 0.0
    swing_mask = (~in_contact).float()
    left_phase = torch.remainder(gait_phase + 0.5, 1.0)
    desired_contact = torch.stack(
        (gait_phase < gait_duty_factor, left_phase < gait_duty_factor),
        dim=1,
    ).float()
    desired_swing = 1.0 - desired_contact
    support_match = torch.sum(desired_contact * in_contact.float(), dim=1)
    rew_phase_contact = rew_scale_phase_contact * support_match * moving_command * progress_pay
    swing_violation = torch.sum(desired_swing * in_contact.float(), dim=1)
    rew_swing_contact = rew_scale_swing_contact * swing_violation * moving_command
    rew_single_support = rew_scale_single_support * single_support.float() * moving_command * progress_pay
    rew_no_fly = rew_scale_no_fly * no_contact.float()
    # Foot clearance: reward swing feet whose sole gap is near the target height (positive
    # shaping), rather than a pure penalty that would tempt the policy to just not lift.
    clearance_err = torch.square(feet_height - foot_clearance_target)
    rew_feet_clearance = (
        rew_scale_feet_clearance
        * torch.sum(torch.exp(-clearance_err / 0.0025) * swing_mask, dim=1)
        * moving_command
        * progress_pay
    )
    # penalize rapid contact-force changes (anti-jitter: damps force spikes that cause
    # sim2real standing oscillation)
    contact_smooth_weight = 0.25 + 0.75 * stable_stand_gate
    rew_contact_force_rate = rew_scale_contact_force_rate * contact_force_rate * contact_smooth_weight
    total_reward = (
        rew_alive
        + rew_termination
        + rew_upright
        + rew_height
        + rew_ang_vel
        + rew_track_lin_vel
        + rew_track_ang_vel
        + rew_progress
        + rew_lateral
        + rew_joint_pos
        + rew_joint_vel
        + rew_action_rate
        + rew_feet_air_time
        + rew_feet_slide
        + rew_stand_still
        + rew_phase_contact
        + rew_swing_contact
        + rew_single_support
        + rew_no_fly
        + rew_feet_clearance
        + rew_contact_force_rate
    )
    reward_terms = {
        "alive": rew_alive,
        "terminated": rew_termination,
        "upright": rew_upright,
        "height": rew_height,
        "ang_vel": rew_ang_vel,
        "track_lin_vel": rew_track_lin_vel,
        "track_ang_vel": rew_track_ang_vel,
        "progress": rew_progress,
        "lateral": rew_lateral,
        "joint_pos": rew_joint_pos,
        "joint_vel": rew_joint_vel,
        "action_rate": rew_action_rate,
        "feet_air_time": rew_feet_air_time,
        "feet_slide": rew_feet_slide,
        "stand_still": rew_stand_still,
        "phase_contact": rew_phase_contact,
        "swing_contact": rew_swing_contact,
        "single_support": rew_single_support,
        "no_fly": rew_no_fly,
        "feet_clearance": rew_feet_clearance,
        "contact_force_rate": rew_contact_force_rate,
    }
    return total_reward, reward_terms
