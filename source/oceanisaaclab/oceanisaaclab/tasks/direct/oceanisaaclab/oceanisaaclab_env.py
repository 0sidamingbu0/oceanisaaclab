# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import warp as wp
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .nested_contact_sensor import NestedBodyContactSensor
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
        # per-env domain randomization (mass / friction / PD gains), sampled once at startup
        if self.cfg.enable_domain_rand:
            self._apply_domain_randomization()
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

    def _apply_domain_randomization(self) -> None:
        """Per-env physical-parameter randomization, sampled once at startup.

        sim2sim showed the policy falling on the first step despite fall=0.23 in
        training — the classic symptom of missing domain randomization (only obs
        noise + action latency were randomized before). Each env gets a fixed
        random mass / friction / PD-gain draw; with thousands of envs the policy
        sees the whole distribution every rollout.
        """
        from isaaclab.actuators import ImplicitActuator

        all_env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        # 1) body masses: scale every link mass by a single per-env factor
        #    ((num_envs, 1) scale broadcasts over all bodies of that env).
        lo, hi = self.cfg.dr_mass_scale_range
        mass_attr = "body_mass" if hasattr(self.robot.data, "body_mass") else "default_mass"
        default_mass = getattr(self.robot.data, mass_attr).torch.clone()
        mass_scale = sample_uniform(lo, hi, (self.num_envs, 1), self.device)
        # 存下质量缩放供非对称 critic 的特权观测使用（路线 B）
        self._dr_mass_scale = mass_scale.squeeze(1).clone()
        body_ids = torch.arange(self.robot.num_bodies, dtype=torch.int32, device=self.device)
        self.robot.set_masses_index(
            masses=(default_mass * mass_scale).contiguous(),
            body_ids=body_ids,
            env_ids=all_env_ids,
        )
        # 2) PD gains: independent per-env/per-joint stiffness and damping factors.
        lo, hi = self.cfg.dr_pd_gain_scale_range
        for actuator in self.robot.actuators.values():
            joint_indices = actuator.joint_indices
            if isinstance(joint_indices, slice):
                joint_indices = torch.arange(self.robot.num_joints, device=self.device)[joint_indices]
            stiffness = actuator.stiffness * sample_uniform(lo, hi, actuator.stiffness.shape, self.device)
            damping = actuator.damping * sample_uniform(lo, hi, actuator.damping.shape, self.device)
            actuator.stiffness[:] = stiffness
            actuator.damping[:] = damping
            if isinstance(actuator, ImplicitActuator):
                self.robot.write_joint_stiffness_to_sim_index(
                    stiffness=stiffness,
                    joint_ids=joint_indices.to(dtype=torch.int32),
                    env_ids=all_env_ids,
                )
                self.robot.write_joint_damping_to_sim_index(
                    damping=damping,
                    joint_ids=joint_indices.to(dtype=torch.int32),
                    env_ids=all_env_ids,
                )
        # 3) contact friction: scale static/dynamic friction of every collision shape
        #    by a per-env factor (requires the PhysX view; skipped with a warning on
        #    backends that do not expose it).
        physx_view = getattr(self.robot, "root_physx_view", None)
        if physx_view is not None:
            lo, hi = self.cfg.dr_friction_scale_range
            # get_material_properties() returns a warp array (host/CPU buffer, shape
            # (num_envs, num_shapes, 3): static friction, dynamic friction, restitution).
            # Wrap with wp.to_torch to edit as a torch tensor, scale the two friction
            # columns per-env, then hand back warp arrays (CPU int32 env ids) as the
            # PhysX tensor API requires.
            materials = wp.to_torch(physx_view.get_material_properties())
            friction_scale = sample_uniform(lo, hi, (materials.shape[0], 1, 1), materials.device)
            # 存下摩擦缩放供非对称 critic 的特权观测使用（路线 B）
            self._dr_friction_scale = friction_scale.view(-1).to(self.device).clone()
            materials[..., :2] = materials[..., :2] * friction_scale
            env_ids_cpu = torch.arange(self.num_envs, dtype=torch.int32)
            physx_view.set_material_properties(
                wp.from_torch(materials.contiguous(), dtype=wp.float32),
                wp.from_torch(env_ids_cpu, dtype=wp.int32),
            )
        else:
            import omni.log

            omni.log.warn("domain rand: robot has no root_physx_view; skipping friction randomization.")

    def _feet_current_contact_time(self) -> torch.Tensor:
        return torch.cat(
            (
                self.right_foot_contact_sensor.data.current_contact_time.torch,
                self.left_foot_contact_sensor.data.current_contact_time.torch,
            ),
            dim=1,
        )

    def _feet_last_air_time(self) -> torch.Tensor:
        return torch.cat(
            (
                self.right_foot_contact_sensor.data.last_air_time.torch,
                self.left_foot_contact_sensor.data.last_air_time.torch,
            ),
            dim=1,
        )

    def _feet_first_contact(self) -> torch.Tensor:
        return torch.cat(
            (
                self.right_foot_contact_sensor.compute_first_contact(self.step_dt).torch,
                self.left_foot_contact_sensor.compute_first_contact(self.step_dt).torch,
            ),
            dim=1,
        )

    def _feet_net_forces_history(self) -> torch.Tensor:
        return torch.cat(
            (
                self.right_foot_contact_sensor.data.net_forces_w_history.torch,
                self.left_foot_contact_sensor.data.net_forces_w_history.torch,
            ),
            dim=2,
        )

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
        sensor_cfgs = [self.cfg.right_foot_contact_sensor, self.cfg.left_foot_contact_sensor]
        if hasattr(self.cfg, "torso_ground_contact_sensor"):
            sensor_cfgs.extend(
                (self.cfg.torso_ground_contact_sensor, self.cfg.head_ground_contact_sensor)
            )
        # The URDF importer preserves the link hierarchy. Tag the exact source
        # rigid bodies before cloning so every environment inherits contact reporting.
        sensor_body_names = {sensor_cfg.prim_path.rsplit("/", 1)[-1] for sensor_cfg in sensor_cfgs}
        self._activate_contact_sensors("/World/envs/env_0/Robot", sensor_body_names)
        self.right_foot_contact_sensor = NestedBodyContactSensor(self.cfg.right_foot_contact_sensor)
        self.left_foot_contact_sensor = NestedBodyContactSensor(self.cfg.left_foot_contact_sensor)
        if hasattr(self.cfg, "torso_ground_contact_sensor"):
            self.torso_ground_contact_sensor = NestedBodyContactSensor(
                self.cfg.torso_ground_contact_sensor
            )
            self.head_ground_contact_sensor = NestedBodyContactSensor(self.cfg.head_ground_contact_sensor)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene._terrain = self._terrain
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["right_foot_contact_sensor"] = self.right_foot_contact_sensor
        self.scene.sensors["left_foot_contact_sensor"] = self.left_foot_contact_sensor
        if hasattr(self, "torso_ground_contact_sensor"):
            self.scene.sensors["torso_ground_contact_sensor"] = self.torso_ground_contact_sensor
            self.scene.sensors["head_ground_contact_sensor"] = self.head_ground_contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _activate_contact_sensors(self, robot_prim_path: str, body_names: set[str]) -> None:
        """Add the PhysX contact-report API to selected nested rigid bodies.

        ``activate_contact_sensors`` only tags the top-most rigid body (base_link) for
        this robot's nested link hierarchy, so the configured bodies are tagged here.
        """
        from pxr import UsdPhysics

        stage = sim_utils.get_current_stage()
        tagged = 0
        for prim in stage.Traverse():
            if not prim.GetPath().pathString.startswith(robot_prim_path):
                continue
            if prim.GetName() not in body_names:
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            applied = prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied:
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
            if "PhysxContactReportAPI" not in applied:
                prim.AddAppliedSchema("PhysxContactReportAPI")
            tagged += 1
        if tagged != len(body_names):
            raise RuntimeError(
                f"Expected to tag {len(body_names)} rigid bodies for contact sensing under "
                f"'{robot_prim_path}', tagged {tagged}. Check link names."
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
        # Silence the gait clock for zero-velocity commands. The clock used to tick
        # unconditionally, so a standing policy kept seeing "time to step" oscillations
        # while stand_still penalized stepping — the structural cause of the recurring
        # zero-command marching. BDX-style setups clamp the reference/phase to stance at
        # zero command; here the (0, 0) clock is a distinct "stance" signal.
        moving_mask = (
            torch.max(torch.abs(self._commands[:, :2]), dim=1).values > self.cfg.move_command_threshold
        ) | (torch.abs(self._commands[:, 2]) > self.cfg.move_command_threshold)
        gait_clock = gait_clock * moving_mask.float().unsqueeze(1)
        # NOTE: both-feet contact booleans were dropped from the observation on purpose.
        # The real BDX uses per-foot switches (Open Duck Mini even multi-point ones for
        # uneven ground), but this project deliberately ships without foot-contact hardware
        # to keep the sole simple. The gait clock already carries the phase; measured
        # contact only added an actual-vs-commanded feedback channel, and feeding a clean
        # sim contact while the real robot has none (or a flaky switch) is a sim2real trap.
        # obs is therefore 41-dim (no feet_contact). Contact still drives the imitation
        # reward (route B) and gait-phase rewards (route A) — those are training signals,
        # not policy inputs, so the real robot needs no foot sensing.

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
        air_time = self._feet_last_air_time()
        contact_time = self._feet_current_contact_time()
        first_contact = self._feet_first_contact()
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
        force_hist = self._feet_net_forces_history()
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
        # With random pushes disabled, "instability" only ever reflects the policy's own
        # wobble. Feeding that back into the gates lets a zero-command policy shuffle to
        # dodge the stand-still/anti-jitter penalties AND collect stepping rewards. Zeroing
        # it here makes stand_command stance fully constrained (planted feet, max smoothing,
        # no stepping reward). Re-enable when training with pushes again.
        if not self.cfg.enable_instability_gate:
            instability = torch.zeros_like(instability)

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
            self.cfg.walk_lean_angle,
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
        # 仅前 len(leg) 维是腿位置目标（route A action=腿；route B walk 后 4 维为脖子，
        # 由子类另行处理）。按腿宽度切片赋值，避免 action_space>10 时形状不匹配。
        n_leg = self._default_leg_joint_pos.shape[1]
        self._processed_actions[env_ids, :n_leg] = self._default_leg_joint_pos[env_ids]
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
        # snap sub-threshold vx samples to exactly 0 so the low-speed end of the range
        # keeps a crisp stance/move semantic (commands below the threshold are treated
        # as stance by the reward gates anyway).
        small_vx = torch.abs(self._commands[env_ids, 0]) < self.cfg.move_command_threshold
        self._commands[env_ids[small_vx], 0] = 0.0
        non_standing = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.stand_still_prob
        self._commands[env_ids[standing], 0] = 0.0
        non_standing[standing] = False
        turn_in_place = (torch.rand(len(env_ids), device=self.device) < self.cfg.turn_in_place_prob) & non_standing
        self._commands[env_ids[turn_in_place], 0] = 0.0
        # Of the envs that actually translate, flip a fraction to a backward vx command so
        # forward and backward walking are both inside the training distribution (before
        # this, vx was always sampled positive → only one direction was ever trained).
        moving = non_standing & ~turn_in_place
        backward = (torch.rand(len(env_ids), device=self.device) < self.cfg.backward_prob) & moving
        self._commands[env_ids[backward], 0] *= -1.0
        moving_or_turning = ~standing
        # Lateral vy command for every non-standing env: this robot has no ankle roll,
        # so lateral stability can only come from hip roll + stepping — it must be
        # explicitly trained (vy used to be always 0 and unconditionally penalized,
        # making sim2sim lateral inputs out-of-distribution → "left key falls").
        vy_cmd = sample_uniform(
            self.cfg.command_vy_range[0],
            self.cfg.command_vy_range[1],
            (len(env_ids),),
            self.device,
        )
        vy_cmd[torch.abs(vy_cmd) < self.cfg.move_command_threshold] = 0.0
        self._commands[env_ids[moving_or_turning], 1] = vy_cmd[moving_or_turning]
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
        # Reference-state init: give ~half of the vx-commanded envs a starting velocity
        # matching their (signed, possibly backward) command (default yaw≈0 so world-x ≈
        # body-x). Bootstrapping the first step from a standstill is the hardest part of
        # learning to walk; starting some envs already moving removes that barrier and
        # weakens the standing attractor.
        seeded = moving & (torch.rand(len(env_ids), device=self.device) < 0.5)
        vx_seed = self._commands[env_ids, 0] + sample_uniform(
            -0.05, 0.05, (len(env_ids),), self.device
        )
        # head-forward = forward_vx_sign * body-x; at reset yaw≈0 so world-x ≈ body-x.
        # vx_seed carries the command sign, so backward commands seed a backward velocity.
        default_root_vel[seeded, 0] = self.cfg.forward_vx_sign * vx_seed[seeded]
        # same convention for the lateral command (head-left = forward_vx_sign * body-y)
        default_root_vel[seeded, 1] = self.cfg.forward_vx_sign * self._commands[env_ids, 1][seeded]

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        denominator = max(1, len(env_ids))
        extras["Episode_Termination/fall_rate"] = (
            torch.count_nonzero(self.reset_terminated[env_ids]).item() / denominator
        )
        extras["Episode_Termination/time_out_rate"] = (
            torch.count_nonzero(self.reset_time_outs[env_ids]).item() / denominator
        )
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
    walk_lean_angle: float,
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
    command_norm = torch.max(torch.abs(commands), dim=1).values
    moving_command = (command_norm > move_command_threshold).float()
    stand_command = 1.0 - moving_command
    stepping_gate = torch.maximum(moving_command, instability)
    stable_stand_gate = stand_command * stability

    # Directional-progress gate. For a planar (vx/vy) command, gait-shaping rewards are
    # scaled by how much of the commanded speed is actually realized IN THE COMMANDED
    # DIRECTION: fwd_frac in [0,1] is 0 for a robot marching in place and 1 at/above
    # commanded speed. This kills the "march in place" local optimum — stepping only
    # pays if it produces motion the right way. Non-translating commands (turn-in-place /
    # stance) keep gate=1 so their turning/gait rewards are unaffected; and instability lifts
    # the gate so push-recovery stepping is never gated.
    # Head-frame planar velocity. URDF base_link +x points to the robot's tail (head
    # faces -x), so both body-x and body-y velocities must be sign-corrected (a 180° yaw
    # flip negates both) before comparing against the head-frame command. Without this
    # the policy learns to walk backwards relative to the head (the 07-01 "W walks
    # backwards / S falls forward" bug).
    cmd_planar = commands[:, :2]
    act_planar = forward_vx_sign * root_lin_vel_b[:, :2]
    cmd_planar_norm = torch.norm(cmd_planar, dim=1)
    forward_command = (cmd_planar_norm > move_command_threshold).float()
    # project realized planar velocity onto the commanded direction: positive when moving
    # the commanded way, negative when moving opposite → clamped to 0. Divide by the
    # command magnitude so fwd_frac hits 1 at/above the commanded speed.
    along_cmd = torch.sum(act_planar * cmd_planar, dim=1)
    cmd_norm_safe = torch.clamp(cmd_planar_norm, min=1e-3)
    fwd_frac = torch.clamp(along_cmd / (cmd_norm_safe * cmd_norm_safe), 0.0, 1.0)
    fwd_gate = torch.where(forward_command > 0.5, fwd_frac, torch.ones_like(fwd_frac))
    progress_pay = torch.maximum(fwd_gate, instability)

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # Upright with a nominal forward lean while moving. The BDX reference gait bakes a
    # deliberate trunk pitch (go_bdx ≈8°, Open Duck mini ≈3°) because this leg (like
    # ours) has no ankle roll and only ankle pitch: walking requires shifting weight to
    # the front of the support polygon. The old term rewarded absolute vertical
    # unconditionally, actively fighting that natural lean. Target projected gravity for
    # a nose-down lean of walk_lean_angle: proj_g_x = forward_vx_sign * sin(angle)
    # (head faces the forward_vx_sign direction along body-x); stance keeps a vertical
    # target.
    lean_target_x = forward_vx_sign * math.sin(walk_lean_angle) * moving_command
    upright_err = torch.square(projected_gravity_b[:, 0] - lean_target_x) + torch.square(projected_gravity_b[:, 1])
    rew_upright = rew_scale_upright * torch.exp(-4.0 * upright_err)
    rew_height = rew_scale_height * torch.exp(-20.0 * torch.square(base_height - target_base_height))
    # only penalize roll/pitch angular velocity (keep body steady, do not fight yaw)
    rew_ang_vel = rew_scale_ang_vel * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    # Track planar (vx, vy) and yaw commands. The sim2sim arrow keys map naturally to
    # these command slots, so all must be trained and rewarded (vy used to be excluded,
    # leaving lateral inputs out-of-distribution).
    lin_vel_error = torch.sum(torch.square(cmd_planar - act_planar), dim=1)
    rew_track_lin_vel = rew_scale_track_lin_vel * torch.exp(-lin_vel_error / lin_vel_track_sigma)
    ang_vel_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_track_ang_vel = rew_scale_track_ang_vel * torch.exp(-ang_vel_error / ang_vel_track_sigma)
    # Linear directional-progress reward. The exp tracking term is nearly flat near zero
    # speed, giving a weak gradient to *start* translating; this linear term pays
    # proportionally to realized speed along the commanded planar direction (fwd_frac,
    # capped at the command magnitude) so there is a constant push to move as commanded.
    # Zero for stance and turn-in-place, so it cannot be farmed by marching in place.
    rew_progress = rew_scale_progress * fwd_frac * forward_command
    # penalize deviation of lateral speed from the vy command (was an unconditional vy²
    # penalty, which conflicted with training lateral commands). Head-frame vy.
    rew_lateral = rew_scale_lateral * torch.square(commands[:, 1] - act_planar[:, 1])
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
