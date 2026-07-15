# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-v0`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import oceanisaaclab.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument(
    "--debug_policy_steps",
    type=int,
    default=0,
    help="Print policy obs/action/target diagnostics for the first N play steps, then exit.",
)
parser.add_argument(
    "--debug_push_steps",
    type=int,
    default=0,
    help="Run a deterministic play push test for N steps, print push/base/joint diagnostics, then exit.",
)
parser.add_argument("--debug_push_start", type=int, default=80, help="Step at which the debug push starts.")
parser.add_argument("--debug_push_duration", type=int, default=18, help="Debug push duration in policy steps.")
parser.add_argument("--debug_push_force_x", type=float, default=0.0, help="Debug push force X in world frame [N].")
parser.add_argument("--debug_push_force_y", type=float, default=60.0, help="Debug push force Y in world frame [N].")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if args_cli.debug_push_steps:
            env.unwrapped.cfg.enable_random_push = False

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        foot_body_ids = []
        foot_body_names = []
        if args_cli.debug_push_steps:
            configured_foot_ids = getattr(env.unwrapped, "_feet_body_ids", None)
            if configured_foot_ids is not None:
                foot_body_ids = [int(body_id) for body_id in configured_foot_ids]
                foot_body_names = [env.unwrapped.robot.body_names[body_id] for body_id in foot_body_ids]
            else:
                for body_id, body_name in enumerate(env.unwrapped.robot.body_names):
                    body_name_lower = body_name.lower()
                    if (
                        "foot" in body_name_lower
                        or "ankle" in body_name_lower
                        or body_name_lower.endswith("leg_r5_link")
                        or body_name_lower.endswith("leg_l5_link")
                    ):
                        foot_body_ids.append(body_id)
                        foot_body_names.append(body_name)
            print(
                "[debug_push] "
                f"force_w=[{args_cli.debug_push_force_x:.1f},{args_cli.debug_push_force_y:.1f},0.0]N "
                f"start={args_cli.debug_push_start} duration={args_cli.debug_push_duration} "
                f"foot_bodies={foot_body_names}"
            )

        # reset environment
        obs = env.get_observations()
        timestep = 0
        debug_foot_stats = None
        if args_cli.debug_push_steps and foot_body_ids and hasattr(env.unwrapped, "_feet_current_contact_time"):
            num_debug_feet = len(foot_body_ids)
            debug_foot_stats = {
                "initial_pos_w": None,
                "latest_pos_w": None,
                "previous_pos_w": None,
                "previous_contact": None,
                "max_xy_delta_w": torch.zeros(num_debug_feet, 2),
                "max_xy_distance": torch.zeros(num_debug_feet),
                "max_liftoff_height": torch.zeros(num_debug_feet),
                "liftoff_count": torch.zeros(num_debug_feet, dtype=torch.int64),
                "touchdown_count": torch.zeros(num_debug_feet, dtype=torch.int64),
                "liftoff_origin_xy_w": torch.zeros(num_debug_feet, 2),
                "liftoff_origin_z_w": torch.zeros(num_debug_feet),
                "liftoff_origin_valid": torch.zeros(num_debug_feet, dtype=torch.bool),
                "max_touchdown_step_delta_w": torch.zeros(num_debug_feet, 2),
                "max_touchdown_step_distance": torch.zeros(num_debug_feet),
                "reset_count": 0,
            }
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    if args_cli.debug_push_steps:
                        env_unwrapped = env.unwrapped
                        push_active = (
                            args_cli.debug_push_start
                            <= timestep
                            < args_cli.debug_push_start + args_cli.debug_push_duration
                        )
                        push_forces = torch.zeros_like(env_unwrapped._push_forces)
                        push_torques = torch.zeros_like(env_unwrapped._push_torques)
                        if push_active:
                            push_forces[:, 0, 0] = args_cli.debug_push_force_x
                            push_forces[:, 0, 1] = args_cli.debug_push_force_y
                        env_unwrapped.robot.permanent_wrench_composer.set_forces_and_torques(
                            push_forces,
                            push_torques,
                            body_ids=env_unwrapped._base_body_id,
                            is_global=True,
                        )
                        debug_foot_pos_w = None
                        debug_foot_contact = None
                        if debug_foot_stats is not None:
                            debug_foot_pos_w = (
                                env_unwrapped.robot.data.body_pos_w.torch[0, foot_body_ids].detach().cpu()
                            )
                            debug_foot_contact = (
                                env_unwrapped._feet_current_contact_time()[0].detach().cpu() > 0.0
                            )
                            if debug_foot_contact.numel() != len(foot_body_ids):
                                raise RuntimeError(
                                    "Debug foot body/contact count mismatch: "
                                    f"{len(foot_body_ids)} bodies vs {debug_foot_contact.numel()} contacts."
                                )

                            if debug_foot_stats["initial_pos_w"] is None:
                                debug_foot_stats["initial_pos_w"] = debug_foot_pos_w.clone()
                            else:
                                initial_pos_w = debug_foot_stats["initial_pos_w"]
                                xy_delta_w = debug_foot_pos_w[:, :2] - initial_pos_w[:, :2]
                                xy_distance = torch.norm(xy_delta_w, dim=1)
                                farther_from_start = xy_distance > debug_foot_stats["max_xy_distance"]
                                debug_foot_stats["max_xy_distance"] = torch.maximum(
                                    debug_foot_stats["max_xy_distance"], xy_distance
                                )
                                debug_foot_stats["max_xy_delta_w"][farther_from_start] = xy_delta_w[
                                    farther_from_start
                                ]

                                previous_contact = debug_foot_stats["previous_contact"]
                                previous_pos_w = debug_foot_stats["previous_pos_w"]
                                if previous_contact is not None:
                                    liftoff = previous_contact & ~debug_foot_contact
                                    touchdown = ~previous_contact & debug_foot_contact
                                    debug_foot_stats["liftoff_count"] += liftoff.to(torch.int64)
                                    debug_foot_stats["liftoff_origin_xy_w"][liftoff] = previous_pos_w[
                                        liftoff, :2
                                    ]
                                    debug_foot_stats["liftoff_origin_z_w"][liftoff] = previous_pos_w[
                                        liftoff, 2
                                    ]
                                    debug_foot_stats["liftoff_origin_valid"][liftoff] = True

                                    valid_airborne = (
                                        ~debug_foot_contact
                                        & debug_foot_stats["liftoff_origin_valid"]
                                    )
                                    liftoff_height = torch.clamp(
                                        debug_foot_pos_w[:, 2]
                                        - debug_foot_stats["liftoff_origin_z_w"],
                                        min=0.0,
                                    )
                                    debug_foot_stats["max_liftoff_height"][valid_airborne] = (
                                        torch.maximum(
                                            debug_foot_stats["max_liftoff_height"][valid_airborne],
                                            liftoff_height[valid_airborne],
                                        )
                                    )

                                    valid_touchdown = touchdown & debug_foot_stats["liftoff_origin_valid"]
                                    debug_foot_stats["touchdown_count"] += valid_touchdown.to(torch.int64)
                                    touchdown_step_delta_w = (
                                        debug_foot_pos_w[:, :2] - debug_foot_stats["liftoff_origin_xy_w"]
                                    )
                                    touchdown_step_distance = torch.norm(touchdown_step_delta_w, dim=1)
                                    longer_step = (
                                        valid_touchdown
                                        & (
                                            touchdown_step_distance
                                            > debug_foot_stats["max_touchdown_step_distance"]
                                        )
                                    )
                                    debug_foot_stats["max_touchdown_step_distance"] = torch.maximum(
                                        debug_foot_stats["max_touchdown_step_distance"],
                                        torch.where(
                                            valid_touchdown,
                                            touchdown_step_distance,
                                            torch.zeros_like(touchdown_step_distance),
                                        ),
                                    )
                                    debug_foot_stats["max_touchdown_step_delta_w"][longer_step] = (
                                        touchdown_step_delta_w[longer_step]
                                    )
                                    debug_foot_stats["liftoff_origin_valid"][touchdown] = False

                            debug_foot_stats["latest_pos_w"] = debug_foot_pos_w.clone()
                            debug_foot_stats["previous_pos_w"] = debug_foot_pos_w.clone()
                            debug_foot_stats["previous_contact"] = debug_foot_contact.clone()

                        should_print = push_active or timestep % 10 == 0
                        if should_print:
                            leg_ids = env_unwrapped._leg_dof_idx
                            q0 = env_unwrapped.robot.data.joint_pos.torch[0, leg_ids].detach().cpu()
                            dq0 = env_unwrapped.robot.data.joint_vel.torch[0, leg_ids].detach().cpu()
                            action0 = actions[0].detach().cpu()
                            clipped0 = action0.clamp(-1.0, 1.0)
                            # 路线 B（论文版）是逐关节映射 action_joint_ranges，路线 A 是标量 action_scale
                            ranges0 = getattr(env_unwrapped.cfg, "action_joint_ranges", None)
                            scale0 = (
                                torch.tensor(ranges0) if ranges0 is not None else env_unwrapped.cfg.action_scale
                            )
                            leg_action0 = clipped0[: len(leg_ids)]
                            target0 = (
                                env_unwrapped._default_leg_joint_pos[0].detach().cpu()
                                + scale0 * leg_action0
                            )
                            root_pos = env_unwrapped.robot.data.root_pos_w.torch[0].detach().cpu()
                            root_lin_vel_b = env_unwrapped.robot.data.root_lin_vel_b.torch[0].detach().cpu()
                            root_ang_vel_b = env_unwrapped.robot.data.root_ang_vel_b.torch[0].detach().cpu()
                            gravity_b = env_unwrapped.robot.data.projected_gravity_b.torch[0].detach().cpu()
                            foot_state_text = ""
                            if debug_foot_pos_w is not None:
                                initial_pos_w = debug_foot_stats["initial_pos_w"]
                                foot_xy_delta_w = debug_foot_pos_w[:, :2] - initial_pos_w[:, :2]
                                foot_lift = torch.clamp(
                                    debug_foot_pos_w[:, 2] - initial_pos_w[:, 2], min=0.0
                                )
                                foot_state_text = (
                                    f" foot_xy_delta_w={foot_xy_delta_w.numpy().round(3).tolist()}"
                                    f" foot_lift={foot_lift.numpy().round(3).tolist()}"
                                    f" foot_contact={debug_foot_contact.to(torch.int8).tolist()}"
                                )
                            print(
                                "[debug_push] "
                                f"step={timestep} active={int(push_active)} "
                                f"force_w=[{push_forces[0, 0, 0].item():+.1f},{push_forces[0, 0, 1].item():+.1f},0.0] "
                                f"base_pos={root_pos.numpy().round(3).tolist()} "
                                f"lin_vel_b={root_lin_vel_b.numpy().round(3).tolist()} "
                                f"ang_vel_b={root_ang_vel_b.numpy().round(3).tolist()} "
                                f"grav_b={gravity_b.numpy().round(3).tolist()} "
                                f"q={q0.numpy().round(3).tolist()} "
                                f"dq={dq0.numpy().round(3).tolist()} "
                                f"action={action0.numpy().round(3).tolist()} "
                                f"target={target0.numpy().round(3).tolist()}"
                                f"{foot_state_text}"
                            )
                    if args_cli.debug_policy_steps and timestep < args_cli.debug_policy_steps:
                        env_unwrapped = env.unwrapped
                        action0 = actions[0].detach().cpu()
                        obs0 = obs[0].detach().cpu() if isinstance(obs, torch.Tensor) else obs["policy"][0].detach().cpu()
                        clipped0 = action0.clamp(-1.0, 1.0)
                        leg_ids = getattr(env_unwrapped, "_leg_dof_idx", None)
                        if leg_ids is not None:
                            joint_names = [env_unwrapped.robot.joint_names[i] for i in leg_ids]
                            q0 = env_unwrapped.robot.data.joint_pos.torch[0, leg_ids].detach().cpu()
                            dq0 = env_unwrapped.robot.data.joint_vel.torch[0, leg_ids].detach().cpu()
                            default0 = env_unwrapped._default_leg_joint_pos[0].detach().cpu()
                            ranges0 = getattr(env_unwrapped.cfg, "action_joint_ranges", None)
                            scale0 = (
                                torch.tensor(ranges0) if ranges0 is not None else env_unwrapped.cfg.action_scale
                            )
                            leg_action0 = clipped0[: len(leg_ids)]
                            target0 = default0 + scale0 * leg_action0
                            print(f"[debug_play] step={timestep} joint_names={joint_names}")
                            print(
                                "[debug_play] "
                                f"q={q0.numpy().round(3).tolist()} "
                                f"dq={dq0.numpy().round(3).tolist()} "
                                f"action={action0.numpy().round(3).tolist()} "
                                f"target={target0.numpy().round(3).tolist()}"
                            )
                        print(
                            "[debug_play] "
                            f"obs_first9={obs0[:9].numpy().round(3).tolist()} "
                            f"action_absmax={float(action0.abs().max()):.3f} "
                            f"clipped_absmax={float(clipped0.abs().max()):.3f} "
                            f"sat_count={int((clipped0.abs() > 0.98).sum())}/{clipped0.numel()}"
                        )
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    if debug_foot_stats is not None and bool(dones.reshape(-1)[0].item()):
                        debug_foot_stats["reset_count"] += 1
                        debug_foot_stats["previous_pos_w"] = None
                        debug_foot_stats["previous_contact"] = None
                        debug_foot_stats["liftoff_origin_valid"].zero_()
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break
                elif args_cli.debug_policy_steps:
                    timestep += 1
                    if timestep >= args_cli.debug_policy_steps:
                        break
                elif args_cli.debug_push_steps:
                    timestep += 1
                    if timestep >= args_cli.debug_push_steps:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            if debug_foot_stats is not None and debug_foot_stats["initial_pos_w"] is not None:
                initial_pos_w = debug_foot_stats["initial_pos_w"]
                latest_pos_w = debug_foot_stats["latest_pos_w"]
                print(f"[debug_push_summary] episode_resets={debug_foot_stats['reset_count']}")
                for foot_index, foot_name in enumerate(foot_body_names):
                    final_xy_delta_w = latest_pos_w[foot_index, :2] - initial_pos_w[foot_index, :2]
                    print(
                        "[debug_push_summary] "
                        f"foot={foot_name} "
                        f"initial_xy_w={initial_pos_w[foot_index, :2].numpy().round(4).tolist()} "
                        f"final_xy_w={latest_pos_w[foot_index, :2].numpy().round(4).tolist()} "
                        f"final_xy_delta_w={final_xy_delta_w.numpy().round(4).tolist()} "
                        f"max_xy_delta_w={debug_foot_stats['max_xy_delta_w'][foot_index].numpy().round(4).tolist()} "
                        f"max_xy_distance={debug_foot_stats['max_xy_distance'][foot_index].item():.4f}m "
                        f"max_liftoff_height={debug_foot_stats['max_liftoff_height'][foot_index].item():.4f}m "
                        f"liftoffs={debug_foot_stats['liftoff_count'][foot_index].item()} "
                        f"touchdowns={debug_foot_stats['touchdown_count'][foot_index].item()} "
                        "max_touchdown_step_delta_w="
                        f"{debug_foot_stats['max_touchdown_step_delta_w'][foot_index].numpy().round(4).tolist()} "
                        "max_touchdown_step_distance="
                        f"{debug_foot_stats['max_touchdown_step_distance'][foot_index].item():.4f}m"
                    )

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
