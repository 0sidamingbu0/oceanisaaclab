"""Headless diagnostics for a 77-D perpetual-standing policy.

The test fixes every torso/head command to the shared neutral pose and disables
disturbance, observation noise and action latency. It reports reset consistency,
foot spacing, torso jitter, contact chatter and joint speed.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import oceanisaaclab.tasks  # noqa: F401, E402
from oceanisaaclab.tasks.direct.oceanisaaclab.oceanisaaclab_stand_env_cfg import (  # noqa: E402
    OceanisaaclabStandEnvCfg,
)

cfg = OceanisaaclabStandEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device if args.device else "cuda:0"
cfg.enable_paper_disturbance = False
cfg.enable_obs_noise = False
cfg.enable_action_latency = False
cfg.stand_disturbance_quiet_prob = 1.0
cfg.stand_zero_command_prob = 1.0
cfg.stand_rsi_prob = 1.0
cfg.torso_command_h_range = (0.0, 0.0)
cfg.torso_command_pitch_range = (0.0, 0.0)
cfg.torso_command_yaw_range = (0.0, 0.0)
cfg.torso_command_roll_range = (0.0, 0.0)
cfg.head_command_dh_range = (0.0, 0.0)
cfg.head_command_pitch_range = (0.0, 0.0)
cfg.head_command_yaw_range = (0.0, 0.0)
cfg.head_command_roll_range = (0.0, 0.0)

env = gym.make("Ocean-BDX-StandPaper-Direct-v0", cfg=cfg).unwrapped

# Load the normalized MLP directly so this diagnostic remains independent of runner versions.
ck = torch.load(args.checkpoint, map_location=cfg.sim.device, weights_only=False)
a = ck["actor_state_dict"]
dev = cfg.sim.device
obs_mean = a["obs_normalizer._mean"].to(dev)
obs_std = a["obs_normalizer._std"].to(dev).clamp_min(1.0e-2)
W0, b0 = a["mlp.0.weight"].to(dev), a["mlp.0.bias"].to(dev)
W2, b2 = a["mlp.2.weight"].to(dev), a["mlp.2.bias"].to(dev)
W4, b4 = a["mlp.4.weight"].to(dev), a["mlp.4.bias"].to(dev)
W6, b6 = a["mlp.6.weight"].to(dev), a["mlp.6.bias"].to(dev)
elu = torch.nn.functional.elu


def policy(obs: torch.Tensor) -> torch.Tensor:
    x = (obs - obs_mean) / obs_std
    x = elu(torch.nn.functional.linear(x, W0, b0))
    x = elu(torch.nn.functional.linear(x, W2, b2))
    x = elu(torch.nn.functional.linear(x, W4, b4))
    return torch.tanh(torch.nn.functional.linear(x, W6, b6))


obs_dict = env.reset()[0]
obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
if obs.shape[1] != 77 or obs_mean.shape[0] != 77:
    raise RuntimeError(
        f"Expected a new 77-D stand checkpoint/environment, got obs={obs.shape[1]} "
        f"checkpoint={obs_mean.shape[0]}. Old 74-D checkpoints are incompatible."
    )

q0 = env.robot.data.joint_pos.torch[:, env._leg_dof_idx]
reset_target_error = torch.max(torch.abs(env._filtered_joint_target - q0)).item()
neck_q0 = env.robot.data.joint_pos.torch[:, env._neck_dof_idx]
reset_neck_target_error = torch.max(torch.abs(env._filtered_neck_target - neck_q0)).item()
history_error = torch.max(
    torch.abs(env._action_history - env._actions.unsqueeze(1))
).item()
prev_action_error = max(
    torch.max(torch.abs(env._previous_actions - env._actions)).item(),
    torch.max(torch.abs(env._prev_prev_actions - env._actions)).item(),
)

heights = []
projected_gravity = []
force_mag = []
force_rate = []
joint_speed = []
foot_in_contact = []
both_contact = []
foot_spacing = []
prev_force = None
warmup = min(50, max(0, args.steps // 4))
for step in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs_dict, _, _, _, _ = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    if step < warmup:
        continue
    heights.append(env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2])
    projected_gravity.append(env.robot.data.projected_gravity_b.torch)
    foot_force = env._feet_net_forces_history()[:, -1]
    force_norm = torch.norm(foot_force, dim=-1)
    in_contact = force_norm > 1.0
    force_mag.append(force_norm)
    joint_speed.append(env.robot.data.joint_vel.torch[:, env._leg_dof_idx].abs())
    foot_in_contact.append(in_contact.float())
    both_contact.append(torch.all(in_contact, dim=1).float())
    feet_xy = env.robot.data.body_pos_w.torch[:, env._feet_body_ids, :2]
    foot_spacing.append(torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=1))
    if prev_force is not None:
        force_rate.append(torch.norm(foot_force - prev_force, dim=-1))
    prev_force = foot_force

H = torch.stack(heights)
PG = torch.stack(projected_gravity)
F = torch.stack(force_mag)
JS = torch.stack(joint_speed)
IC = torch.stack(foot_in_contact)
BC = torch.stack(both_contact)
FS = torch.stack(foot_spacing)
FR = torch.stack(force_rate) if force_rate else torch.zeros_like(F[:1])

print("\n================ STAND POLICY DIAGNOSTICS (neutral, disturbances OFF) ================")
print(f"checkpoint: {args.checkpoint}")
print(f"obs/action={obs.shape[1]}/{W6.shape[0]}  envs={args.num_envs}  measured_steps={len(H)}")
print(
    f"reset consistency: leg target-q max={reset_target_error:.2e} rad  "
    f"neck target-q max={reset_neck_target_error:.2e} rad  "
    f"action-ring max={history_error:.2e}  previous-action max={prev_action_error:.2e}"
)
print(
    f"base height [m]: mean={H.mean():.4f} std={H.std():.4f}  "
    f"per-env jitter={H.std(dim=0).mean():.5f}"
)
print(
    f"upright projection -g_z: mean={(-PG[..., 2]).mean():.4f}  "
    f"minimum={(-PG[..., 2]).min():.4f}"
)
print(f"foot spacing [m]: mean={FS.mean():.4f} std={FS.std():.4f}")
print(f"contact force/foot [N]: mean={F.mean():.2f} max={F.max():.1f}")
print(
    f"contact-force rate [N/step]: mean={FR.mean():.2f} "
    f"p95={torch.quantile(FR.flatten(), 0.95):.2f}"
)
print(
    f"leg joint speed [rad/s]: mean={JS.mean():.3f} "
    f"p95={torch.quantile(JS.flatten(), 0.95):.3f} max={JS.max():.2f}"
)
print(f"per-foot contact fraction={IC.mean():.4f}  double-support fraction={BC.mean():.4f}")
lift_events = ((IC[:-1] > 0.5) & (IC[1:] < 0.5)).float().sum(dim=0).mean()
print(f"average contact->air events per foot={lift_events:.2f}")
print("=======================================================================================\n")

env.close()
simulation_app.close()
