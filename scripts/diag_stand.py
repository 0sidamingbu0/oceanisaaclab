"""Headless diagnostic: load a trained stand policy, run it, and measure real physical
quantities to decide reward-weight tuning (contact-force magnitude, base height,
foot-lift/jitter frequency, joint velocity)."""

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

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import oceanisaaclab.tasks  # noqa: F401, E402
from oceanisaaclab.tasks.direct.oceanisaaclab.oceanisaaclab_env_cfg import OceanisaaclabEnvCfg  # noqa: E402

cfg = OceanisaaclabEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device if args.device else "cuda:0"
# turn OFF random push so we measure intrinsic standing stability, not push recovery
cfg.enable_random_push = False

env = gym.make("Ocean-BDX-Stand-Direct-v0", cfg=cfg).unwrapped

# load actor directly from checkpoint (obs normalizer + MLP), avoids rsl_rl runner version issues
import os  # noqa: E402

ck = torch.load(args.checkpoint, map_location=cfg.sim.device, weights_only=False)
a = ck["actor_state_dict"]
dev = cfg.sim.device
obs_mean = a["obs_normalizer._mean"].to(dev)
obs_std = a["obs_normalizer._std"].to(dev).clamp_min(1e-2)  # floor std: command dims are constant 0 (std=0) under stand_still_prob=1.0
W0, b0 = a["mlp.0.weight"].to(dev), a["mlp.0.bias"].to(dev)
W2, b2 = a["mlp.2.weight"].to(dev), a["mlp.2.bias"].to(dev)
W4, b4 = a["mlp.4.weight"].to(dev), a["mlp.4.bias"].to(dev)
W6, b6 = a["mlp.6.weight"].to(dev), a["mlp.6.bias"].to(dev)
elu = torch.nn.functional.elu


def policy(o):
    x = (o - obs_mean) / obs_std
    x = elu(torch.nn.functional.linear(x, W0, b0))
    x = elu(torch.nn.functional.linear(x, W2, b2))
    x = elu(torch.nn.functional.linear(x, W4, b4))
    return torch.nn.functional.linear(x, W6, b6)  # mean action (deterministic)


obs_dict = env.reset()[0]
obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
feet = env._feet_body_ids
feet_c = env._feet_contact_ids

heights, force_mag, force_rate, joint_speed, foot_in_contact, both_contact = [], [], [], [], [], []
prev_force = None
warmup = 50  # let it settle
for t in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs_dict, _, _, _, _ = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    if t < warmup:
        continue
    h = env.robot.data.root_pos_w.torch[:, 2]
    fz = env.contact_sensor.data.net_forces_w.torch[:, feet_c, :]  # (N,2,3)
    fnorm = torch.norm(fz, dim=-1)  # (N,2)
    in_c = fnorm > 1.0
    jv = env.robot.data.joint_vel.torch[:, env._leg_dof_idx]
    heights.append(h)
    force_mag.append(fnorm)
    joint_speed.append(jv.abs())
    foot_in_contact.append(in_c.float())
    both_contact.append((in_c.sum(dim=1) == 2).float())
    if prev_force is not None:
        force_rate.append(torch.norm(fz - prev_force, dim=-1))  # (N,2)
    prev_force = fz

H = torch.stack(heights)        # (T,N)
F = torch.stack(force_mag)      # (T,N,2)
FR = torch.stack(force_rate)    # (T-1,N,2)
JS = torch.stack(joint_speed)   # (T,N,10)
IC = torch.stack(foot_in_contact)  # (T,N,2)
BC = torch.stack(both_contact)  # (T,N)

print("\n================ STAND POLICY DIAGNOSTICS (push OFF) ================")
print(f"checkpoint: {args.checkpoint}")
print(f"envs={args.num_envs}  measured_steps={args.steps-warmup}  target_height=0.42m")
print(f"\nbase height [m]      mean={H.mean():.3f}  std={H.std():.3f}  min={H.min():.3f}  max={H.max():.3f}")
print(f"  -> per-env height std (jitter up/down): {H.std(dim=0).mean():.4f}")
print(f"\ncontact force/foot [N] mean={F.mean():.2f}  max={F.max():.1f}  (robot weight ~ 2*mean if both feet)")
print(f"contact-force RATE/foot [N/step] mean={FR.mean():.2f}  p95={torch.quantile(FR.flatten(),0.95):.1f}  max={FR.max():.1f}")
print(f"\nleg joint speed [rad/s] mean={JS.mean():.3f}  p95={torch.quantile(JS.flatten(),0.95):.3f}  max={JS.max():.2f}")
print(f"\nfraction of time each foot in contact: {IC.mean():.3f}  (1.0=never lifts)")
print(f"fraction of time BOTH feet down      : {BC.mean():.3f}  (1.0=never single-support)")
# count foot-lift events (contact -> no contact transitions) as jitter proxy
lift_events = ((IC[:-1] > 0.5) & (IC[1:] < 0.5)).float().sum(dim=0).mean()
print(f"avg foot-lift events / foot over {args.steps-warmup} steps: {lift_events:.1f}  (high => chattering)")
print("====================================================================\n")

env.close()
simulation_app.close()
