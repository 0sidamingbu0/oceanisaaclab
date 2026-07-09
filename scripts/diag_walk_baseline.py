"""Headless 行走诊断（无脖子 10-DOF / 57 维 walk 基线版）：测策略是否真前进/抬脚。

用于复现验证 2026-07-07 无脖子基线（model_21099）是否走出抬脚步态，作为加脖子退化的对照。
需在无脖子代码状态（如 commit 8c0edd8）下运行。手写 actor 前向避免 runner 版本问题。
关扰动/噪声测策略纯步态意图；关键量：前进速度跟踪 + 摆动脚离地间隙 + leg 追踪残差。

运行：./_isaaclab/isaaclab.sh -p scripts/diag_walk_baseline.py --checkpoint <path> --vx 0.15
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--vx", type=float, default=0.15)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import oceanisaaclab.tasks  # noqa: F401, E402
from oceanisaaclab.tasks.direct.oceanisaaclab.oceanisaaclab_walk_env_cfg import (  # noqa: E402
    OceanisaaclabWalkEnvCfg,
)

cfg = OceanisaaclabWalkEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device if args.device else "cuda:0"
for attr in ("enable_paper_disturbance", "enable_obs_noise", "enable_action_latency"):
    if hasattr(cfg, attr):
        setattr(cfg, attr, False)

env = gym.make("Ocean-BDX-Walk-Direct-v0", cfg=cfg).unwrapped

ck = torch.load(args.checkpoint, map_location=cfg.sim.device, weights_only=False)
a = ck["actor_state_dict"]
dev = cfg.sim.device
obs_mean = a["obs_normalizer._mean"].to(dev)
obs_std = a["obs_normalizer._std"].to(dev).clamp_min(1e-2)
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
    return torch.nn.functional.linear(x, W6, b6)


cmd = torch.tensor([args.vx, args.vy, args.wz], device=dev)
env.reset()
env._commands[:] = cmd
obs = env._get_observations()["policy"]
print(f"[check] obs dim = {obs.shape[1]}  normalizer dim = {obs_mean.shape[0]}  action dim = {W6.shape[0]}")
feet = env._feet_body_ids
feet_c = env._feet_contact_ids

fwd, base_h, ang_xy = [], [], []
foot_h_swing, in_contact_frac = [], []
leg_resid = []
warmup = 60
for t in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        env._commands[:] = cmd
        env.step(actions)
        env._commands[:] = cmd
        obs = env._get_observations()["policy"]
    if t < warmup:
        continue
    fwd.append(cfg.forward_vx_sign * env.robot.data.root_lin_vel_b.torch[:, 0])
    base_h.append(env.robot.data.root_pos_w.torch[:, 2])
    ang_xy.append(torch.norm(env.robot.data.root_ang_vel_b.torch[:, :2], dim=1))
    fh = env.robot.data.body_pos_w.torch[:, feet, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    fnorm = torch.norm(env.contact_sensor.data.net_forces_w.torch[:, feet_c, :], dim=-1)
    in_c = fnorm > 1.0
    swing = fh.clone()
    swing[in_c] = float("nan")
    foot_h_swing.append(swing)
    in_contact_frac.append(in_c.float())
    ref = env._reference_gait.sample(env._commands, env._phase)
    q = env.robot.data.joint_pos.torch[:, env._leg_dof_idx]
    leg_resid.append(((q - ref["joint_pos"]) ** 2).sum(dim=1))

FWD = torch.stack(fwd)
BH = torch.stack(base_h)
AXY = torch.stack(ang_xy)
FHS = torch.stack(foot_h_swing)
IC = torch.stack(in_contact_frac)
RESID = torch.stack(leg_resid)
foot_origin_offset = 0.067
clearance = 0.035

print("\n========== WALK BASELINE DIAGNOSTICS (no-neck, disturb/noise OFF) ==========")
print(f"checkpoint: {args.checkpoint}")
print(f"cmd vx={args.vx} vy={args.vy} wz={args.wz}  envs={args.num_envs}  steps={args.steps-warmup}")
print(f"\n[跟踪] 前进 vx mean={FWD.mean():.3f} std={FWD.std():.3f} (命令 {args.vx}) 跟踪比 {FWD.mean()/max(args.vx,1e-6):.2f}")
print(f"[机身] base height mean={BH.mean():.3f}  roll/pitch 角速度模 p95={torch.quantile(AXY.flatten(),0.95):.2f}")
print(f"\n[追踪残差] leg Σ(q-q̂)² mean={RESID.mean():.4f}  每关节 RMS = {(RESID.mean()/10).sqrt()*57.3:.2f} deg")
sv = FHS[~torch.isnan(FHS)]
if sv.numel() > 0:
    print(f"[脚高] 摆动脚原点高度 p50={torch.quantile(sv,0.5):.4f} p95={torch.quantile(sv,0.95):.4f} max={sv.max():.4f}")
    print(f"[抬脚] 实际离地间隙 p50={(torch.quantile(sv,0.5)-foot_origin_offset)*100:.1f}cm p95={(torch.quantile(sv,0.95)-foot_origin_offset)*100:.1f}cm  (参考 {clearance*100:.1f}cm)")
else:
    print("[脚高] 摆动脚样本为 0：双脚几乎从不单支撑（纯贴地蹭走）")
print(f"[接触] 每脚接触时间比例 mean={IC.mean():.3f} (1.0=从不抬脚)")
print("===========================================================================\n")

env.close()
simulation_app.close()
