"""Headless 行走诊断：强制给定 vx/wz 命令，测量策略是否真前进/转向，
以及脚 link 原点离地高度分布（用来解释 feet_clearance 恒为 0），机身抖动。
复用 diag_stand 的手写 actor 加载方式，避免 rsl_rl runner 版本问题。"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--vx", type=float, default=0.3)
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
from oceanisaaclab.tasks.direct.oceanisaaclab.oceanisaaclab_env_cfg import OceanisaaclabEnvCfg  # noqa: E402

cfg = OceanisaaclabEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device if args.device else "cuda:0"
cfg.enable_random_push = False  # 测内在行走能力，不测抗推

env = gym.make("Ocean-BDX-Stand-Direct-v0", cfg=cfg).unwrapped

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
feet = env._feet_body_ids
feet_c = env._feet_contact_ids

fwd, yaw, base_h = [], [], []
lat = []
foot_h_all, foot_h_swing, in_contact_frac, both_c = [], [], [], []
ang_xy = []
warmup = 60
for t in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        env._commands[:] = cmd  # 抵消 reset 时的随机重采样
        env.step(actions)
        env._commands[:] = cmd
        obs = env._get_observations()["policy"]
    if t < warmup:
        continue
    fwd.append(cfg.forward_vx_sign * env.robot.data.root_lin_vel_b.torch[:, 0])
    lat.append(cfg.forward_vx_sign * env.robot.data.root_lin_vel_b.torch[:, 1])
    yaw.append(env.robot.data.root_ang_vel_b.torch[:, 2])
    base_h.append(env.robot.data.root_pos_w.torch[:, 2])
    ang_xy.append(torch.norm(env.robot.data.root_ang_vel_b.torch[:, :2], dim=1))
    fh = env.robot.data.body_pos_w.torch[:, feet, 2] - env.scene.env_origins[:, 2].unsqueeze(1)  # (N,2)
    fnorm = torch.norm(env.contact_sensor.data.net_forces_w.torch[:, feet_c, :], dim=-1)
    in_c = fnorm > 1.0
    foot_h_all.append(fh)
    swing = fh.clone()
    swing[in_c] = float("nan")  # 只看摆动(非接触)脚的高度
    foot_h_swing.append(swing)
    in_contact_frac.append(in_c.float())
    both_c.append((in_c.sum(dim=1) == 2).float())

FWD = torch.stack(fwd)
LAT = torch.stack(lat)
YAW = torch.stack(yaw)
BH = torch.stack(base_h)
AXY = torch.stack(ang_xy)
FHA = torch.stack(foot_h_all)
FHS = torch.stack(foot_h_swing)
IC = torch.stack(in_contact_frac)
BC = torch.stack(both_c)

print("\n================ WALK POLICY DIAGNOSTICS (push OFF) ================")
print(f"checkpoint: {args.checkpoint}")
print(f"commanded vx={args.vx}  vy={args.vy}  wz={args.wz}   envs={args.num_envs}  measured_steps={args.steps-warmup}")
print(f"\n[跟踪] 实际前进(头部朝向) vx [m/s] mean={FWD.mean():.3f} std={FWD.std():.3f}  (命令 {args.vx}) -> 跟踪比 {FWD.mean()/max(args.vx,1e-6):.2f}")
print(f"[跟踪] 实际侧向(头部左向) vy [m/s] mean={LAT.mean():.3f} std={LAT.std():.3f}  (命令 {args.vy}) -> 跟踪比 {LAT.mean()/args.vy if abs(args.vy)>1e-6 else float('nan'):.2f}")
print(f"[跟踪] 实际 yaw rate [rad/s] mean={YAW.mean():.3f} std={YAW.std():.3f}  (命令 {args.wz})")
print(f"\n[机身] base height [m] mean={BH.mean():.3f} std={BH.std():.3f}  (目标 0.42)")
print(f"[机身] roll/pitch 角速度模 [rad/s] mean={AXY.mean():.3f} p95={torch.quantile(AXY.flatten(),0.95):.2f} max={AXY.max():.2f}  (抖动指标)")
print(f"\n[脚高] 脚 link 原点离地 [m] 全程 mean={FHA.mean():.4f} min={FHA.min():.4f} max={FHA.max():.4f}")
swing_valid = FHS[~torch.isnan(FHS)]
if swing_valid.numel() > 0:
    print(f"[脚高] 摆动(离地)脚原点高度 [m] mean={swing_valid.mean():.4f} p50={torch.quantile(swing_valid,0.5):.4f} p95={torch.quantile(swing_valid,0.95):.4f} max={swing_valid.max():.4f}")
    print(f"       -> foot_clearance_target=0.05 相对该分布是否偏低? (若 min 都 >0.05 则该惩罚恒为0)")
else:
    print("[脚高] 摆动脚样本为 0：脚几乎从不离地(纯贴地/双支撑)")
print(f"\n[接触] 每只脚接触时间比例 mean={IC.mean():.3f} (1.0=从不抬脚)")
print(f"[接触] 双脚同时着地比例 mean={BC.mean():.3f} (1.0=从不单支撑)")
lift = ((IC[:-1] > 0.5) & (IC[1:] < 0.5)).float().sum(dim=0).mean()
print(f"[接触] 每只脚离地事件数/{args.steps-warmup}步: {lift:.1f}")
print("====================================================================\n")

env.close()
simulation_app.close()
