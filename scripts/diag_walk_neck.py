"""Headless 行走诊断（walk env / 14-DOF 带脖子版）：头命令强制=0，测策略是否真抬脚。

针对 sim2sim 蹭脚问题：验证关节追踪残差是否吃掉参考抬脚余量，并区分左右脚。
- 加载 Ocean-BDX-Walk-Direct-v0（80 维 obs，3×512 actor），手写 actor 前向避免 runner 版本问题；
- 头命令 _head_commands 每步清零（复现 sim2sim 无头命令场景）；
- 关扰动/噪声/延迟，测策略"想做什么"的纯步态意图；
- 关键量：摆动脚原点离地高度分布 + 对参考步态的 leg 关节追踪残差 Σ(q-q̂)²
  （当前权重为 -25，可直接和 tensorboard 对照）。
- path-frame 位置奖励分解：检查原地扭动/摆胯是否仍能吃到 torso_pos_xy 分。

运行：./_isaaclab/isaaclab.sh -p scripts/diag_walk_neck.py --checkpoint <path> --vx 0.15
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
# 测策略内在步态意图：关全部外部扰动/观测噪声/动作延迟
cfg.enable_paper_disturbance = False
cfg.enable_obs_noise = False
cfg.enable_action_latency = False
cfg.command_vx_range = (args.vx, args.vx)
cfg.command_vy_range = (args.vy, args.vy)
cfg.command_wz_range = (args.wz, args.wz)
cfg.backward_prob = 0.0
cfg.stand_still_prob = 0.0
cfg.turn_in_place_prob = 0.0
cfg.head_command_dh_range = (0.0, 0.0)
cfg.head_command_pitch_range = (0.0, 0.0)
cfg.head_command_yaw_range = (0.0, 0.0)
cfg.head_command_roll_range = (0.0, 0.0)
cfg.control_resample_interval_s = (1.0e6, 1.0e6)

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
    return torch.tanh(torch.nn.functional.linear(x, W6, b6))


cmd = torch.tensor([args.vx, args.vy, args.wz], device=dev)
env.reset()
env._commands[:] = cmd
env._head_commands[:] = 0.0  # 强制无头命令
obs = env._get_observations()["policy"]
feet = env._feet_body_ids

fwd, base_h, ang_xy = [], [], []
foot_h_swing, in_contact_frac = [], []
leg_resid, contact_match = [], []
leg_resid_joint, tau_motor, tau_applied = [], [], []
neck_pos_hist, neck_target_hist, neck_action_hist = [], [], []
pos_pf_hist, ref_pf_hist, pos_err_hist, torso_pos_rew_hist, pf_world_dist_hist = [], [], [], [], []
warmup = 60
for t in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        env._commands[:] = cmd
        env._head_commands[:] = 0.0
        env._control_resample_left.fill_(1.0e6)
        obs_dict, _, _, _, _ = env.step(actions)
        env._commands[:] = cmd
        env._head_commands[:] = 0.0
        env._control_resample_left.fill_(1.0e6)
        obs = obs_dict["policy"]
    if t < warmup:
        continue
    fwd.append(cfg.forward_vx_sign * env.robot.data.root_lin_vel_b.torch[:, 0])
    base_h.append(env.robot.data.root_pos_w.torch[:, 2])
    ang_xy.append(torch.norm(env.robot.data.root_ang_vel_b.torch[:, :2], dim=1))
    fh = env.robot.data.body_pos_w.torch[:, feet, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    in_c = env._feet_current_contact_time() > 0.0
    swing = fh.clone()
    swing[in_c] = float("nan")
    foot_h_swing.append(swing)
    in_contact_frac.append(in_c.float())
    # 对参考步态的 leg 关节追踪残差（= 训练 leg_joint_pos 奖励 /-15）
    ref = env._reference_gait.sample(env._commands, env._phase)
    q = env.robot.data.joint_pos.torch[:, env._leg_dof_idx]
    joint_resid = (q - ref["joint_pos"]) ** 2
    leg_resid.append(joint_resid.sum(dim=1))
    leg_resid_joint.append(joint_resid)
    tau_motor.append(env._last_tau_m.clone())
    tau_applied.append(env._applied_leg_torque.clone())
    neck_pos_hist.append(env.robot.data.joint_pos.torch[:, env._neck_dof_idx].clone())
    neck_target_hist.append(env._neck_target.clone())
    neck_action_hist.append(env._actions[:, env._neck_action_slice].clone())
    ref_contact = (ref["feet_contact"] >= 0.5).float()
    contact_match.append((in_c.float() == ref_contact).float().sum(dim=1))
    base_xy = env.robot.data.root_pos_w.torch[:, :2]
    pos_pf, _ = env._path_frame.base_in_path_frame(base_xy, env._head_yaw())
    pos_err = pos_pf - ref["base_pos_pf"]
    pos_pf_hist.append(pos_pf)
    ref_pf_hist.append(ref["base_pos_pf"])
    pos_err_hist.append(pos_err)
    torso_pos_rew_hist.append(torch.exp(-env.cfg.rew_k_torso_pos_xy * torch.sum(pos_err**2, dim=1)))
    pf_world_dist_hist.append(torch.norm(env._path_frame.pos - base_xy, dim=1))

FWD = torch.stack(fwd)
BH = torch.stack(base_h)
AXY = torch.stack(ang_xy)
FHS = torch.stack(foot_h_swing)
IC = torch.stack(in_contact_frac)
RESID = torch.stack(leg_resid)
RESID_JOINT = torch.stack(leg_resid_joint)
TAU_MOTOR = torch.stack(tau_motor)
TAU_APPLIED = torch.stack(tau_applied)
NECK_POS = torch.stack(neck_pos_hist)
NECK_TARGET = torch.stack(neck_target_hist)
NECK_ACTION = torch.stack(neck_action_hist)
CM = torch.stack(contact_match)
POS_PF = torch.stack(pos_pf_hist)
REF_PF = torch.stack(ref_pf_hist)
POS_ERR = torch.stack(pos_err_hist)
TORSO_POS_REW = torch.stack(torso_pos_rew_hist)
PF_WORLD_DIST = torch.stack(pf_world_dist_hist)
foot_origin_offset = 0.067  # cfg 一致：leg5_link 原点到脚底
clearance = env._reference_gait.foot_clearance

print("\n========== WALK+NECK DIAGNOSTICS (head_cmd=0, disturb/noise OFF) ==========")
print(f"checkpoint: {args.checkpoint}")
print(f"cmd vx={args.vx} vy={args.vy} wz={args.wz}  envs={args.num_envs}  steps={args.steps-warmup}")
tracking_ratio = FWD.mean() / args.vx if abs(args.vx) > 1.0e-6 else torch.tensor(float("nan"), device=dev)
print(f"\n[跟踪] 机体前向 vx mean={FWD.mean():.3f} std={FWD.std():.3f} (命令 {args.vx}) 跟踪比 {tracking_ratio:.2f}")
print(f"[机身] base height mean={BH.mean():.3f}  roll/pitch 角速度模 p95={torch.quantile(AXY.flatten(),0.95):.2f}")
print(f"\n[追踪残差] leg Σ(q-q̂)² mean={RESID.mean():.4f}")
print(f"           每关节 RMS = {(RESID.mean()/10).sqrt():.4f} rad = {(RESID.mean()/10).sqrt()*57.3:.2f} deg")
print(f"[接触匹配] Σ I[c=ĉ] mean={CM.mean():.3f} (满分 2)")
joint_rms_deg = torch.sqrt(RESID_JOINT.mean(dim=(0, 1))) * 57.2958
print(f"[逐腿关节 RMS] right={joint_rms_deg[:5].mean():.2f}deg left={joint_rms_deg[5:].mean():.2f}deg")
tau_peak = torch.quantile(torch.abs(TAU_APPLIED).flatten(0, 1), 0.99, dim=0)
tau_clip = (torch.abs(TAU_MOTOR - TAU_APPLIED) > 0.25).float().mean(dim=(0, 1))
print(
    "[电机力矩] applied |tau| p99 right={:.2f}Nm left={:.2f}Nm; "
    "模型限幅/摩擦介入率 right={:.2%} left={:.2%}".format(
        tau_peak[:5].mean(), tau_peak[5:].mean(), tau_clip[:5].mean(), tau_clip[5:].mean()
    )
)
print("\n[path-frame xy]")
print(
    f"pos_pf mean=({POS_PF[...,0].mean():+.4f}, {POS_PF[...,1].mean():+.4f}) "
    f"std=({POS_PF[...,0].std():.4f}, {POS_PF[...,1].std():.4f})"
)
print(
    f"ref_pf mean=({REF_PF[...,0].mean():+.4f}, {REF_PF[...,1].mean():+.4f}) "
    f"std=({REF_PF[...,0].std():.4f}, {REF_PF[...,1].std():.4f})"
)
print(
    f"err x/y mean=({POS_ERR[...,0].mean():+.4f}, {POS_ERR[...,1].mean():+.4f}) "
    f"rms=({torch.sqrt(torch.mean(POS_ERR[...,0]**2)):.4f}, {torch.sqrt(torch.mean(POS_ERR[...,1]**2)):.4f})"
)
print(
    f"torso_pos_xy instant reward mean={TORSO_POS_REW.mean():.4f} "
    f"p50={torch.quantile(TORSO_POS_REW.flatten(),0.50):.4f} "
    f"p95={torch.quantile(TORSO_POS_REW.flatten(),0.95):.4f}"
)
print(
    f"path-frame world distance mean={PF_WORLD_DIST.mean():.4f} "
    f"p95={torch.quantile(PF_WORLD_DIST.flatten(),0.95):.4f} "
    f"(clamp {env.cfg.path_frame_max_pos_deviation:.2f}m)"
)
print(f"\n[脚高] 支撑相脚原点离地 ≈ {foot_origin_offset:.3f} m；参考摆动峰值应 ≈ {foot_origin_offset+clearance:.3f} m")
for foot_idx, foot_name in enumerate(("right", "left")):
    sv = FHS[..., foot_idx]
    sv = sv[~torch.isnan(sv)]
    contact_rate = IC[..., foot_idx].mean()
    if sv.numel() > 0:
        swing_clear_p50 = torch.quantile(sv, 0.5) - foot_origin_offset
        swing_clear_p95 = torch.quantile(sv, 0.95) - foot_origin_offset
        low_clearance = (sv - foot_origin_offset < 0.015).float().mean()
        print(
            f"[抬脚-{foot_name}] swing={1.0-contact_rate:.3f} contact={contact_rate:.3f} "
            f"clearance p50/p95={swing_clear_p50*100:.2f}/{swing_clear_p95*100:.2f}cm "
            f"<1.5cm={low_clearance:.2%} max={(sv.max()-foot_origin_offset)*100:.2f}cm"
        )
    else:
        print(f"[抬脚-{foot_name}] 无摆动样本，contact={contact_rate:.3f}")
neck_pos_rms = torch.sqrt(torch.mean(torch.square(NECK_POS), dim=(0, 1)))
neck_target_rms = torch.sqrt(torch.mean(torch.square(NECK_TARGET), dim=(0, 1)))
neck_action_rms = torch.sqrt(torch.mean(torch.square(NECK_ACTION), dim=(0, 1)))
print(f"[脖子] q RMS rad={neck_pos_rms.tolist()}")
print(f"       target RMS rad={neck_target_rms.tolist()} action RMS={neck_action_rms.tolist()}")
print("===========================================================================\n")

env.close()
simulation_app.close()
