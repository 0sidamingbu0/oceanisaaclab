"""参考步态开环回放体检（路线 B）：把策略动作直接固定成参考关节角，
验证**参考步态库本身**在物理里能否大致行走——即动力学可行性，不需要 checkpoint。

原理：walk env 的动作管线是 target_joint_pos = default_leg(恒 0) + action_joint_ranges * action
（逐关节线性映射，BDX 论文附录 A）。要让目标角 == 参考帧关节角 ref_q，就令
action = (ref_q - default) / action_joint_ranges。
每个控制步按 env 当前相位采样参考帧、下发为动作、步进物理，统计机器人是否真前进、
是否站得住不摔、脚步接触时序是否贴合参考。

为纯粹测「参考步态本身可不可行」，关掉一切扰动（推力/DR/观测噪声/动作延迟）。

用法（用 Isaac Lab 自带解释器）：
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx 0.2
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx 0.2 --viz kit  # 开GUI观察
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx 0 --vy 0.1
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx 0        # 零命令站立
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx -0.2     # 后退

--viz 是 AppLauncher 自带参数（--visualizer 别名），接后端 CSV（kit/newton/rerun/viser）；
不加 --viz 为 headless 纯统计模式，加了则开图形窗口实时播放并循环回放。

判读达标线（关掉扰动、开环）：
    - fall 率 ≈ 0（参考步态若动力学不可行会成片摔）；base height 稳定在 ~0.385；
    - 头前向 vx 跟踪比 > ~0.6（开环无反馈，能到命令的六七成即算可跟）；
    - 实测接触与参考接触一致度 > ~0.8；摆动脚脚底间隙接近库里的 foot_clearance。
    不达标 → 调 gen_reference_gait.py 的 base_height / foot_clearance / lateral_sway /
    walk_lean_angle 重新生成库。
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--vx", type=float, default=0.2, help="[m/s] 头前向速度命令")
parser.add_argument("--vy", type=float, default=0.0, help="[m/s] 头左向速度命令")
parser.add_argument("--wz", type=float, default=0.0, help="[rad/s] 逆时针偏航命令")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# AppLauncher 自带 --viz/--visualizer（接后端 CSV，如 `--viz kit`）：给了就开图形窗口，
# 此时实时节奏播放、统计跑完后持续循环回放；不给则 headless 纯统计。
viz_on = bool(args.visualizer)
args.headless = not viz_on

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
# 纯测参考步态本身的动力学可行性：关掉一切扰动/域随机化/噪声/延迟
cfg.enable_random_push = False
cfg.enable_domain_rand = False
cfg.enable_obs_noise = False
cfg.enable_action_latency = False

env = gym.make("Ocean-BDX-Walk-Direct-v0", cfg=cfg).unwrapped

dev = cfg.sim.device
# 论文版动作管线：target = default_leg(0) + action_joint_ranges * action（逐关节映射）
joint_ranges = torch.tensor(cfg.action_joint_ranges, device=dev, dtype=torch.float32)
fsign = cfg.forward_vx_sign
foot_offset = cfg.foot_origin_offset
cmd = torch.tensor([args.vx, args.vy, args.wz], device=dev, dtype=torch.float32)

env.reset()
env._commands[:] = cmd
feet = env._feet_body_ids
default_leg = env._default_leg_joint_pos  # (N,10)，腿部 default 恒 0
default_neck = env._default_neck_joint_pos
neck_ranges = env._neck_joint_ranges
env._head_commands.zero_()


def reference_action(ref):
    action = torch.zeros(env.num_envs, cfg.action_space, device=dev)
    action[:, env._leg_action_slice] = (ref["joint_pos"] - default_leg) / joint_ranges
    action[:, env._neck_action_slice] = (ref["neck_pos"] - default_neck) / neck_ranges
    return action.clamp(-1.0, 1.0)

# RSI 探针：reset 之后、任何 step 之前，直接读姿态，隔离「RSI 写入是否就已倾倒」
pg0 = env.robot.data.projected_gravity_b.torch
h0 = env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
tilt0 = (-pg0[:, 2] < cfg.walk_min_upright_projection)
print("\n[RSI探针] reset 后未 step：not_upright(倾>49°) env={}/{}  "
      "base_h mean={:.3f}  -proj_g_z min={:.3f} (阈值{:.2f})".format(
          int(tilt0.sum()), args.num_envs, float(h0.mean()), float((-pg0[:, 2]).min()),
          cfg.walk_min_upright_projection))

N = args.num_envs
# 每个 env 只统计其「首次摔倒之前」的一段生命（避免摔倒后乱翻的脚/姿态污染指标）。
fallen = torch.zeros(N, dtype=torch.bool, device=dev)
first_fall = torch.full((N,), args.steps, dtype=torch.long, device=dev)  # 没摔=steps
term_cause = []  # (step, base_too_low, not_upright, joint_oob, terminated) 前 40 步
vx_head, vy_head, yaw = [], [], []
base_h, ang_xy = [], []
contact_match, foot_h_swing = [], []

for t in range(args.steps):
    with torch.inference_mode():
        env._commands[:] = cmd
        env._head_commands.zero_()
        phase = env._gait_phase()
        ref = env._reference_gait.sample(env._commands, phase)
        env.step(reference_action(ref))
        env._commands[:] = cmd
        env._head_commands.zero_()
    if viz_on:
        time.sleep(float(env.step_dt))  # 实时节奏，便于肉眼观察

    term = env.reset_terminated
    # 终止原因分解探针：复刻 base _get_dones 的三个条件，看首次摔倒是哪一项触发
    if t < 40:
        rp = env.robot.data.root_pos_w.torch
        pg = env.robot.data.projected_gravity_b.torch
        jp = env.robot.data.joint_pos.torch[:, env._leg_dof_idx]
        lo = env.robot.data.soft_joint_pos_limits.torch[:, env._leg_dof_idx, 0]
        hi = env.robot.data.soft_joint_pos_limits.torch[:, env._leg_dof_idx, 1]
        base_height = rp[:, 2] - env.scene.env_origins[:, 2]
        c_low = base_height < cfg.walk_min_base_height
        c_tilt = (-pg[:, 2] < cfg.walk_min_upright_projection)
        c_joint = torch.any((jp < lo) | (jp > hi), dim=1)
        term_cause.append((t, int(c_low.sum()), int(c_tilt.sum()), int(c_joint.sum()), int(term.sum())))
    newly = term & ~fallen
    first_fall[newly] = t
    fallen |= term
    alive = ~fallen  # 排除已摔过(含本步刚摔并被 RSI 重置)的 env

    def masked(vec):
        out = vec.clone().float()
        out[~alive] = float("nan")
        return out

    lin_b = env.robot.data.root_lin_vel_b.torch
    vx_head.append(masked(fsign * lin_b[:, 0]))
    vy_head.append(masked(fsign * lin_b[:, 1]))
    yaw.append(masked(env.robot.data.root_ang_vel_b.torch[:, 2]))
    base_h.append(masked(env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]))
    ang_xy.append(masked(torch.norm(env.robot.data.root_ang_vel_b.torch[:, :2], dim=1)))

    ref_now = env._reference_gait.sample(env._commands, env._gait_phase())
    in_c = (env._feet_current_contact_time() > 0.0).float()
    ref_contact = (ref_now["feet_contact"] >= 0.5).float()
    contact_match.append(masked(1.0 - torch.mean(torch.abs(in_c - ref_contact), dim=1)))

    fh = (
        env.robot.data.body_pos_w.torch[:, feet, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
        - foot_offset
    )
    swing = fh.clone()
    swing[in_c > 0.5] = float("nan")          # 只看摆动脚
    swing[~alive] = float("nan")              # 排除已摔 env
    foot_h_swing.append(swing)


def nanmean(x):
    return float(torch.nanmean(x))


def nanstd(x):
    v = x[~torch.isnan(x)]
    return float(v.std()) if v.numel() > 1 else float("nan")


VXH = torch.stack(vx_head)
VYH = torch.stack(vy_head)
YAW = torch.stack(yaw)
BH = torch.stack(base_h)
AXY = torch.stack(ang_xy)
CM = torch.stack(contact_match)
FHS = torch.stack(foot_h_swing)

step_dt = float(env.step_dt)
never_fell = int((first_fall == args.steps).sum().item())
survive_steps = first_fall.float()
mean_surv = float(survive_steps.mean()) * step_dt
med_surv = float(survive_steps.median()) * step_dt

print("\n========== REFERENCE-GAIT OPEN-LOOP PLAYBACK (无扰动, 只统计摔前) ==========")
print(f"command: vx={args.vx:+.3f} vy={args.vy:+.3f} wz={args.wz:+.3f}   "
      f"envs={N}  steps={args.steps}  step_dt={step_dt:.4f}s  joint_ranges={cfg.action_joint_ranges}")
print(f"gait_period={env._reference_gait.gait_period}  gait_duty={env._reference_gait.gait_duty}")
print("注: 双足开环关节回放本就无平衡反馈, 迟早会摔; 看的是摔前是否朝命令方向走、"
      "抬脚/接触对不对、能撑多久, 不是 fall≈0。")

print("\n[存活] 首次摔倒 平均={:.2f}s 中位={:.2f}s (满程 {:.2f}s)  全程未摔 env={}/{}".format(
    mean_surv, med_surv, args.steps * step_dt, never_fell, N))
print("[终止原因] 前若干步 (base_too_low / not_upright / joint_oob / terminated):")
for row in term_cause[:12]:
    print("  step {:2d}: low={:2d} tilt={:2d} joint={:2d} term={:2d}".format(*row))
print(f"[稳定性] base height [m] mean={nanmean(BH):.3f} std={nanstd(BH):.3f}  (参考≈0.385)")
print("[稳定性] roll/pitch 角速度模 [rad/s] mean={:.3f} max={:.2f}  (抖动指标)".format(
    nanmean(AXY), float(AXY[~torch.isnan(AXY)].max()) if (~torch.isnan(AXY)).any() else float("nan")))


def _ratio(x, target):
    return nanmean(x) / target if abs(target) > 1e-6 else float("nan")


print("\n[跟踪] 头前向 vx [m/s] mean={:+.3f} std={:.3f}  (命令 {:+.3f}) -> 跟踪比 {:.2f}".format(
    nanmean(VXH), nanstd(VXH), args.vx, _ratio(VXH, args.vx)))
print("[跟踪] 头左向 vy [m/s] mean={:+.3f} std={:.3f}  (命令 {:+.3f}) -> 跟踪比 {:.2f}".format(
    nanmean(VYH), nanstd(VYH), args.vy, _ratio(VYH, args.vy)))
print("[跟踪] yaw rate [rad/s] mean={:+.3f} std={:.3f}  (命令 {:+.3f}) -> 跟踪比 {:.2f}".format(
    nanmean(YAW), nanstd(YAW), args.wz, _ratio(YAW, args.wz)))

print("\n[接触] 实测/参考接触一致度 mean={:.3f}  (1.0=时序完全贴合参考; <0.8=脚步没跟上)".format(
    nanmean(CM)))
swing_valid = FHS[~torch.isnan(FHS)]
if swing_valid.numel() > 0:
    print("[抬脚] 摆动脚脚底间隙 [m] mean={:.4f} p50={:.4f} p95={:.4f}  (对照库里 foot_clearance≈0.035)".format(
        float(swing_valid.mean()), float(torch.quantile(swing_valid, 0.5)),
        float(torch.quantile(swing_valid, 0.95))))
else:
    print("[抬脚] 无摆动脚样本（双脚几乎全程着地——零命令站立时正常）")
print("============================================================================\n")

# 可视化模式：统计跑完后持续循环回放，方便肉眼观察（关窗 / Ctrl-C 退出）
if viz_on:
    print("[viz] 持续循环回放中，关闭窗口或 Ctrl-C 退出 ...")
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                env._commands[:] = cmd
                env._head_commands.zero_()
                ref = env._reference_gait.sample(env._commands, env._gait_phase())
                env.step(reference_action(ref))
                env._commands[:] = cmd
                env._head_commands.zero_()
            time.sleep(float(env.step_dt))
    except KeyboardInterrupt:
        print("\n[viz] stopped.")

env.close()
simulation_app.close()
