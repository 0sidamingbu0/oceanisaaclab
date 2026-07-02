"""Headless 方向诊断：验证 base_link +x 与机器人头部朝向的关系。

背景（2026-07-02）：sim2sim 实测 07-01 checkpoint「+vx 命令朝物理后方走得好」。
URDF 静态分析：neck_n1_joint 挂在 base_link x=-0.074（头部侧），腿挂在 x=+0.021，
leg_r1 在 y=+0.045 / leg_l1 在 y=-0.047（右=+y，与面朝 -x 自洽）→ base_link +x 指向尾部。

本脚本零动作 spawn 机器人，直接在 sim 里量两件事，给出最终判定：
1. 头部 link（neck_n4_link）原点在 base 系下的 x 坐标：应为负（头在 -x 侧）。
2. 给 root 一个沿 body +x 的初速度，看机器人相对头部朝哪边移动。

用法（PC 端 Isaac Lab 环境）：
    ./_isaaclab/isaaclab.sh -p scripts/diag_direction.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
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
cfg.enable_random_push = False
cfg.enable_obs_noise = False

env = gym.make("Ocean-BDX-Stand-Direct-v0", cfg=cfg).unwrapped
env.reset()
env._commands[:] = 0.0

# --- 1) 头部 link 在 base 系下的位置 ---
head_ids, head_names = env.robot.find_bodies("neck_n4_link")
base_ids, _ = env.robot.find_bodies("base_link")
zero_actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
for _ in range(30):  # 静置几步让姿态稳定
    env.step(zero_actions)
    env._commands[:] = 0.0

base_pos = env.robot.data.body_pos_w.torch[:, base_ids[0]]
base_quat = env.robot.data.body_quat_w.torch[:, base_ids[0]]  # (w,x,y,z)
head_pos = env.robot.data.body_pos_w.torch[:, head_ids[0]]

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

head_in_base = quat_apply_inverse(base_quat, head_pos - base_pos)
head_x = head_in_base[:, 0].mean().item()
print("\n================ DIRECTION DIAGNOSTICS ================")
print(f"[1] {head_names[0]} 原点在 base 系下: x={head_x:+.4f} m "
      f"(y={head_in_base[:, 1].mean():+.4f}, z={head_in_base[:, 2].mean():+.4f})")
print(f"    -> 头部在 base_link 的 {'−x（尾部=+x，需要 forward_vx_sign=-1）' if head_x < 0 else '+x（+x 即前向，forward_vx_sign 应为 +1）'} 侧")

# --- 2) 沿 body +x 给初速度，看相对头部往哪边走 ---
env.reset()
env._commands[:] = 0.0
root_vel = env.robot.data.default_root_vel.torch.clone()
root_vel[:, 0] = 0.3  # world +x（reset 后 yaw=0，world x ≈ body x）
env.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=torch.arange(env.num_envs, device=env.device))
start = env.robot.data.root_pos_w.torch[:, :2].clone()
for _ in range(20):
    env.step(zero_actions)
    env._commands[:] = 0.0
disp = (env.robot.data.root_pos_w.torch[:, :2] - start).mean(dim=0)
print(f"[2] 给 body+x 初速度 0.3 m/s 后位移: dx={disp[0]:+.3f} dy={disp[1]:+.3f} m")
print(f"    -> body +x 位移方向与 [1] 结合判断: +x 移动 = {'远离头部（后退）' if head_x < 0 else '朝头部（前进）'}")
print(f"\n结论: forward_vx_sign 应为 {'-1.0' if head_x < 0 else '+1.0'}（当前 cfg = {cfg.forward_vx_sign}）")
print("=======================================================\n")

env.close()
simulation_app.close()
