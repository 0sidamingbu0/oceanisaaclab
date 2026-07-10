"""Inspect PhysX tensor paths for contact reporting on the nested Ocean URDF."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import oceanisaaclab.tasks  # noqa: F401, E402
from oceanisaaclab.tasks.direct.oceanisaaclab.oceanisaaclab_walk_env_cfg import (  # noqa: E402
    OceanisaaclabWalkEnvCfg,
)


cfg = OceanisaaclabWalkEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.enable_obs_noise = False
cfg.enable_domain_rand = False
cfg.robot_cfg.spawn.self_collision = False
cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = False
env = gym.make("Ocean-BDX-Walk-Direct-v0", cfg=cfg).unwrapped

view = env.robot.root_physx_view
print("body_names:", env.robot.body_names)
for attribute in ("prim_paths", "link_paths", "link_names", "shared_metatype"):
    value = getattr(view, attribute, "<missing>")
    if attribute == "shared_metatype" and value != "<missing>":
        value = {
            name: getattr(value, name)
            for name in dir(value)
            if "link" in name and not name.startswith("_")
        }
    print(f"root_physx_view.{attribute}:", value)

sensor_names = ("right_foot_contact_sensor", "left_foot_contact_sensor")
for name in sensor_names:
    sensor = getattr(env, name)
    print(
        name,
        "count=",
        sensor.body_physx_view.count,
        "paths=",
        list(sensor.body_physx_view.prim_paths),
        "contact_sensor_count=",
        sensor.contact_view.sensor_count,
    )

actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
force_samples = []
contact_samples = {name: [] for name in sensor_names}
head_body_id = env.robot.find_bodies("neck_n4_link")[0][0]
print("first control steps:")
for step in range(100):
    env.step(actions)
    wrench = wp.to_torch(view.get_link_incoming_joint_force()).clone()
    force_samples.append(torch.linalg.norm(wrench[..., :3], dim=-1))
    for name in contact_samples:
        sensor_force = getattr(env, name).data.net_forces_w.torch
        contact_samples[name].append(torch.linalg.norm(sensor_force, dim=-1).clone())
    if step < 5:
        root_height = env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        head_height = (
            env.robot.data.body_pos_w.torch[:, head_body_id, 2] - env.scene.env_origins[:, 2]
        )
        upright = -env.robot.data.projected_gravity_b.torch[:, 2]
        print(
            f"  step={step + 1}",
            f"root_z=[{root_height.min().item():.4f}, {root_height.max().item():.4f}]",
            f"head_z=[{head_height.min().item():.4f}, {head_height.max().item():.4f}]",
            f"upright=[{upright.min().item():.4f}, {upright.max().item():.4f}]",
            f"terminated={int(env.reset_terminated.count_nonzero())}/{env.num_envs}",
        )
force_norm = torch.stack(force_samples)
print("incoming joint force norm mean/max by link:")
for index, body_name in enumerate(env.robot.body_names):
    print(
        f"  {index:2d} {body_name:16s}",
        f"mean={force_norm[..., index].mean().item():8.3f}",
        f"max={force_norm[..., index].amax().item():8.3f}",
    )
print("contact sensor force norm mean/max:")
for name, samples in contact_samples.items():
    norms = torch.stack(samples)
    print(name, f"mean={norms.mean().item():8.3f}", f"max={norms.amax().item():8.3f}")

env_ids = torch.arange(env.num_envs, device=env.device)
env._reset_idx(env_ids)
print("normal reset terminated:")
for step in range(5):
    env.step(actions)
    root_height = env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    upright = -env.robot.data.projected_gravity_b.torch[:, 2]
    print(
        f"  step={step + 1}",
        f"terminated={int(env.reset_terminated.count_nonzero())}/{env.num_envs}",
        f"root_z_min={root_height.min().item():.4f}",
        f"upright_min={upright.min().item():.4f}",
    )

root_pose = env.robot.data.default_root_pose.torch.clone()
root_pose[:, :3] = env.scene.env_origins
root_pose[:, 2] += 0.08
env.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
env.robot.write_root_velocity_to_sim_index(
    root_velocity=torch.zeros_like(env.robot.data.default_root_vel.torch), env_ids=env_ids
)
env.step(actions)
print(
    "forced low-pose terminated:",
    f"{int(env.reset_terminated.count_nonzero())}/{env.num_envs}",
)

env.close()
simulation_app.close()
