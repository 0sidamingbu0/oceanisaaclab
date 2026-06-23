# BDX 站立策略训练与迁移说明

本文档记录本 Isaac Lab 工程中 BDX 最小站立策略的训练接口。`/home/ocean/oceanbdx` 的 sim2sim/sim2real 工程可按这里的观测、动作和导出约定迁移。

## 任务入口

Gym task id:

```bash
Ocean-BDX-Stand-Direct-v0
```

训练入口继续使用本工程已有脚本：

```bash
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Ocean-BDX-Stand-Direct-v0 --num_envs 2048
```

回放和导出：

```bash
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py --task Ocean-BDX-Stand-Direct-v0 --num_envs 16 --checkpoint <model.pt>
```

`play.py` 会导出：

```text
logs/rsl_rl/bdx_stand_direct/<run>/exported/policy.pt
logs/rsl_rl/bdx_stand_direct/<run>/exported/policy.onnx
```

迁移到 `/home/ocean/oceanbdx` 时，将 `policy.onnx` 放到：

```text
/home/ocean/oceanbdx/policy/policy.onnx
```

## 机器人与关节约定

训练使用 [../assets/urdf/ocean.urdf](../assets/urdf/ocean.urdf)。

腿部 10 关节顺序固定为：

```text
[
  leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint, leg_r5_joint,
  leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint, leg_l5_joint,
]
```

脖子关节 `neck_n1_joint` 到 `neck_n4_joint` 不进入动作空间，训练中固定在 0。

默认站姿：

```text
default_dof_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

也就是 URDF 零位等于站立姿态。实机机械零位和结构限位偏移不要改训练定义，应在 `/home/ocean/oceanbdx/config/oceanbdx.yaml` 的 `calibration` 段处理。

## 观测格式

策略观测为 39 维，顺序与 `/home/ocean/oceanbdx/sim2sim/mujoco_sim.py` 的 `Policy.step()` 对齐：

```text
obs = [
  base_ang_vel * 0.25,               # 3, body frame gyro, rad/s
  projected_gravity,                 # 3, quat_rotate_inverse(q, [0, 0, -1])
  command * [2.0, 2.0, 0.25],        # 3, [vx, vy, wz]
  (joint_pos - default_dof_pos),     # 10, rad
  joint_vel * 0.05,                  # 10, rad/s
  last_action,                       # 10
]
```

当前站立版本里 `command = [0, 0, 0]`。后续低速行走可以在同一个观测槽里采样小速度命令，不需要改部署接口。

YIS320 迁移对应关系：

```text
quat  -> projected_gravity
gyro  -> base_ang_vel
accel -> 当前训练策略不使用
```

注意 `quat` 必须统一为 `w, x, y, z`。如果 YIS320 输出坐标系和 URDF/base_link 不一致，要先在部署侧做静态坐标变换，再构造 projected gravity。

## 动作格式

策略输出 10 维 `action`，部署侧动作解码为：

```text
target_q = default_dof_pos + 0.25 * clip(action, -1, 1)
```

训练中 `default_dof_pos` 为全 0，因此第一版等价于：

```text
target_q = 0.25 * action
```

部署侧仍建议保留软限位裁剪和动作变化率限制。

## 控制参数

Isaac Lab 训练配置中腿部 actuator 初值：

```text
kp = 50.0
kd = 2.5
effort_limit = 23.0 N*m
velocity_limit = 6.0 rad/s
```

这与 `/home/ocean/oceanbdx/config/oceanbdx.yaml` 里的 `control.rl_kp`、`control.rl_kd`、`torque_limits` 对齐。若实机上必须降增益，推荐先在 sim2sim 和 Isaac 里同步降，再导出新策略。

## 频率

当前训练环境：

```text
sim.dt = 1 / 120
decimation = 2
policy step = 60 Hz
```

你补充的实机电机控制上限为 116 Hz。迁移时建议两种选择之一：

1. 实机主循环 116 Hz，策略每 2 个周期运行一次，约 58 Hz。最接近当前训练的 60 Hz。
2. 若要策略 116 Hz，训练侧应把 `decimation` 或 `sim.dt` 同步调整后重新训练。

`/home/ocean/oceanbdx/config/oceanbdx.yaml` 目前仍写着 `dt=0.005, decimation=4`，即 200 Hz 主循环、50 Hz 策略。迁移本策略时需要按实际 116 Hz 总线能力更新该配置。

## 训练文件

主要实现文件：

```text
source/oceanisaaclab/oceanisaaclab/tasks/direct/oceanisaaclab/oceanisaaclab_env_cfg.py
source/oceanisaaclab/oceanisaaclab/tasks/direct/oceanisaaclab/oceanisaaclab_env.py
source/oceanisaaclab/oceanisaaclab/tasks/direct/oceanisaaclab/agents/rsl_rl_ppo_cfg.py
source/oceanisaaclab/oceanisaaclab/tasks/direct/oceanisaaclab/__init__.py
```

## 上实机前检查

迁移到 `oceanbdx` 前必须逐项确认：

1. Isaac Lab 解析出的 `robot.joint_names` 与 `oceanbdx.yaml` 的 `joint_names` 顺序一致。
2. `policy.num_obs` 应为 39，或保持 `0` 自动推断为 `9 + 3 * num_joints`。
3. `default_dof_pos` 为全 0。
4. `action_scale = 0.25`。
5. `ang_vel_scale = 0.25`，`dof_pos_scale = 1.0`，`dof_vel_scale = 0.05`。
6. `commands_scale = [2.0, 2.0, 0.25]`。
7. IMU 四元数顺序和坐标系已对齐 base_link。
8. 真机先吊起或支撑测试，确认 10 个关节目标方向正确后再落地。

## 后续低速行走扩展

当前代码已保留 `command = [vx, vy, wz]` 的观测槽。低速行走版本需要增加：

```text
command sampling: vx 0.0-0.3 m/s, vy 0, wz 小范围
reward: linear velocity tracking, yaw tracking, foot slip penalty, action smoothness
termination: base height, projected gravity, joint limits
```

动作和部署接口保持不变。