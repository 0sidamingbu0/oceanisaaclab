# BDX 论文站立策略训练与部署

本文档描述路线 B 的 perpetual standing policy。它与 periodic walking policy 分开训练，
但共享机器人模型、14 维动作语义、低层执行器、path frame 和运行时动作历史。

## 任务与模型

任务 id：

```text
Ocean-BDX-StandPaper-Direct-v0
```

训练日志与导出目录：

```text
logs/rsl_rl/bdx_stand_perpetual/<run>/
logs/rsl_rl/bdx_stand_perpetual/<run>/exported/policy.pt
logs/rsl_rl/bdx_stand_perpetual/<run>/exported/policy.onnx
logs/rsl_rl/bdx_stand_perpetual/<run>/exported/policy.onnx.data
```

站立和行走是两个独立模型：

| 策略 | Policy obs | Critic obs | Action | 相位 | 命令尾部 |
|---|---:|---:|---:|---|---|
| StandPaper | 77 | 79 | 14 | 无 | torso 4 + head 4 |
| Walk | 80 | 82 | 14 | 二阶谐波 4 | velocity 3 + head 4 |

旧 `74/76` 维 StandPaper checkpoint/ONNX 与当前接口不兼容，不能 resume，必须重新训练和导出。

## 观测布局

站立 policy 观测为 77 维：

```text
[0:2]   torso xy in path frame * 4.0
[2:4]   sin(yaw_pf), cos(yaw_pf)
[4:7]   projected gravity in body frame
[7:10]  body linear velocity * 2.0
[10:13] body angular velocity * 0.25
[13:23] leg joint position * 1.0
[23:27] neck joint position * 1.0
[27:37] leg joint velocity * 0.05
[37:41] neck joint velocity * 0.05
[41:55] a_(t-1), 14-D normalized action
[55:69] a_(t-2), 14-D normalized action
[69:73] torso command (h, pitch, yaw, roll) * (10, 1, 1, 1)
[73:77] head command (dh, pitch, yaw, roll) * (20, 1, 1, 1)
```

Critic 在无噪声 77 维状态后追加摩擦和质量随机化系数，共 79 维。足底接触只用于奖励和
切换判定，不进入 policy 观测。`projected_gravity` 必须由 base_link 姿态得到，不能省略；
它提供站立平衡所需的绝对 roll/pitch 信息。

torso 命令范围来自本机完整 `5^4=625` 点双脚约束 IK 扫描：

```text
h      [-0.040, +0.010] m
pitch  [-0.17, +0.17] rad
yaw    [-0.24, +0.24] rad
roll   [-0.09, +0.09] rad
```

torso/head 命令在 episode 内每 `4-8s` 重采样，而不是只在 reset 时固定一次；这对应论文
perpetual policy 接收连续控制输入的设定。

## 动作与低层控制

动作顺序固定为 10 个腿关节和 4 个脖子关节：

```text
leg_r1..r5, leg_l1..l5, neck_n1..n4
```

网络输出先裁剪到 `[-1, 1]`，再逐关节映射：

```text
leg range  = [0.35, 0.35, 0.8, 0.9, 0.8] * 2
neck range = [0.8, 0.8, 1.2, 0.7]
target     = default_joint_pos + range * action
```

训练控制链与论文一致：

```text
50Hz policy -> 14-D FOH -> 37.5Hz low-pass -> 200Hz low-level
```

腿目标还会围绕实测关节角按 `tau_max / kP` 限幅。腿部使用附录 B Go1 软件执行器模型：
`kP=10`、`kD=0.3`、峰值力矩约 `23.7Nm`，并包含编码器偏移、摩擦、背隙、速度相关力矩
限制、噪声和反射惯量。脖子目标同样经过 FOH/低通，再进入 `stiffness=50`、`damping=2`
的位置伺服。旧文档中的统一 `50/2.5` 腿部 PD 和标量 `action_scale=0.25` 不适用于该任务。

## 参考、初始化与脚距

`stand_pose.npz` 的精确零命令被强制为 URDF q=0，并与 `reference_gait.npz` 的零速帧共享：

```text
base height = 0.38498640060424805 m
base_pos_pf = [0.04638837, -0.00091510] m
q_leg       = 0
foot span   = 0.25169745297 m
```

环境启动时检查站立/行走零速腿角、base height 和脚 link-frame yaw 标定一致，防止两策略
在零速下形成不同脚距。

站立 reset 使用独立 RSI，不再执行 walking RSI：

- `stand_zero_command_prob=0.5`：一半环境的 torso/head 命令同时为零，覆盖公共 neutral 状态。
- `stand_rsi_prob=0.8`：非 neutral 环境可从命令对应参考姿态出生；其余从 neutral 学习过渡。
- episode 内的 torso/head 命令每 `4-8s` 再采样一次，使策略学习连续姿态切换。
- 物理 q、动作延迟环、`a_(t-1)`、`a_(t-2)`、腿/脖子 FOH 和低通目标全部由同一个
  可表示 setpoint 初始化，避免第一步重放陈旧行走动作。

站立 path frame 以 1 秒时间常数收敛到双脚中心和双脚平均 heading。左右脚末端 link frame
在 q=0 时相差约 pi，因此必须先减去参考资产中的各脚 yaw 偏置，再做圆均值；不能用躯干 yaw
替代双脚 heading。

## 奖励与扰动

站立使用论文 Table I 权重：

| 项 | 核/权重 |
|---|---|
| Torso position xy | `exp(-200 * error^2)`, weight 1 |
| Torso orientation | `exp(-20 * error^2)`, weight 1 |
| Linear velocity xy/z | `exp(-8 * error^2)`, weight 1/1 |
| Angular velocity xy/z | `exp(-2 * error^2)`, weight 0.5/0.5 |
| Leg / neck joint position | negative L2, 15 / 100 |
| Leg / neck joint velocity | negative L2, 0.001 / 1 |
| Contact match | 1 per foot |
| All joint torques / accelerations | leg + neck negative L2, 0.001 / 2.5e-6 |
| Leg / neck action rate | negative L2, 1.5 / 5 |
| Leg / neck action acceleration | negative L2, 0.45 / 5 |
| Survival | 20 |

躯干高度命令通过站立腿关节参考进入模仿奖励，不再添加论文之外的独立 torso-height reward。
论文三档扰动从训练开始，在前 1500 iteration 线性放开。

当前站立参考仍是双脚约束的运动学 IK，不是论文原系统的完整逆动力学/CoP 优化，因此通过
收窄命令域做硬件适配。完整 625 点网格的最大脚位置残差为 `2.899mm`；6250 个腿关节标量中
没有 action-range 或 soft-limit 夹持，最大原始归一化动作约 `0.7364`。该参考仍不应被解释成
真实 CoP/ZMP 轨迹。

## 训练、回放与诊断

从头训练：

```bash
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Ocean-BDX-StandPaper-Direct-v0 \
  --num_envs 8192 \
  --max_iterations 100000 \
  --headless
```

回放并导出：

```bash
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Ocean-BDX-StandPaper-Direct-v0 \
  --num_envs 16 \
  --viz kit \
  --checkpoint logs/rsl_rl/bdx_stand_perpetual/<run>/model_<iter>.pt
```

无界面诊断：

```bash
./_isaaclab/isaaclab.sh -p scripts/diag_stand.py \
  --checkpoint logs/rsl_rl/bdx_stand_perpetual/<run>/model_<iter>.pt \
  --num_envs 64 --steps 400 --headless
```

重点关注 `fall_rate`、双脚接触率、关节速度、动作率、零命令脚距，以及切换第一帧的
`a_(t-1)/a_(t-2)` 和目标角连续性。

## OceanBDX 部署

将新导出的站立模型文件放到：

```text
/home/ocean/oceanbdx/policy/stand/policy.onnx
/home/ocean/oceanbdx/policy/stand/policy.onnx.data
```

行走模型继续使用：

```text
/home/ocean/oceanbdx/policy/policy.onnx
/home/ocean/oceanbdx/policy/policy.onnx.data
```

Python MuJoCo sim2sim 已支持 stand 77 / walk 80 双模型。策略切换必须满足：

1. 两个策略共享最近两帧实际归一化动作。
2. 保留 path frame、当前下发 setpoint、腿/脖子 FOH 和低通状态。
3. 切换请求只在 200Hz 主控制循环边界消费；stand→walk 初始化合适的行走相位。
4. walk→stand 等待下一次确认的双支撑，并从实测躯干高度、pitch/yaw/roll 初始化 torso 命令，
   不把正在行走的姿态突然命令到全零；head 命令保持连续。
5. 两个策略使用同一套腿部 `10/0.3` 附录 B 低层模型。
6. `STAND_UP→RL_STAND` 先用实测躯干姿态初始化命令，再用 `0.5s` 余弦接管平滑回
   用户/neutral 命令，同时渐入腿/脖子策略目标并从脚本 `50/3` 接到 RL `10/0.3`，避免
   新目标在高增益阶段造成力矩冲击。

MuJoCo 的接触判定继续使用 sole geom，但 path-frame 运动学必须使用末端脚 link 的 body
origin/quaternion，才能对应 IsaacLab 的 `body_pos_w/body_quat_w`。此前使用 sole geom center
会在 neutral pose 制造约 `2.585cm` 的训练/部署 path-frame 位置偏差。

OceanBDX 的 C++ 真机主控目前仍是旧 41 维、10 动作、单策略路径；Python `--real` 仅把
MuJoCo 产生的目标通过桥发送给电机，策略观测仍来自 MuJoCo，不是使用真机状态估计器的完整
闭环。因此完成 C++ 77/80 双模型、14 动作、path frame、状态估计和切换逻辑前，不能宣称
真机已支持无缝切换。

本次脖子 plant 增加了 FOH/低通。旧 80 维 walking ONNX 在维度上仍可加载，但训练 plant
已经变化，必须至少完整回归零速、各方向行走、头命令和策略切换；推荐重新训练或微调后导出。

## 当前验证基线

- 64 env、1 PPO iteration smoke test：`77/79/14`，`fall_rate=0`，无 traceback/NaN。
- 范围内非零 yaw RSI：参考重构误差约 `2.7e-7rad`，solved/Isaac 双脚 heading 差约
  `0.00176rad`。
- reset 后腿/脖子 target-q、动作延迟环和前两帧动作误差均为 0。
- stand/walk q=0 脚距差约 `2.1e-12m`，base height 与 `base_pos_pf` 完全一致。
- 站立参考最坏 IK 残差 `2.899mm`，action-range/soft-limit 夹持为 0。
- Python sim2sim 轻量测试通过 77/80 维观测、动作历史传递、控制周期边界、当前/下一双支撑、
  零速冻结及超时取消逻辑。
- 真实 MuJoCo 用当前 80 维 walking policy 作为 standing fallback 验证接管状态机：首个 RL
  控制步腿目标/力矩跳变为 `2.24e-5rad / 0.079Nm`，整个 `0.5s` 窗口最大单步变化为
  `0.00681rad / 0.175Nm`；10 秒末仍为双脚支撑的 `RL_STAND`。该测试不能替代重新训练后的
  77 维 standing policy 闭环验收。
