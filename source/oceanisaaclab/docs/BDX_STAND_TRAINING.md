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

旧 `74/76` 维 StandPaper checkpoint/ONNX 与当前接口形状不兼容。此前的 77 维模型虽然形状
匹配，但训练时只使用约 1/3 head 命令域，也没有当前受扰恢复门控与探索分布；它不能 resume
后当作本配置的结果，更不能直接接收完整 head 命令。当前 StandPaper 必须从头训练和重新导出。

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

站立 head 命令恢复 `neck_head_map.npz` 的完整可达域：

```text
dh     [-0.020, +0.020] m
pitch  [-0.50, +0.50] rad
yaw    [-1.00, +1.00] rad
roll   [-0.60, +0.60] rad
```

walking 为抑制行进时的头部甩动，继续使用约 1/3 范围：`±0.007m / ±0.17 / ±0.33 /
±0.20rad`。两套范围必须按当前实际运行的 policy 分开限幅，不能用 walking 限幅削弱站立表现，
也不能把完整站立命令送给 walking policy。

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

静态参考的 contact schedule 仍为双脚支撑，但论文的受扰实验允许策略偏离参考轨迹和接触序列
来恢复平衡。为避免双脚接触和静态腿模仿把策略锁死在原地，环境用一个连续的恢复门控调整现有
Table I 项，而不是新增脚高、指定左右摆动脚或迈步奖励。门控只使用真机可获得的量：

| 失稳量 | 开始放松 | 完全放松 |
|---|---:|---:|
| 相对命令 torso 姿态的 roll/pitch 倾斜误差 | `0.08rad` | `0.22rad` |
| body 水平速度 | `0.18m/s` | `0.45m/s` |
| body roll/pitch 角速度 | `0.45rad/s` | `1.40rad/s` |
| path-frame torso 平面位置误差 | `0.025m` | `0.065m` |

四路瞬时信号取最大值。倾斜量相对命令姿态计算，正常的 torso pitch/roll 表演命令不会被误判
为失衡；torso/head 命令每次重采样后的 `0.35s` grace period 内只把瞬时 tilt signal 置零，
避免姿态参考跳变被当作推力。body 水平速度、roll/pitch 角速度和 path-frame 位置误差仍能开启
恢复。数据扫描表明 stand_pose 全命令域的 `base_pos_pf` 相对 neutral 最大只变化 `1.70mm`，远
低于 position gate 的 `25mm` 起点，因此命令本身不会误触发位置门控；持续保留位置通道还能
避免重采样后出现 `0.35s` 的慢速平移失稳盲区。

raw `recovery_signal` 直接跟随更大的瞬时信号，因而快速开启；瞬时信号超过 `0.10` 时开始或
刷新恢复保持。一旦恢复开始，只要仍是单脚支撑就持续刷新保持时间，强静态约束不会在摆动脚
落地前回来；重新双脚支撑后再保持 `0.30s`，然后按满量程 `0.40s` 的速率线性慢释放。若释放
期间再次失稳，信号立即重新升高。

真正作用于奖励的 `recovery_gate = recovery_signal * recovery_gate_scale`。scale 与论文扰动
课程同步，从训练开始在前 1500 iteration（36000 environment steps）由 0 线性升到 1，避免
随机初始策略第一轮就关闭双脚接触和强腿模仿，先建立稳站基线再学习捕获步。课程放满且
`recovery_signal=1` 时，双脚接触权重缩放到 0，腿关节位置权重从 `-15` 放松到 `-3`，腿
action rate 从 `-1.5` 放松到 `-0.25`，腿 action acceleration 从 `-0.45` 放松到 `-0.075`。
躯干位置/姿态、线/角速度、survival、力矩和关节加速度奖励始终保留，因此策略仍需把身体带回
稳定区，而不是仅靠抬脚得分。

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

Stand PPO 使用 `entropy_coef=0.001` 和 squashed Gaussian `min_std=0.05`。这是为了避免前
1500 iteration 扰动逐步放大前探索方差已经塌缩；不要继续使用旧站立配置的
`entropy_coef=0` / `min_std=0.03` checkpoint。

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

重点关注 `fall_rate`、双脚接触率、关节速度、动作率、零命令脚距，以及以下恢复指标：

- `Metrics/recovery_signal_mean`：课程缩放前、含接触保持/慢释放状态机的 raw 恢复信号均值。
- `Metrics/recovery_gate_mean`：乘以课程 scale 后、实际作用到奖励权重的有效门控均值。
- `Curriculum/recovery_gate_scale`：恢复门控对奖励的课程比例，前 36000 steps 从 0 升到 1。
- `Metrics/recovery_active_rate`：有效 `gate > 0.10` 的时间占比。
- `Metrics/recovery_single_support_rate`：恢复激活期间单脚支撑占比。
- `Metrics/liftoff_events_per_episode`：每个 episode 双脚合计离地事件数。
- `Metrics/recovery_liftoff_events_per_episode`：恢复激活时发生的离地事件数。
- `Metrics/touchdown_events_per_episode`：离地后重新接触地面的事件数。
- `Metrics/touchdown_step_distance_cm`：所有 touchdown 从对应 liftoff 到落脚的平均脚 XY 距离。
- `Metrics/recovery_touchdown_events_per_episode`：离地或落脚发生在恢复期的 touchdown 数。
- `Metrics/recovery_touchdown_step_distance_cm`：上述恢复 touchdown 的平均脚 XY 距离。

`recovery_signal_mean` 高而 `recovery_gate_mean` 很低在课程早期是预期现象，必须结合
`Curriculum/recovery_gate_scale` 解读。恢复期 liftoff 增加只说明开始卸载/抬脚；同时出现
touchdown、非零落脚距离且 `fall_rate` 下降，才更接近有效恢复步。部署切换还要检查第一帧的
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
3. 切换请求只在 200Hz 主控制循环边界消费。
4. walk→stand 先由 walking policy 平滑减速，在参考双支撑窗口过零并用 IMU、腿 q/dq 连续
   确认稳定；切换后 standing torso 命令为 neutral。
5. stand→walk 先由 standing policy 将 torso 命令平滑回 neutral 并确认稳定，再切换模型；
   head 命令在两模型间保持连续，但按当前实际运行的模型使用各自限幅。
6. 两个策略使用同一套腿部 `10/0.3` 附录 B 低层模型。
7. `STAND_UP→RL_STAND` 先用实测躯干姿态初始化命令，再用 `0.5s` 余弦接管平滑回
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

OceanBDX 必须为 stand/walk 分别配置 head 限幅。新的完整站立限幅只能和本配置从头训练后导出
的 StandPaper ONNX 一起使用；旧站立 ONNX 应继续受旧范围限制，不能仅修改部署限幅强行外推。

## 当前验证基线

- 最新完整配置通过 64 env × 1 PPO iteration smoke test：`policy/critic/action=77/79/14`、
  `fall_rate=0`，无 traceback/NaN。首轮 `Metrics/recovery_signal_mean=0.9910`，但
  `Curriculum/recovery_gate_scale=0.0003`、有效 `Metrics/recovery_gate_mean=0.0002`；这证明
  随机初始策略虽产生很强失稳信号，课程仍成功阻止它在训练起点大幅关闭稳站约束。
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
- 新配置的完整从头训练和侧推捕获步效果尚待后续 checkpoint 验收；smoke test 只验证运行时、
  shape、数值稳定性和门控课程生效。
