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

旧 `74/76` 维 StandPaper checkpoint/ONNX 与当前接口形状不兼容。77 维模型保持接口兼容，
但头部命令范围和足端奖励属于训练分布/目标的一部分；修改后续训只能用于短期 A/B，正式模型
应从头训练并重新导出。

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

站立 head 命令使用 `neck_head_map.npz` 完整可达域的 `2/3`，比此前完整范围减小 `1/3`：

```text
dh     [-0.013333, +0.013333] m
pitch  [-0.333333, +0.333333] rad
yaw    [-0.666667, +0.666667] rad
roll   [-0.40, +0.40] rad
```

walking 使用完整域的 1/2：`±0.01m / ±0.25 / ±0.50 / ±0.30rad`。两套范围必须按当前
实际运行的 policy 分开限幅。

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

非零命令参考按论文 perpetual reference 的静态平衡目标联合使用 torso4/head4。为避免
`5^8` 表和 256 角点运行时插值，资产保存 625 个 neutral-head torso 平衡姿态、625 个 head
全身 CoM 偏移，以及每个 torso 节点的腿 IK/基座平移雅可比。当前资产标定值：总质量
`10.01488kg`，neutral CoM xy `[21.190,-0.274]mm`，collision sole 低层包围中心
`[20.342,-1.441]mm`；两者只差约 `1.44mm`，因此以 neutral CoM 为目标保留共享 q=0。

站立 reset 使用独立 RSI，不再执行 walking RSI：

- 标准参考始终来自 `stand_pose.npz`；`stand_recovery_resets.npz` 是独立 reset 库，不参与
  reward reference。窄脚距或错位状态因此必须恢复到标准站姿，不能被策略当作新的模仿目标。
- 默认约 50% episode 从标准无扰动状态出生，20% 从稳定但非标准的脚位无扰动出生，另外
  30% 使用完整 Table V 扰动。错误脚位 reset 强制 torso/head neutral。
- 站立 episode 为 20s，使接触状态下 10--30s 时间尺度的慢速脚位漂移能在单个 episode 暴露；
  PPO 每次 collection 仍为 24 control steps。
- reset 库含 52 个双脚贴地且 CoM 平衡的 IK 解：单脚内收、双脚对称内收、单脚外扩、前后错位
  和单脚 yaw。最大脚距缩小 50mm、加宽 40mm、前后错位 50mm、相对 yaw 0.08rad。
- 错误脚位幅度课程从 30% 开始，在 36000 control steps（1500 iteration）内放开到 100%。
- 标准非 neutral episode 仍由 `stand_rsi_prob=0.8` 决定是否从命令对应参考出生；其余从
  neutral 学习姿态过渡。
- episode 内的 torso/head 命令每 `4-8s` 再采样一次，使策略学习连续姿态切换。
- 物理 q、动作延迟环、`a_(t-1)`、`a_(t-2)`、腿/脖子 FOH 和低通目标全部由同一个
  可表示 setpoint 初始化；错误脚位 reset 还同步 root xy 和 path frame，避免第一步产生伪误差。

reset 库可重复生成：

```bash
python3 scripts/gen_stand_recovery_resets.py
```

站立 path frame 以 1 秒时间常数收敛到双脚中心和双脚平均 heading。左右脚末端 link frame
在 q=0 时相差约 pi，因此必须先减去参考资产中的各脚 yaw 偏置，再做圆均值；不能用躯干 yaw
替代双脚 heading。

## 奖励与扰动

站立以论文 Table I 为主体，并为本机严格锁脚目标增加两项足端约束：

| 项 | 核/权重 |
|---|---|
| Torso position xy | `exp(-200 * error^2)`, weight 1 |
| Torso orientation | `exp(-20 * error^2)`, weight 1 |
| Linear velocity xy/z | `exp(-8 * error^2)`, weight 1/1 |
| Angular velocity xy/z | `exp(-2 * error^2)`, weight 0.5/0.5 |
| Leg / neck joint position | negative L2, 15 / 100 |
| Leg / neck joint velocity | negative L2, 0.001 / 1 |
| Contact match | 1 per foot |
| Contact-foot planar speed | negative L1, 10 |
| Airborne foot | negative count, 4 per foot |
| All joint torques / accelerations | leg + neck negative L2, 0.001 / 2.5e-6 |
| Leg / neck action rate | negative L2, 1.5 / 5 |
| Leg / neck action acceleration | negative L2, 0.45 / 5 |
| Survival | 20 |

躯干高度命令通过站立腿关节参考进入模仿奖励，不再添加论文之外的独立 torso-height reward。
论文三档扰动从训练开始，在前 1500 iteration 线性放开。

为适配本机“无扰动时脚应锁定、大扰动时允许捕获步、捕获后回标准脚位”的验收目标，当前训练
将 episode 分为约 50% 标准无扰动、20% 错误脚位无扰动和 30% 完整 Table V 独立扰动。
扰动 episode 内的力/矩幅值、开关时长和课程不变；reset 类别和恢复完成状态不进入观测，
也不改变任何奖励权重。恢复完成判定只用于统计，要求脚距、前后错位、相对 yaw 和物理稳定性
连续满足门槛；它不指定抬哪只脚，也不直接发动作。

静态参考的 contact schedule 仍为双脚支撑，但论文的受扰实验允许策略偏离参考轨迹和接触序列
来恢复平衡。论文还明确指出，去掉腿姿态、腿速度或接触项会导致策略快速挪脚，因此当前实现
始终固定使用这些核心项，并额外保持锁脚项；恢复状态不调整奖励，也没有脚高、指定摆动脚或迈步奖励。
环境只用真机可获得的量计算恢复诊断状态：

| 失稳量 | 归一化起点 | 归一化满量程 |
|---|---:|---:|
| 相对命令 torso 姿态的 roll/pitch 倾斜误差 | `0.08rad` | `0.22rad` |
| body 水平速度 | `0.18m/s` | `0.45m/s` |
| body roll/pitch 角速度 | `0.45rad/s` | `1.40rad/s` |
| path-frame torso 平面位置误差 | `0.025m` | `0.065m` |

四路瞬时信号取最大值。倾斜量相对命令姿态计算，正常的 torso pitch/roll 表演命令不会被误判
为失衡；命令重采样后的 `0.35s` grace 只屏蔽瞬时 tilt 信号。Schmitt 状态在信号达到 `0.35`
时进入，低于 `0.10` 且双脚稳定 `0.30s` 后退出；该状态只用于统计，不参与奖励或动作。

不存在作用于奖励的 `recovery_gate` 或恢复课程。训练时应分别查看 quiet liftoff、恢复期
touchdown 落脚距离和 `fall_rate`：无扰动窗口的离地应接近 0；大推后若策略产生有效捕获步，
应出现至少约 `0.12s` 离地、`3cm` 以上落脚位移并重新双脚支撑。指标只用于区分接触抖动和有效迈步。

当前生成器用静态条件下 `CoP_xy = CoM_xy` 的等价关系做全身质量补偿，而不是输出运行中的
真实 CoP/ZMP 轨迹。完整 625 点 torso 网格的 neutral-head CoM 残差最大 `0.020mm`、脚 IK
残差最大 `0.886mm`；每个 torso 节点叠加 head 四维超立方体角点后，最大 CoM/脚锚点误差为
`0.080mm / 4.220mm`。精确零命令不做数值“纠偏”，以免破坏 stand/walk 共享姿态。

可视化保存后的联合参考：

```bash
python3 scripts/gen_stand_pose.py --viz-only --viz-loops 1 --viz-hold 0.03
```

Meshcat 会依次扫 torso4 和 head4；终端同时报告每帧 CoM xy 与脚锚点误差。

## 训练、回放与诊断

从头训练：

```bash
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Ocean-BDX-StandPaper-Direct-v0 \
  --num_envs 8192 \
  --max_iterations 100000 \
  --headless
```

Stand PPO 恢复论文附录 A 的 `entropy_coef=0` 和默认 squashed Gaussian 方差下限；扰动本身
提供恢复探索，不再人为维持较大的腿动作方差。

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

- `Metrics/recovery_signal_mean`：raw 失稳信号均值，仅用于诊断。
- `Metrics/recovery_active_state_mean`：Schmitt 恢复状态占比，仅用于诊断，不是奖励 gate。
- `Metrics/recovery_active_rate`：恢复状态占用的时间比例。
- `Metrics/recovery_single_support_rate`：恢复激活期间单脚支撑占比。
- `Metrics/liftoff_events_per_episode`：每个 episode 双脚合计离地事件数。
- `Metrics/recovery_liftoff_events_per_episode`：恢复激活时发生的离地事件数。
- `Metrics/touchdown_events_per_episode`：离地后重新接触地面的事件数。
- `Metrics/touchdown_step_distance_cm`：所有 touchdown 从对应 liftoff 到落脚的平均脚 XY 距离。
- `Metrics/recovery_touchdown_events_per_episode`：离地或落脚发生在恢复期的 touchdown 数。
- `Metrics/recovery_touchdown_step_distance_cm`：上述恢复 touchdown 的平均脚 XY 距离。
- `Metrics/recovery_capture_steps_per_episode`：满足最小离地时间和落脚距离的恢复步数。
- `Metrics/quiet_liftoff_events_per_episode`：恢复状态关闭时的离地事件数，稳定站立应接近 0。
- `Metrics/quiet_foot_drift_cm`：无恢复状态时连续接触脚的累计 XY 漂移。
- `Metrics/clean_episode_fraction`：当前统计窗口中无外力 episode 的比例，默认约 `0.50`。
- `Metrics/clean_liftoff_events_per_episode`：无外力 episode 的离地事件数，稳定策略应接近 `0`。
- `Metrics/clean_double_support_rate`：无外力 episode 的双支撑率，应接近 `1`。
- `Metrics/stance_width_error_cm`：双脚横向间距相对标准站姿的平均绝对误差。
- `Metrics/stance_stagger_error_cm`：双脚前后错位相对标准站姿的平均绝对误差。
- `Metrics/stance_yaw_error_rad`：双脚相对 yaw 相对标准站姿的平均绝对误差。
- `Metrics/displaced_reset_fraction`：当前完成 episode 中错误脚位 reset 的比例。
- `Metrics/displaced_recovery_success_rate`：错误脚位 episode 在结束前恢复标准脚位并稳定的比例。
- `Metrics/displaced_recovery_time_s`：成功样本从 reset 到稳定恢复的平均时间。
- `Metrics/displaced_liftoff_events_per_episode`：错误脚位 episode 的平均离地事件数。
- `Metrics/displaced_double_support_rate`：错误脚位 episode 的双支撑时间比例。
- `Curriculum/stand_displaced_reset_scale`：当前错误脚位位移课程幅度。

`recovery_signal_mean` 高只表示当前物理状态偏离参考，不会改变奖励。恢复期 liftoff 增加只说明
开始离地；同时出现满足阈值的 touchdown、非零落脚距离且 `fall_rate` 下降，才更接近有效恢复步。部署切换还要检查第一帧的
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
   同时只用腿编码器 FK 检查双脚相对脚距、前后错位和 yaw。脚位不合格时保持 standing neutral
   闭环继续恢复，超过等待时间也不强切、不关电机；恢复合格后自动放行。head 同 torso 一起回正。
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

- 代码修改后已通过 Python 编译和 `git diff --check`；64 env × 1 PPO iteration smoke test
  确认 `policy/critic/action=77/79/14`、首轮 `fall_rate=0`，无 traceback/NaN。
- 范围内非零 yaw RSI：参考重构误差约 `2.7e-7rad`，solved/Isaac 双脚 heading 差约
  `0.00176rad`。
- reset 后腿/脖子 target-q、动作延迟环和前两帧动作误差均为 0。
- stand/walk q=0 脚距差约 `2.1e-12m`，base height 与 `base_pos_pf` 完全一致。
- 联合站立参考的 neutral-head torso 网格脚残差最大 `0.886mm`；叠加 head 全域角点的
  factorized 雅可比近似最大 `4.220mm`，action-range/soft-limit 夹持为 0。
- 恢复 reset 库共 52 个稳定 IK 状态，最大脚误差 `1.274mm`、CoM 平衡误差 `0.019mm`；
  GPU 各课程档抽样验证通过。
- Python sim2sim 轻量测试通过 77/80 维观测、动作历史传递、控制周期边界、当前/下一双支撑、
  编码器 FK 脚位门槛和超时后持续等待恢复逻辑。
- 真实 MuJoCo 用当前 80 维 walking policy 作为 standing fallback 验证接管状态机：首个 RL
  控制步腿目标/力矩跳变为 `2.24e-5rad / 0.079Nm`，整个 `0.5s` 窗口最大单步变化为
  `0.00681rad / 0.175Nm`；10 秒末仍为双脚支撑的 `RL_STAND`。该测试不能替代重新训练后的
  77 维 standing policy 闭环验收。
- 新配置的完整从头 100000 iteration 训练、侧推捕获步和回标准脚位效果尚待后续 checkpoint
  验收；smoke test 只验证运行时、shape、数值稳定性和 reset 课程生效。
