# Template for Isaac Lab Projects

## Overview

这是复刻迪士尼BDX机器人的行走算法训练工程。基于ISAACSIM6和ISAACLAB 3.0.2


## Python 运行约定（重要）

本工程的所有 Python 脚本都必须用 Isaac Lab 自带的解释器运行，而**不是系统 `python`**：

```bash
./_isaaclab/isaaclab.sh -p <脚本> [参数]
```

系统 `python` 缺少 `tensorboard`、`torch`、`isaaclab` 等依赖，直接跑会 `ModuleNotFoundError`。
例如离线解析训练曲线：

```bash
./_isaaclab/isaaclab.sh -p scripts/analyze_tb.py logs/rsl_rl/<task>/<run>
```

## 本工程任务与常用命令（Ocean BDX）

本工程有两个行走任务，分别对应两条技术路线：

| 任务 id | 路线 | 范式 | 训练日志目录 |
|---|---|---|---|
| `Ocean-BDX-Stand-Direct-v0` | 路线 A | 手工塑形奖励（速度跟踪 + 步态相位 + 稳定性等 20+ 项），观测 41 维 | `logs/rsl_rl/bdx_walk_phase/` |
| `Ocean-BDX-Walk-Direct-v0` | 路线 B（行走预训练） | 论文 periodic 行走策略，平面 `8192×24` 高吞吐预训练，观测 80 维、动作 14 | `logs/rsl_rl/bdx_walk_imitation/` |
| `Ocean-BDX-WalkRough-Direct-v0` | 路线 B（粗糙/坡面微调） | 与行走预训练相同接口，40% 平面 + 30% `±12mm` 连续粗糙面 + 15% 上坡 + 15% 下坡（约 `0–5°`），`2048×96` | `logs/rsl_rl/bdx_walk_imitation/` |
| `Ocean-BDX-StandPaper-Direct-v0` | 路线 B（站立） | **BDX 论文 perpetual 站立策略**（无相位，命令 g_perp=躯干4+头部4），policy/critic `77/79` 维、动作 14 | `logs/rsl_rl/bdx_stand_perpetual/` |

本项目不装足底接触开关，观测里不含双足接触量（接触只用于奖励）。

路线 B 按论文 divide-and-conquer 拆成**两个独立模型**：periodic 行走 + perpetual 站立，运行时按需切换（论文图 9）。行走 policy/critic 为 `80/82` 维，站立为 `77/79` 维，动作都为 14 维。旧行走 77 维和旧站立 `74/76` 维 checkpoint/ONNX 均不兼容。站立必须从头重训；本次还把脖子目标纳入与腿一致的 FOH + 37.5Hz 低通，因此现有 80 维行走模型至少要完整回归，推荐重新训练或微调后再导出。

### 路线 B（行走）：BDX 论文复刻 `Ocean-BDX-Walk-Direct-v0`

对照迪士尼论文《Design and Control of a Bipedal Robotic Character》（工程根目录
`BD_X_paper.pdf`）的 periodic walking policy，核心机制：

- **path frame**（论文 V-A / Fig.4）：行走按命令速度积分推进、站立收敛到双脚中心、
  最大偏差投影拉回；速度跟踪由「躯干贴住 path 系参考位置」隐式实现，无显式速度跟踪奖励。
- **观测 80 维**（论文式 (8) + 附录 A）：
  path 系躯干 xy(2) + path 系 yaw sin/cos(2) + projected gravity(3)
  + body 系线速度(3) + body 系角速度(3)
  + 腿 q(10) + 脖子 q(4) + 腿 q̇(10) + 脖子 q̇(4) + a_{t-1}(14) + a_{t-2}(14)
  + 相位二阶谐波 sin/cos(2πφ),sin/cos(4πφ)(4) + 命令 vx,vy,wz(3) + 头部命令(4)。
  非对称 critic 额外收无噪声观测 + 摩擦/质量随机化系数（82 维）。
- **动作 14 维**：10 腿（力矩直驱）+ 4 脖子（位置伺服，跟随头部命令参考角）。
- **头部命令 4-DOF**（论文 g_peri 的 Δh_head/Δθ_head）：Δh 头高 / pitch 点头 / yaw 摇头 /
  roll 歪头。命令→脖子参考角由 `neck_head_map.npz` 四线性插值提供（脚与脖子 IK 解耦，
  头部命令不进步态库网格）。
- **奖励 = 论文表 I**：躯干 path 系位置/朝向、body 系线/角速度 exp 核 + 腿关节角(-25·L2)/
  角速度 + 接触匹配 + 脖子关节角模仿(-100·L2，跟头部命令参考角)/脖子角速度 + 力矩/关节
  加速度/腿·脖子动作率/动作加速度正则 + 存活，权重×step_dt。
- **动作管线**（论文 V-C/V-D + 附录 A）：50Hz 策略 → 逐关节线性映射（腿：0=标称站姿，
  1=每关节预期范围 `action_joint_ranges`；脖子：0=默认位，±1=`neck_action_joint_ranges`）→
  腿围绕实测关节角 ±τmax/kP 限幅 → 14 维目标统一做一阶保持插值 + 37.5Hz 低通 →
  200Hz 附录 B 执行器模型（脖子滤波后进入位置伺服通路）。
- **执行器模型**（附录 B / 表 VI，Unitree A1/Go1 辨识参数；髋 roll/pitch/膝=A1、
  髋 yaw/踝=Go1；**本机电机若非同款需重新辨识**）：软件 PD + 编码器偏移 ±0.02rad +
  tanh 摩擦 + 速度相关力矩限幅 + 背隙/速度相关噪声编码器读数 + 反射惯量 ±20%，
  编码器/背隙/PD 增益每 episode 重采样；反射惯量在每个并行环境初始化时随机一次，避免
  高频跌倒 reset 时反复写 PhysX articulation 属性。
- **扰动 = 论文表 V** 三档独立进程（髋/脚短小、盆骨长小、盆骨短大推力；大推力按整机
  质量 ≈10/15.4 缩放为 58~97N），从训练开始、前 1500 iter 线性放满。有限 episode
  reset 时从扰动周期随机相位进入，避免 `12-15s` 大推力静默期被 8 秒 episode 永久截断。
- **相位速率 φ̇ 按命令从参考库插值**逐步积分，周期从零速 0.6s 连续缩短到满速
  0.48s；接触占空比 0.6，对应约 20% 双支撑窗口。
- PPO/网络对齐论文表 IV：actor/critic 各 3×512 ELU、epoch 5、entropy 0、自适应
  lr（KL 0.01）。平面预训练恢复论文 `8192 env × 24 steps`；策略稳定后切到粗糙地形任务，
  用 `2048 env × 96 steps` 控制接触峰值。两阶段每次 update 都是 196,608 个样本。
  论文最终每个策略训练 100,000 iterations；1,500 iter 只用于预览 nominal behavior。

- 第一阶段：平面高速预训练（从头训，旧 77 维 checkpoint 不兼容）：

    ```bash
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
      --task Ocean-BDX-Walk-Direct-v0 --num_envs 8192 --max_iterations 100000 --headless
    ```

  `--max_iterations` 在 RSL-RL 的 `--resume` 模式下表示“额外训练多少 iteration”，不是
  目标总 iteration。若要把 `model_N.pt` 严格续训到论文的总计 100,000 iter，应传
  `--max_iterations $((100000-N))`；新扰动课程与旧 checkpoint 的训练分布不同，严格复现
  优先从头训练。

- 第二阶段：当 `fall_rate` 已明显下降、步态成形后，从平面 checkpoint 进入粗糙/坡面微调。
  坡面从中心安全平台向外连续延伸，最大约 5°；单点地面射线只修正坡面上的终止高度，
  不进入策略观测，因此平面 checkpoint 的 80 维接口保持兼容：

    ```bash
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
      --task Ocean-BDX-WalkRough-Direct-v0 --num_envs 2048 --max_iterations 100000 \
      --resume --load_run <平面run目录名> --checkpoint model_<iter>.pt --headless
    ```

- 回放训练好的策略（play）：

    ```bash
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
      --task Ocean-BDX-Walk-Direct-v0 --num_envs 16 --viz kit \
      --checkpoint logs/rsl_rl/bdx_walk_imitation/<run>/model_<iter>.pt
    ```

### 路线 B（站立）：BDX 论文 perpetual 站立策略 `Ocean-BDX-StandPaper-Direct-v0`

论文 divide-and-conquer 的第二个独立策略 π(a | s, g_perp)：**无相位**，静态参考保持双脚
支撑，但受扰恢复时允许偏离参考接触序列并主动迈步。它专管「摆站姿 + 转头」，与行走策略
共用机器人模型与整套 sim2real 机制（附录 B 执行器模型、
表 V 扰动、域随机化、torso 等效触地终止、非对称 critic、path frame、脖子位置伺服）。

- **命令 g_perp（8 维）**：躯干 4-DOF（h 高度 / pitch 前后倾 / yaw 偏航 / roll 侧倾）
  + 头部 4-DOF（Δh / pitch / yaw / roll，复用 `neck_head_map`）。本机双脚约束 IK 扫描后的
  torso 可行域为 h `[-0.04,+0.01]m`、pitch `±0.17rad`、yaw `±0.24rad`、roll
  `±0.09rad`。StandPaper 的 head 域为完整映射的 2/3：Δh `±0.013333m`、pitch
  `±0.333333rad`、yaw `±0.666667rad`、roll `±0.40rad`；walking 使用完整域的 1/2
  (`±0.01m / ±0.25 / ±0.50 / ±0.30rad`)。命令在 episode 内每 `4-8s` 重采样，覆盖论文
  要求的连续控制输入。
- **观测 77 维**：相比行走去掉相位二阶谐波(4)、命令由 vx/vy/wz(3) 换成躯干命令(4)
  （净 −3）；保留 path 系 xy/yaw、`projected_gravity`、body 线/角速度、腿+脖子 q/q̇、
  两帧 14 维动作历史。非对称 critic 再加摩擦/质量特权量，共 79 维。
- **共享零速姿态**：`stand_pose.npz` 的精确零命令、行走参考库零速帧和 URDF q=0 使用同一
  腿角、base height 与双脚锚点：`q_leg=0`、`base_height=0.38498640060424805m`、
  `base_pos_pf=[0.04638837,-0.00091510]m`。环境启动时会检查三者一致，避免两策略零速脚距
  或躯干参考不同。联合参考的 neutral-head torso 网格脚残差最大约 `0.886mm`，叠加完整 head
  角点的 factorized 近似最大约 `4.220mm`；动作映射和软限位夹持数为 0。
- **奖励 = 论文 Table I 主体 + 锁脚约束**：在躯干、关节、接触、正则和 survival 项之外，
  接触脚水平速度按权重 `-10` 惩罚，离地脚按每只 `-4` 惩罚。加上失去的接触奖励，单脚离地
  边际代价为 `5 reward/s`；该代价有限，大扰动下避免摔倒仍可抵消一次短捕获步。
- **受扰恢复（仅诊断）**：论文明确指出，去掉腿姿态、腿速度或接触约束会导致策略快速挪脚，
  因此恢复状态不缩放 contact、leg imitation、锁脚或动作平滑权重。环境只用
  真机可得的 IMU、状态估计速度、path-frame 位置和脚接触计算 Schmitt 诊断状态：失稳信号达到
  `0.35` 才标记恢复，低于 `0.10` 且双脚稳定 `0.30s` 后退出。该状态不进入奖励、不改变动作，
  也没有脚高、指定左右摆动脚或迈步奖励；捕获步仍由 Table V 大扰动下的存活与躯干模仿收益自行学出。
- **站立 RSI 与恢复 reset**：标准站立命令继续使用 `stand_pose.npz`；独立的
  `stand_recovery_resets.npz` 只提供双脚贴地、CoM 平衡但脚距缩小/加宽、前后错位或相对 yaw
  偏移的出生状态。错误脚位绝不成为奖励参考，策略仍由固定 Table I 收益将其校正回 q=0 标准站姿。
  q、root、path frame、FOH/LPF 目标、延迟环和前两帧动作均从同一 reset 状态同步初始化。
- **训练样本分层**：默认约 50% 标准无扰动、20% 错误脚位无扰动、30% 完整 Table V 扰动。
  错误脚位幅度从 30% 开始，在前 1500 iteration 放开到 100%；分布标签不进入观测且不改变奖励。
  站立 episode 为 20s，以覆盖接触状态下 10--30s 的慢速脚位漂移；当前库覆盖最大 50mm 脚距
  缩小、40mm 加宽、50mm 前后错位和 0.08rad 相对 yaw。
- **path frame**：站立分支收敛到双脚中心和消除左右 link-frame 偏置后的双脚平均 heading，
  不是收敛到躯干 yaw；行走和站立使用同一份 path-frame 状态。
- **动作与频率**：14 维动作与行走完全同构，50Hz policy / 200Hz low-level，腿和脖子目标都
  经过 FOH + 37.5Hz 低通。Stand PPO 恢复论文附录 A 的 `entropy_coef=0` 和默认动作方差下限；
  扰动本身提供恢复探索，避免长期随机腿抖动。

    ```bash
    # 训练（与行走各训各的，运行时按需切换策略）
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
      --task Ocean-BDX-StandPaper-Direct-v0 --num_envs 8192 --max_iterations 100000 --headless
    # 回放
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
      --task Ocean-BDX-StandPaper-Direct-v0 --num_envs 16 --viz kit \
      --checkpoint logs/rsl_rl/bdx_stand_perpetual/<run>/model_<iter>.pt
    ```

旧 `74/76` 维站立 checkpoint 和 ONNX 在形状上不兼容；早期 77 维模型也不能续训，因为旧版
恢复门控曾把 contact/腿姿态约束长期关闭，已学成小碎步局部解。必须从头训练并重新导出，再同步替换 OceanBDX 的
`policy/stand/policy.onnx[.data]`。采用 Table I 固定奖励和错误脚位 reset 的兼容 checkpoint 可在本次
样本比例与 episode 时长调整后续训；建议先做 5000--10000 iteration A/B，再决定是否长期续训。
训练时重点监控 `fall_rate`、clean 锁脚、错误脚位恢复成功率/恢复时间、标准脚距误差和受扰捕获步。完整接口与迁移检查见
[`BDX_STAND_TRAINING.md`](source/oceanisaaclab/docs/BDX_STAND_TRAINING.md)。

运行时切换必须遵循论文 Fig.9：两策略共享最近两帧**实际归一化动作**、path frame 以及腿/脖子
FOH/LPF 当前目标。walk→stand 先由 walking policy 平滑减速并在参考双支撑窗口过零，连续确认
IMU 与腿 q/dq 稳定后才切换，进入 standing 后使用 neutral torso 命令；stand→walk 先由
standing policy 将 torso 平滑回 neutral 并确认稳定，再在控制周期边界切换。
OceanBDX 还会用腿编码器 FK 检查双脚相对脚距、前后错位和 yaw；错误脚位未恢复时继续由站立
策略 neutral 闭环校正，恢复到门槛内后自动放行，不会超时强切或关闭关节输出。
Python sim2sim 的 path frame 使用脚 link body origin，而非 sole geom center；两者在 q=0 下会
造成约 `2.585cm` 偏差。起立脚本接管 RL 时另用 `0.5s` 余弦增益过渡，把腿部 `50/3` 平滑降到
论文 Go1 plant 的 `10/0.3`，并以同一权重从实测 torso 命令起点平滑回用户/neutral 命令、
渐入腿/脖子策略目标。不能在策略切换时清零动作历史或把滤波目标重置为实测 q。

### 路线 B 部署（sim2sim / sim2real）：命令如何控制前后左右

常见误解澄清：**参考库只在训练时用**（算模仿奖励 + RSI 重置），参考轨迹**从来不进观测**，
部署侧**不需要**「按 cmd 修改观测里的参考轨迹」。速度命令 (vx, vy, wz) 在 80 维观测的
`[73:76]`，头部命令在 `[76:80]`。sim2sim 已同步到 80 维/14 动作。

速度命令通过三条通道同时作用于策略，部署程序必须全部复刻（头部命令另见下方第 4 条）：

1. **观测 [73:76] = cmd × command_scale**：遥控器 / 上层直接写入。命令为头部系
   （vx 头前向 +、vy 头左向 +、wz 逆时针 +，单位 m/s、rad/s），训练范围
   vx∈[-0.25,0.25]、vy∈[-0.15,0.15]、wz∈[-0.8,0.8]，部署侧应同样限幅。
2. **path frame 按 cmd 积分** → 观测第 1~4 维（path 系躯干 xy×`pos_pf_scale=4.0`、
   相对 yaw 的 sin/cos）。逐行为复刻 `path_frame.py`（纯 torch，可直接翻 numpy）：
   - `|cmd|` 任一分量 > `move_command_threshold=0.08` → 行走分支：path frame 位置
     按命令速度（先旋到世界系）× dt 积分，yaw 按 wz × dt 积分；
   - 否则站立分支：一阶低通（时间常数 1.0s）收敛到**双脚中心 xy** 与校准左右脚 link-frame
     固有 yaw 偏置后的**双脚平均 heading**（需要里程计/状态估计和双脚 FK）；
   - 每步最后做最大偏差投影：位置偏差钳到 0.25m、yaw 偏差钳到 0.6rad。
   - dt = 策略周期 0.02s（50Hz），与训练一致。
   速度跟踪就是这样隐式实现的：path frame 以命令速度跑，策略只有贴住它才有奖励，
   所以推 cmd_vx>0 就是把参考锚点向头前方移动，策略自然向前走。
3. **相位 φ 按 φ̇(cmd) 积分** → 观测相位 4 维（sin/cos(2πφ), sin/cos(4πφ)）。
   φ̇ 从 npz 库的 `phase_rate` 表按 cmd 三线性插值（复刻 `reference_gait.py` 的
   `sample_phase_rate`，网格 vx 5 点 / vy 3 点 / wz 5 点；恒定步频库退化为常数
   1/gait_period）。每策略步 φ ← (φ + φ̇·0.02) mod 1。当
   `max(|vx|,|vy|,|wz|) <= move_command_threshold(0.08)` 时，φ̇=0 并冻结当前相位；
   零命令的腿、躯干和脖子参考均为静态，不得继续向策略输入变化的步态时钟。

行走时脖子周期参考与腿使用同一个 φ 和同一个 φ̇(cmd)。完整左右步态周期内
有两次迈步，因此脖子起伏使用二阶谐波，每只脚迈步时各起伏一次。默认周期运动只让
`neck_n1/n2` 同向小幅运动，形成约 0.7mm 的头部上下起伏；`neck_n3/n4` 对应 yaw/roll，
零头命令时不得周期摆动。命令速度只改变 φ̇，不改变脖子与脚步的相位对齐关系。

当前参考摆动脚峰值离地间隙为 5cm，左右脚使用严格对称的接触时序和镜像关节轨迹。
该高度在论文的动画参考生成框架内作为硬件适配参数，为 IsaacLab 到 MuJoCo/真机的
跟踪误差保留离地裕量。脖子跟踪和动作平滑权重使用论文 Table I 的 `100/1/5/5`。

此外，**头部命令 (Δh, pitch, yaw, roll)** 走独立通道：观测 `[76:80] = head_cmd ×`
`head_command_scale`（遥控器/上层写入），同时脖子 4 关节由策略动作后 4 维直接位置伺服
控制（训练时脖子模仿 `neck_head_map` 的头部命令→脖子角，部署侧脖子跟着策略输出走即可，
**不需要**在部署时查 `neck_head_map`）。头部命令不驱动 path frame（只影响脖子/头姿）。

部署侧需要从 `reference_gait.npz` 读取的只有 `phase_rate` + 三根网格轴
（`vx_grid/vy_grid/wz_grid`）；`joint_pos` 等轨迹表、`neck_head_map.npz`、`stand_pose.npz`
都**不用**在行走部署时加载（它们只在训练/生成阶段用）。sim2sim 参考实现见 oceanbdx 仓库
`sim2sim/mujoco_sim.py`（已同步 80 维 / 14 动作 / 头部命令键盘控制）。

### 路线 A：手工塑形 `Ocean-BDX-Stand-Direct-v0`

```bash
# 训练
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Ocean-BDX-Stand-Direct-v0 --num_envs 8192 --max_iterations 20000 --headless
# 回放
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Ocean-BDX-Stand-Direct-v0 --num_envs 16 --viz kit --checkpoint <run>/model_<iter>.pt
```

### 参考库工具（路线 B 用）

三个离线参考库都用 placo 生成，**必须用系统 `python3`，不是 `isaaclab.sh`**——依赖
placo/meshcat，与 Isaac 自带 python 的 boost 冲突（详见 changelog 2026-07-03）。可视化都用
meshcat（打开日志里打印的 `http://127.0.0.1:7000/static/` 观看）。

- **行走步态库** `reference_gait.npz`（腿部周期步态，按 vx/vy/wz 命令）：

    ```bash
    python3 scripts/gen_reference_gait.py            # 生成 assets/gaits/reference_gait.npz
    python3 scripts/play_reference_gait.py --vx 0.2  # meshcat 看「库里存的参考」运动学回放
    ```

- **脖子/头部映射库** `neck_head_map.npz`（头部命令 4-DOF：Δh 头高 / pitch 点头 / yaw 摇头 /
  roll 歪头 → 4 个脖子关节参考角。脖子运动学单独求解；站立库会进一步使用其全身 CoM
  偏移生成腿部平衡补偿）：

    ```bash
    python3 scripts/gen_neck_head_map.py             # 生成 assets/gaits/neck_head_map.npz
    python3 scripts/gen_neck_head_map.py --viz-only  # meshcat 逐轴扫掠(腿固定站姿,只动脖子)
    python3 scripts/gen_neck_head_map.py --viz       # 生成 + 扫掠可视化
    ```

- **站立姿态库** `stand_pose.npz`（站立策略用；同时输入 torso4 + head4，使用 625 个 torso
  静态平衡节点、625 个 head 全身 CoM 偏移节点和局部 IK 雅可比生成 10 个腿关节角及
  path-frame 躯干参考。双脚固定地面；精确零命令仍等于行走零速 q=0 公共切换姿态）：

    ```bash
    python3 scripts/gen_stand_pose.py                # 生成 assets/gaits/stand_pose.npz
    python3 scripts/gen_stand_pose.py --viz-only     # 播放已保存资产，逐轴扫掠 torso4 + head4
    python3 scripts/gen_stand_pose.py --viz          # 生成 + 扫掠可视化
    ```

  可视化可选参数：`--viz-loops N`（扫掠遍数）、`--viz-hold 秒`（每帧停留）。

- 参考步态动力学体检（**用 `isaaclab.sh`**，把参考角当动作喂进物理开环回放，验证「参考喂进
  物理能不能走」，与上面的运动学回放互补）：

    ```bash
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx 0.2            # 纯统计
    ./_isaaclab/isaaclab.sh -p scripts/diag_reference_playback.py --vx 0.2 --viz kit  # 开 GUI 观察
    ```

## Changelog / 修改记录

每天的修改记录统一放在 [`changelog/`](changelog/) 文件夹中，按日期命名（`YYYY-MM-DD.md`，如 [`changelog/2026-06-25.md`](changelog/2026-06-25.md)）。

目的：记录每天改了什么、为什么改、改动效果如何，便于理解工程的历史上下文，也方便其他工程师快速了解本工程的演进履历。

每个日志建议包含：改动背景与动机、涉及文件、具体改动、验证方式、训练/实验观察、待办与建议。
