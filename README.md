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
| `Ocean-BDX-Walk-Direct-v0` | 路线 B | **BDX 论文完整复刻**（path frame + 表 I 模仿奖励 + 附录 B 执行器模型），观测 57 维 | `logs/rsl_rl/bdx_walk_imitation/` |

本项目不装足底接触开关，观测里不含双足接触量（接触只用于奖励）。脖子固定（高刚度锁死默认位），只训 10 个腿部电机。

### 路线 B：BDX 论文复刻 `Ocean-BDX-Walk-Direct-v0`

对照迪士尼论文《Design and Control of a Bipedal Robotic Character》（工程根目录
`BD_X_paper.pdf`）的 periodic walking policy，核心机制：

- **path frame**（论文 V-A / Fig.4）：行走按命令速度积分推进、站立收敛到双脚中心、
  最大偏差投影拉回；速度跟踪由「躯干贴住 path 系参考位置」隐式实现，无显式速度跟踪奖励。
- **观测 57 维**（论文式 (8) + 附录 A）：path 系躯干 xy(2) + path 系 yaw sin/cos(2)
  + body 系线速度(3) + body 系角速度(3) + q(10) + q̇(10) + a_{t-1}(10) + a_{t-2}(10)
  + 相位二阶谐波 sin/cos(2πφ),sin/cos(4πφ)(4) + 命令 vx,vy,wz(3)。
  非对称 critic 额外收无噪声观测 + 摩擦/质量随机化系数（59 维）。
- **奖励 = 论文表 I 腿部子集**（脖子固定，neck 项删除）：躯干 path 系位置/朝向、
  body 系线/角速度 exp 核 + 腿关节角(-15·L2)/角速度 + 接触匹配 + 力矩/关节加速度/
  动作率/动作加速度正则 + 存活 20，权重×step_dt。
- **动作管线**（论文 V-C/V-D + 附录 A）：50Hz 策略 → 逐关节线性映射（0=标称站姿，
  1=每关节预期范围 `action_joint_ranges`）→ 围绕实测关节角 ±τmax/kP 限幅 →
  一阶保持插值 + 37.5Hz 低通 → 200Hz 附录 B 执行器模型。
- **执行器模型**（附录 B / 表 VI，Unitree A1/Go1 辨识参数；髋 roll/pitch/膝=A1、
  髋 yaw/踝=Go1；**本机电机若非同款需重新辨识**）：软件 PD + 编码器偏移 ±0.02rad +
  tanh 摩擦 + 速度相关力矩限幅 + 背隙/速度相关噪声编码器读数 + 反射惯量 ±20%，
  全部每 episode 重采样。
- **扰动 = 论文表 V** 三档独立进程（髋/脚短小、盆骨长小、盆骨短大推力；大推力按整机
  质量 ≈10/15.4 缩放为 58~97N），前 1500 iter 线性课程。
- **相位速率 φ̇ 按命令从参考库插值**逐步积分（库默认恒定步频；`gen_reference_gait.py
  --gait-period-fast` 可生成速度相关步频库）。
- PPO/网络对齐论文表 IV：actor/critic 各 3×512 ELU、epoch 5、entropy 0、自适应
  lr（KL 0.01）、batch 8192 env × 24 steps。

- 训练（从头训，观测/奖励/动作语义与旧 checkpoint 全部不兼容，不 resume）：

    ```bash
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
      --task Ocean-BDX-Walk-Direct-v0 --num_envs 8192 --max_iterations 20000 --headless
    ```

- 回放训练好的策略（play）：

    ```bash
    ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
      --task Ocean-BDX-Walk-Direct-v0 --num_envs 16 --viz kit \
      --checkpoint logs/rsl_rl/bdx_walk_imitation/<run>/model_<iter>.pt
    ```

### 路线 A：手工塑形 `Ocean-BDX-Stand-Direct-v0`

```bash
# 训练
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Ocean-BDX-Stand-Direct-v0 --num_envs 8192 --max_iterations 20000 --headless
# 回放
./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Ocean-BDX-Stand-Direct-v0 --num_envs 16 --viz kit --checkpoint <run>/model_<iter>.pt
```

### 参考步态库工具（路线 B 用）

- 生成 / 可视化参考步态库（**用系统 `python3`，不是 `isaaclab.sh`**——依赖 placo/meshcat，
  与 Isaac 自带 python 的 boost 冲突，详见 changelog 2026-07-03）：

    ```bash
    python3 scripts/gen_reference_gait.py            # 生成 assets/gaits/reference_gait.npz
    python3 scripts/play_reference_gait.py --vx 0.2  # meshcat 看「库里存的参考」运动学回放
    ```

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


