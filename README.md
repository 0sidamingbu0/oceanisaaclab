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
| `Ocean-BDX-Stand-Direct-v0` | 路线 A | 手工塑形奖励（速度跟踪 + 步态相位 + 稳定性等 20+ 项） | `logs/rsl_rl/bdx_walk_phase/` |
| `Ocean-BDX-Walk-Direct-v0` | 路线 B | BDX 式参考步态模仿（关节角/接触/姿态匹配为主） | `logs/rsl_rl/bdx_walk_imitation/` |

观测 41 维（两条路线一致，sim2sim 链路通用）。本项目不装足底接触开关，观测里不含双足接触量。

### 路线 B：参考步态模仿 `Ocean-BDX-Walk-Direct-v0`

- 训练（从头训，观测/奖励语义与旧 checkpoint 不兼容，不 resume）：

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


