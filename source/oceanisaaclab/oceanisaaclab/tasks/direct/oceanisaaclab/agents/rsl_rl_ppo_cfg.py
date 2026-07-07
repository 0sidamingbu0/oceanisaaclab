# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "bdx_walk_phase"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WalkPPORunnerCfg(PPORunnerCfg):
    """路线 B（BDX 论文复刻）runner：网络/算法对齐论文附录 A 表 IV。

    - actor/critic 各 3×512 ELU（论文：three fully connected layers of 512 hidden
      units and ELU activations，critic 用同规模独立网络）；
    - PPO：24 steps/env、mini-batch 4、epoch 5、clip 0.2、entropy 0、γ 0.99、
      GAE λ 0.95、自适应学习率（目标 KL 0.01）、梯度范数 1.0；
    - 非对称 critic：critic 观测组用环境返回的 "critic"（无噪声观测 + 摩擦/质量
      随机化系数特权信息）。
    - batch 8192×24 需要以 --num_envs 8192 启动训练。
    """

    experiment_name = "bdx_walk_imitation"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 论文原值 0；但 07-07 复盘发现 5800 iter 时 mean_std 从 init 0.3 塌到 0.013、
        # entropy -29，在发现"迈步比站着值"之前探索就死了、锁死站立局部最优。
        # 加小 entropy_coef 恢复逃出盆地的能力（值取小以防重新引入不稳定）。
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
