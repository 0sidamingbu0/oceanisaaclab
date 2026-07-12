# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SquashedGaussianDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Configuration for the bounded action distribution used by walking."""

    class_name: str = (
        "oceanisaaclab.tasks.direct.oceanisaaclab.agents.squashed_gaussian:"
        "SquashedGaussianDistribution"
    )
    init_std: float = 0.3
    min_std: float = 0.03
    max_std: float = 0.6


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
        distribution_cfg=SquashedGaussianDistributionCfg(),
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
    - PPO：24 steps/env、mini-batch 4、epoch 5、clip 0.2、γ 0.99、
      GAE λ 0.95、自适应学习率（目标 KL 0.01）、梯度范数 1.0；
      硬件适配课程使用很小的 entropy 0.001，避免头命令/扰动放开前探索方差过早收缩；
    - 非对称 critic：critic 观测组用环境返回的 "critic"（无噪声观测 + 摩擦/质量
      随机化系数特权信息）。
    - 平面预训练使用论文配置 8192×24；粗糙地形微调由 WalkRoughPPORunnerCfg
      使用 2048×96，两者每次 update 都是 196,608 samples。
    """

    num_steps_per_env = 24
    max_iterations = 100000
    experiment_name = "bdx_walk_imitation"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=SquashedGaussianDistributionCfg(),
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
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WalkRoughPPORunnerCfg(WalkPPORunnerCfg):
    """Rough-terrain fine-tuning with the same total PPO batch as flat training."""

    num_steps_per_env = 96


@configclass
class StandPPORunnerCfg(WalkPPORunnerCfg):
    """路线 B（BDX 论文复刻）站立（perpetual）runner。

    网络/算法沿用 WalkPPORunnerCfg（3×512 ELU + 非对称 critic + obs_groups），维度由环境
    观测/状态空间自动推断，无需改。仅换日志目录名。站立任务不存在"站着不动=最优"的行走
    局部最优（本来就要站稳），entropy_coef 调回论文原值 0（不需要额外探索去破盆地）。
    """

    experiment_name = "bdx_stand_perpetual"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
