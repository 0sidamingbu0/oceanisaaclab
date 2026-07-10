"""Bounded Gaussian action distribution for RSL-RL policies."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.distribution import Distribution


class SquashedGaussianDistribution(Distribution):
    """State-independent Gaussian followed by ``tanh``.

    PPO evaluates the likelihood of the exact bounded action sent to the
    environment. This avoids the optimization mismatch caused by sampling an
    unbounded Gaussian and clipping it later in the environment.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.3,
        min_std: float = 0.03,
        max_std: float = 0.6,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__(output_dim)
        if not 0.0 < min_std <= init_std <= max_std:
            raise ValueError("Expected 0 < min_std <= init_std <= max_std.")
        self.min_log_std = float(torch.log(torch.tensor(min_std)))
        self.max_log_std = float(torch.log(torch.tensor(max_std)))
        init_log_std = float(torch.log(torch.tensor(init_std)))
        init_fraction = (init_log_std - self.min_log_std) / (self.max_log_std - self.min_log_std)
        init_raw = torch.logit(torch.tensor(init_fraction))
        self.log_std_param = nn.Parameter(torch.full((output_dim,), init_raw))
        self.epsilon = epsilon
        self._normal: Normal | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        log_std = self.min_log_std + (
            self.max_log_std - self.min_log_std
        ) * torch.sigmoid(self.log_std_param)
        self._std = torch.exp(log_std).expand_as(mlp_output)
        self._mean = torch.tanh(mlp_output)
        self._normal = Normal(mlp_output, self._std)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self._normal.sample())  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return nn.Tanh()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._mean  # type: ignore[return-value]

    @property
    def std(self) -> torch.Tensor:
        return self._std  # type: ignore[return-value]

    @property
    def entropy(self) -> torch.Tensor:
        # The entropy coefficient is zero for the paper configuration. The base
        # entropy remains a stable diagnostic without Monte-Carlo noise.
        return self._normal.entropy().sum(dim=-1)  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._normal.mean, self._std)  # type: ignore[union-attr,return-value]

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        bounded = outputs.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        pre_tanh = torch.atanh(bounded)
        correction = torch.log(1.0 - bounded.square() + self.epsilon)
        return (self._normal.log_prob(pre_tanh) - correction).sum(dim=-1)  # type: ignore[union-attr]

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        ).sum(dim=-1)
