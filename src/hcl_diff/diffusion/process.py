from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, offset: float) -> torch.Tensor:
    """Cosine noise schedule from Nichol and Dhariwal."""
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    values = torch.cos(((steps / timesteps) + offset) / (1 + offset) * math.pi / 2) ** 2
    values = values / values[0]
    betas = 1 - values[1:] / values[:-1]
    return betas.clamp(1e-8, 0.999).float()


class GaussianDiffusion:
    def __init__(self, betas: torch.Tensor):
        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_previous = torch.cat(
            [torch.ones(1, dtype=self.alpha_cumprod.dtype), self.alpha_cumprod[:-1]]
        )
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    @property
    def timesteps(self) -> int:
        return int(self.betas.numel())

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        for name in (
            "betas",
            "alphas",
            "alpha_cumprod",
            "alpha_cumprod_previous",
            "sqrt_alpha_cumprod",
            "sqrt_one_minus_alpha_cumprod",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_cumprod, timesteps, clean.shape) * clean
            + self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, clean.shape) * noise
        )

    def predict_clean(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(self.alpha_cumprod, timesteps, noisy.shape)
        return (noisy - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)
