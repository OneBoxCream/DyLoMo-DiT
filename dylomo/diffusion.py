"""DDPM ancestral sampling for the class-conditional models (the DiT evaluation protocol).

The models are trained with the 1000-step linear-beta DDPM objective, predicting the noise and a variance that
interpolates in the log domain between the posterior variance and beta (Nichol & Dhariwal, 2021). Sampling runs on a
respaced grid (250 steps by default) with classifier-free guidance.

The respacing rule and the variance parameterisation are those of OpenAI's improved-diffusion / guided-diffusion
(MIT licence), which the DiT protocol also uses; this file is a compact implementation of the sampling path only.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def linear_betas(num_timesteps: int = 1000) -> np.ndarray:
    scale = 1000 / num_timesteps
    return np.linspace(scale * 0.0001, scale * 0.02, num_timesteps, dtype=np.float64)


def spaced_timesteps(num_timesteps: int, steps: int) -> List[int]:
    """``steps`` timesteps spread evenly over ``0 .. num_timesteps - 1`` (both ends included), ascending."""
    if not 1 <= steps <= num_timesteps:
        raise ValueError(f"cannot take {steps} steps from {num_timesteps} timesteps")
    stride = 1 if steps <= 1 else (num_timesteps - 1) / (steps - 1)
    position, taken = 0.0, []
    for _ in range(steps):
        taken.append(round(position))
        position += stride
    return sorted(set(taken))


class SpacedDDPM:
    """Coefficients of the reverse process on a respaced grid.

    ``timestep_map[i]`` is the training timestep (0 .. 999) of grid index ``i``; it is what the network receives and
    what the pass schedule of the model is evaluated on.
    """

    def __init__(self, steps: int = 250, num_train_timesteps: int = 1000):
        base_alphas_cumprod = np.cumprod(1.0 - linear_betas(num_train_timesteps), axis=0)
        use = set(spaced_timesteps(num_train_timesteps, steps))
        betas, self.timestep_map, last = [], [], 1.0
        for timestep, alpha_cumprod in enumerate(base_alphas_cumprod):
            if timestep in use:
                betas.append(1 - alpha_cumprod / last)
                last = alpha_cumprod
                self.timestep_map.append(timestep)
        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.betas = betas
        self.num_steps = len(betas)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        # the posterior variance is 0 at the first grid index; its log is clipped to the value of the next one
        self.posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.log_betas = np.log(betas)

    def step_statistics(self, model_output: torch.Tensor, x: torch.Tensor, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance of ``p(x_{i-1} | x_i)`` from the network output ``[B, 2C, H, W]`` at grid index ``i``."""
        def coefficient(values: np.ndarray) -> torch.Tensor:
            return torch.tensor(float(values[index]), dtype=torch.float32, device=x.device)

        eps, variance = torch.split(model_output, x.shape[1], dim=1)
        frac = (variance + 1) / 2  # the variance channels are in [-1, 1] for [posterior variance, beta]
        log_variance = (frac * coefficient(self.log_betas)
                        + (1 - frac) * coefficient(self.posterior_log_variance_clipped))
        x0 = coefficient(self.sqrt_recip_alphas_cumprod) * x - coefficient(self.sqrt_recipm1_alphas_cumprod) * eps
        mean = coefficient(self.posterior_mean_coef1) * x0 + coefficient(self.posterior_mean_coef2) * x
        return mean, log_variance


def sample_generators(ids: Sequence[int], seed: int, device) -> List[torch.Generator]:
    """One noise generator per sample ID: a sample depends on its ID only, not on the batch or the number of GPUs."""
    return [torch.Generator(device=device).manual_seed((seed + i * 1000003) % (2 ** 63 - 1)) for i in ids]


def sample_labels(ids: Sequence[int], label_seed: int, num_classes: int = 1000) -> List[int]:
    """A class label per sample ID, drawn uniformly."""
    return [torch.randint(num_classes, (1,), generator=torch.Generator().manual_seed(label_seed + i * 1000003)).item()
            for i in ids]


def _draw(generators: Sequence[torch.Generator], shape, device) -> torch.Tensor:
    return torch.stack([torch.randn(shape, generator=g, device=device) for g in generators])


@torch.no_grad()
def sample_class_conditional(model, ids: Sequence[int], seed: int = 0, label_seed: int = 42, steps: int = 250,
                             cfg_scale: float = 1.5, device="cuda", latent_size: int = 32,
                             labels: Optional[Sequence[int]] = None, precision: str = "fp32",
                             step_callback: Optional[Callable] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample latents ``[N, 4, latent_size, latent_size]`` for the sample IDs ``ids``.

    Classifier-free guidance follows DiT: the conditional and the null-class prediction are evaluated in one batch
    and guidance is applied to the first three noise channels. The model chooses its number of passes from the
    timestep of every step (its pass schedule); nothing has to be configured here.
    Returns the latents (still scaled by the VAE scaling factor) and the labels.
    """
    device = torch.device(device)
    model.eval()
    ids = list(ids)
    count = len(ids)
    generators = sample_generators(ids, seed, device)
    shape = (model.in_channels, latent_size, latent_size)
    x = _draw(generators, shape, device)
    if labels is None:
        labels = sample_labels(ids, label_seed, model.num_classes)
    labels = torch.as_tensor(labels, device=device, dtype=torch.long)
    if labels.shape != (count,) or not bool(((labels >= 0) & (labels < model.num_classes)).all()):
        raise ValueError("need one class label per sample")
    y = torch.cat([labels, torch.full_like(labels, model.num_classes)])  # second half: null class

    ddpm = SpacedDDPM(steps)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with autocast:
        for index in range(ddpm.num_steps - 1, -1, -1):
            t = torch.full((2 * count,), ddpm.timestep_map[index], device=device, dtype=torch.long)
            if cfg_scale == 1.0:
                out = model(x, t[:count], labels)
            else:
                out = model.forward_with_cfg(torch.cat([x, x]), t, y, cfg_scale)[:count]
            mean, log_variance = ddpm.step_statistics(out, x, index)
            if index > 0:
                x = mean + torch.exp(0.5 * log_variance) * _draw(generators, shape, device)
            else:
                x = mean
            if step_callback is not None:
                step_callback(x, ddpm.timestep_map[index])
    return x, labels
