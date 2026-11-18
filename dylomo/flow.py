"""Rectified-flow sampling for the text-to-image model: Euler steps with adaptive projected guidance (APG).

``x_t = (1 - sigma) * x0 + sigma * noise`` and the network predicts ``v = noise - x0``. Sampling integrates from
``sigma = 1`` to ``0`` with Euler steps; ``sigma`` is taken at equally spaced points and passed through the shift
transform ``sigma <- s * sigma / (1 + (s - 1) * sigma)``.

Guidance. The unconditional branch is the empty text (no text tokens). ``"apg"`` (Sadat et al., 2024) acts on the
predicted clean image: the difference between the conditional and the unconditional prediction (with a running
average, ``momentum``) is split into the components parallel and orthogonal to the conditional prediction, and the
parallel component is down-weighted by ``eta``. ``"cfg"`` is plain classifier-free guidance on the velocity.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch


def shifted_sigmas(steps: int, shift: float, device=None) -> torch.Tensor:
    """``steps + 1`` noise levels from 1 to 0 after the shift transform."""
    sigma = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    return shift * sigma / (1 + (shift - 1) * sigma)


@torch.no_grad()
def sample_euler(model, text: torch.Tensor, text_lens: Sequence[int], noise: torch.Tensor, steps: int = 50,
                 shift: float = 1.6, guidance_scale: float = 5.0, guidance: str = "apg", apg_momentum: float = -0.5,
                 apg_eta: float = 0.15, apg_norm_threshold: float = 0.0, autocast: bool = True,
                 step_callback: Optional[Callable] = None) -> torch.Tensor:
    """Integrate ``noise`` ``[B, C, H, W]`` to a clean latent.

    ``text``: ``[B, L, D]`` text features with ``text_lens`` valid tokens per sample. ``apg_norm_threshold > 0``
    additionally rescales the guidance difference to at most that norm (0 = no rescaling, the setting used for the
    released weights). The running average of APG is kept per call, i.e. it is reset for every sampled batch.
    """
    if guidance not in ("cfg", "apg"):
        raise ValueError("guidance must be 'cfg' or 'apg'")
    values = (shift, guidance_scale, apg_momentum, apg_eta, apg_norm_threshold)
    if steps < 1 or shift <= 0 or apg_norm_threshold < 0 or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid sampling parameters")
    if noise.ndim != 4:
        raise ValueError(f"expected noise of shape [B, C, H, W], got {tuple(noise.shape)}")
    x = noise.clone().float().unsqueeze(2)  # [B, C, 1, H, W]: one latent frame
    sigmas = shifted_sigmas(steps, shift, device=x.device)
    reduce_dims = tuple(range(1, x.ndim))
    no_text = [0] * len(text_lens)
    average = None
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        times = sigma.expand(x.shape[0]).clone()
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
            v = model(x.squeeze(2), text, times, text_lens=text_lens).float().unsqueeze(2)
            if guidance_scale != 1:
                v_null = model(x.squeeze(2), text, times, text_lens=no_text).float().unsqueeze(2)
                if guidance == "apg":
                    s = float(sigma)
                    x0_cond, x0_null = x - s * v, x - s * v_null
                    difference = x0_cond - x0_null
                    if apg_momentum:
                        average = difference if average is None else difference + apg_momentum * average
                        difference = average
                    if apg_norm_threshold > 0:
                        norm = difference.norm(p=2, dim=reduce_dims, keepdim=True)
                        difference = difference * torch.clamp(apg_norm_threshold / norm, max=1.0)
                    unit = x0_cond / x0_cond.norm(p=2, dim=reduce_dims, keepdim=True).clamp_min(1e-12)
                    parallel = (difference * unit).sum(dim=reduce_dims, keepdim=True) * unit
                    x0 = x0_cond + (guidance_scale - 1) * ((difference - parallel) + apg_eta * parallel)
                    v = (x - x0) / s
                else:
                    v = v_null + guidance_scale * (v - v_null)
        x.add_(v, alpha=float(next_sigma - sigma))
        if not torch.isfinite(x).all():
            raise FloatingPointError("non-finite latent during sampling")
        if step_callback is not None:
            step_callback(x.squeeze(2), float(next_sigma))
    return x.squeeze(2)
