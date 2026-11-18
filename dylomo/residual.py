"""Residual connections.

``HyperConnection(mode="none")`` is the ordinary single-stream residual ``h <- h + f(h)`` used by the
class-conditional models. ``mode="ohc"`` is the 4-stream residual of the text-to-image model: the sub-layer reads a
gated combination of the streams, and the streams are mixed by a learned SO(4) rotation (a pair of unit quaternions)
before the sub-layer output is written back with learned per-stream gates:

    x_in = sum_n H_pre[n] * s[n]
    s'   = H_res @ s + H_post[:, None] * f(x_in),        H_res in SO(4)

All gate arithmetic runs in fp32. The form of the read / write gates follows the multi-stream residual of MAGI-2
preview; the SO(4) mixing replaces its doubly stochastic mixing matrix.

Portions of this file are adapted from MAGI-2 preview (Copyright (c) 2025-2026 SandAI, Apache License 2.0,
https://github.com/SandAI-org/MAGI-2-preview/tree/073c84f2102ec3c9287623113a103c14402770ad) and have been modified;
see NOTICE.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _quat_left_matrix(p: torch.Tensor) -> torch.Tensor:
    """``L(p)`` with ``L(p) vec(x) = vec(p * x)`` (Hamilton product). ``[..., 4] -> [..., 4, 4]``."""
    a, b, c, d = p.unbind(-1)
    rows = [torch.stack([a, -b, -c, -d], -1), torch.stack([b, a, -d, c], -1),
            torch.stack([c, d, a, -b], -1), torch.stack([d, -c, b, a], -1)]
    return torch.stack(rows, -2)


def _quat_right_matrix(q: torch.Tensor) -> torch.Tensor:
    """``R(q)`` with ``R(q) vec(x) = vec(x * q)``."""
    a, b, c, d = q.unbind(-1)
    rows = [torch.stack([a, -b, -c, -d], -1), torch.stack([b, a, d, -c], -1),
            torch.stack([c, -d, a, b], -1), torch.stack([d, c, -b, a], -1)]
    return torch.stack(rows, -2)


def _unit_quaternion(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise to unit length; a (near-)zero quaternion falls back to the identity."""
    scale = q.abs().amax(-1, keepdim=True).clamp(min=eps)
    scaled = q / scale
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    unit = scaled / norm.clamp(min=eps)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm >= eps / scale, unit, identity)


def so4_from_quaternions(pq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``pq`` ``[..., 8]`` -> two unit quaternions ``p, q`` -> ``L(p) R(q)`` in SO(4), ``[..., 4, 4]``."""
    p, q = pq[..., :4], pq[..., 4:]
    p, q = _unit_quaternion(p, eps), _unit_quaternion(q, eps)
    left, right = _quat_left_matrix(p), _quat_right_matrix(q)
    # element-wise product-sum instead of matmul: full fp32 precision regardless of the matmul backend settings
    return (left.unsqueeze(-1) * right.unsqueeze(-3)).sum(-2)


class HyperConnection(nn.Module):
    """One read / write pair around a sub-layer (each layer has one for the attention and one for the FFN)."""

    def __init__(self, hidden: int, streams: int, mode: str = "ohc", alpha_init: float = 0.01,
                 magnitude: float = 2.0):
        super().__init__()
        assert mode in ("ohc", "none")
        self.mode = mode
        self.n = 1 if mode == "none" else streams
        self.hidden = hidden
        self.magnitude = magnitude
        if mode == "none":
            return
        assert streams == 4, "the SO(4) mixing needs 4 streams"
        n = self.n
        self.matmul_scale = 1.0 / math.sqrt(float(n * hidden))
        self.alpha_pre = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.bias_pre = nn.Parameter(torch.zeros(n))
        self.bias_post = nn.Parameter(torch.zeros(n))
        # logits of the read gates (n), the write gates (n) and the stream mixing (n * n, of which 8 are used)
        self.phi = nn.Parameter(torch.zeros(n * hidden, n + n + n * n))
        self.alpha_res = nn.Parameter(torch.full((1,), float(alpha_init)), requires_grad=False)
        bias_res = torch.zeros(n, n)
        bias_res.view(-1)[0] = 1.0  # p = (1, 0, 0, 0)
        bias_res.view(-1)[4] = 1.0  # q = (1, 0, 0, 0)
        self.bias_res = nn.Parameter(bias_res)

    def read(self, s: torch.Tensor, normed_flat: Optional[torch.Tensor]):
        """``s``: ``[S, n, hidden]`` streams; ``normed_flat``: RMS-normalised ``[S, n * hidden]`` (fp32).

        Returns the sub-layer input ``[S, hidden]`` and a cache for :meth:`write`.
        """
        if self.mode == "none":
            return s[:, 0], None
        n = self.n
        with torch.autocast(device_type=s.device.type, enabled=False):
            logits = torch.matmul(normed_flat.to(torch.float32), self.phi)
            pre_raw, post_raw, res_raw = torch.split(logits, [n, n, n * n], dim=-1)
            h_pre = torch.sigmoid(self.alpha_pre * self.matmul_scale * pre_raw + self.bias_pre.unsqueeze(0))
            x_in = torch.einsum("tn,tnc->tc", h_pre, s.float())
        return x_in.to(s.dtype), (post_raw, res_raw)

    def write(self, s: torch.Tensor, out: torch.Tensor, cache) -> torch.Tensor:
        """``s``: ``[S, n, hidden]``; ``out``: sub-layer output ``[S, hidden]`` -> new streams."""
        if self.mode == "none":
            return (s[:, 0].float() + out.float()).to(s.dtype).unsqueeze(1)
        n = self.n
        post_raw, res_raw = cache
        with torch.autocast(device_type=s.device.type, enabled=False):
            h_post = self.magnitude * torch.sigmoid(
                self.alpha_post * self.matmul_scale * post_raw + self.bias_post.unsqueeze(0))
            width = res_raw.shape[-1]
            res_logits = (self.alpha_res * self.matmul_scale * res_raw.to(torch.float32)
                          + self.bias_res.reshape(1, n * n)[:, :width])
            h_res = so4_from_quaternions(res_logits[:, :8])
            sf, of = s.to(torch.float32), out.to(torch.float32)
            mixed = torch.einsum("tij,tjc->tic", h_res.to(torch.float32), sf)
            new = mixed + torch.einsum("tn,tc->tnc", h_post.to(torch.float32), of)
        return new.to(s.dtype)
