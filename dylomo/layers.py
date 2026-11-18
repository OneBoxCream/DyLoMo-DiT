"""Building blocks shared by the class-conditional and the text-to-image model.

Token convention. Every layer works on a *packed* token matrix ``[S, C]``: the image tokens of
all samples first (samples contiguous), then the text tokens of all samples. The class-conditional
model simply has no text tokens. :class:`TokenLayout` records the per-sample lengths.

Weights that exist once per token group (image / text) carry a leading ``groups`` dimension; all
layers of the class-conditional model and the sparse-expert layers of the text-to-image model use
``groups == 1``.

The clamped SwiGLU, the Fourier rotary embedding with its coordinate layout and the RMSNorm parameterisation follow
MAGI-2 preview.

Portions of this file are adapted from MAGI-2 preview (Copyright (c) 2025-2026 SandAI, Apache License 2.0,
https://github.com/SandAI-org/MAGI-2-preview/tree/073c84f2102ec3c9287623113a103c14402770ad) and have been modified;
see NOTICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- activation
def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: float = 1.702, limit: float = 7.0,
                   out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``g * sigmoid(alpha * g) * (u + 1)`` with ``g`` clipped to ``<= limit`` and ``u`` to ``[-limit, limit]``.

    Computed in fp32 and cast to ``out_dtype`` (default: the dtype of ``gate``).
    """
    out_dtype = gate.dtype if out_dtype is None else out_dtype
    g = gate.to(torch.float32).clamp(max=limit)
    u = up.to(torch.float32).clamp(min=-limit, max=limit)
    return (g * torch.sigmoid(alpha * g) * (u + 1.0)).to(out_dtype)


def clamped_swiglu_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Clamped SwiGLU on an interleaved tensor: gate = ``x[..., 0::2]``, up = ``x[..., 1::2]``."""
    out_dtype = x.dtype
    x = x.to(torch.float32)
    return clamped_swiglu(x[..., ::2], x[..., 1::2], out_dtype=out_dtype)


# --------------------------------------------------------------------------- normalisation
class RMSNorm(nn.Module):
    """RMSNorm with gain ``1 + weight`` (``weight`` is zero-initialised), computed in fp32.

    ``weight`` has shape ``[groups, dim]``. With ``groups > 1`` the rows of the input are split into
    contiguous token groups of sizes ``group_sizes`` and each group uses its own gain.
    """

    def __init__(self, dim: int, groups: int = 1, eps: float = 1e-6, out_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dim, self.groups, self.eps, self.out_dtype = dim, groups, eps, out_dtype
        self.weight = nn.Parameter(torch.zeros(groups, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, group_sizes: Optional[Sequence[int]] = None) -> torch.Tensor:
        out_dtype = self.out_dtype or x.dtype
        t = x.float()
        t = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.groups == 1:
            t = t * (self.weight[0] + 1.0)
        else:
            assert group_sizes is not None and len(group_sizes) == self.groups
            parts = torch.split(t, list(group_sizes), dim=0)
            t = torch.cat([p * (self.weight[i] + 1.0) for i, p in enumerate(parts)], dim=0)
        return t.to(out_dtype)


# --------------------------------------------------------------------------- linear
class GroupLinear(nn.Module):
    """Bias-free linear layer with one weight matrix per token group. ``weight``: ``[groups, out, in]``."""

    def __init__(self, in_features: int, out_features: int, groups: int = 1):
        super().__init__()
        self.in_features, self.out_features, self.groups = in_features, out_features, groups
        self.weight = nn.Parameter(torch.empty(groups, out_features, in_features, dtype=torch.float32))
        with torch.no_grad():
            self.weight.normal_(0.0, 1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor, group_sizes: Optional[Sequence[int]] = None) -> torch.Tensor:
        if self.groups == 1:
            return F.linear(x, self.weight[0])
        assert group_sizes is not None and len(group_sizes) == self.groups
        parts = torch.split(x, list(group_sizes), dim=0)
        return torch.cat([F.linear(p, self.weight[i]) for i, p in enumerate(parts)], dim=0)


# --------------------------------------------------------------------------- timestep features (text-to-image)
def sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal features of a diffusion time in ``[0, 1]`` (multiplied by 1000): ``[N] -> [N, dim]`` (cos | sin)."""
    position = position.to(torch.float32) * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=position.device) / half)
    args = position[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# --------------------------------------------------------------------------- rotary position embedding
def grid_coords(shape: Tuple[int, int, int], ref_shape: Tuple[int, int, int], offset: Tuple[int, int, int] = (0, 0, 0),
                device=None, dtype=torch.float32) -> torch.Tensor:
    """Coordinates of a ``(T, H, W)`` grid: ``[T*H*W, 9]`` rows ``(t, h, w, T, H, W, ref_T, ref_H, ref_W)``."""
    t_rng = torch.arange(shape[0], device=device, dtype=dtype) + offset[0]
    h_rng = torch.arange(shape[1], device=device, dtype=dtype) + offset[1]
    w_rng = torch.arange(shape[2], device=device, dtype=dtype) + offset[2]
    tg, hg, wg = torch.meshgrid(t_rng, h_rng, w_rng, indexing="ij")
    coords = torch.stack([tg, hg, wg], dim=-1).reshape(-1, 3)
    meta = torch.tensor([*shape, *ref_shape], device=device, dtype=dtype).expand(coords.size(0), -1)
    return torch.cat([coords, meta], dim=-1)


def text_coords(length: int, device=None) -> torch.Tensor:
    """Coordinates of ``length`` text tokens: a ``(length, 1, 1)`` grid placed before the image on the first axis."""
    return grid_coords((length, 1, 1), (1, 1, 1), offset=(-length, 0, 0), device=device)


class FourierRoPE(nn.Module):
    """Element-wise Fourier rotary embedding over three axes.

    With ``head_dim // 8`` frequency bands per axis the output has ``6 * bands`` channels (sin | cos), which
    rotate the first 3/4 of each attention head; the remaining quarter is left unrotated.
    """

    def __init__(self, head_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.head_dim, self.temperature = head_dim, temperature
        self.register_buffer("bands", self.frequency_bands(head_dim, temperature), persistent=False)
        self.out_dim = 6 * (head_dim // 8)

    @staticmethod
    def frequency_bands(head_dim: int, temperature: float) -> torch.Tensor:
        num_bands = head_dim // 8
        exponent = torch.arange(0, num_bands, 1, dtype=torch.int64, device="cpu").to(torch.float32) / num_bands
        return 1.0 / (temperature ** exponent)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        xyz, sizes, refs = coords[:, :3], coords[:, 3:6], coords[:, 6:9]
        scales = (refs - 1) / (sizes - 1).clamp(min=1)
        # an axis of size one has no extent: keep its coordinate unscaled
        scales = torch.where(sizes == 1, torch.ones_like(scales), scales)
        centers = (sizes - 1) / 2
        centers[:, 0] = 0
        xyz = xyz - centers
        proj = xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands  # [L, 3, bands]
        return torch.cat((proj.sin(), proj.cos()), dim=1).flatten(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate the leading channels of ``x`` ``[S, heads, head_dim]`` (fp32) by ``rope`` ``[S, 6 * bands]`` (sin | cos)."""
    sin, cos = rope.tensor_split(2, -1)
    rotated = cos.shape[-1] * 2
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
    xr = x[..., :rotated]
    return torch.cat([xr * cos + _rotate_half(xr) * sin, x[..., rotated:]], dim=-1)


# --------------------------------------------------------------------------- packed token layout
@dataclass
class TokenLayout:
    """Per-sample token counts of a packed sequence ``[image tokens of all samples..., text tokens of all samples...]``."""

    image_lens: List[int]
    text_lens: List[int]

    def __post_init__(self):
        if not self.image_lens or len(self.image_lens) != len(self.text_lens):
            raise ValueError("need one image length and one text length per sample")
        if any(v <= 0 for v in self.image_lens) or any(t < 0 for t in self.text_lens):
            raise ValueError("every sample needs image tokens and a non-negative text length")

    @property
    def batch_size(self) -> int:
        return len(self.image_lens)

    @property
    def num_image(self) -> int:
        return int(sum(self.image_lens))

    @property
    def num_text(self) -> int:
        return int(sum(self.text_lens))

    @property
    def total(self) -> int:
        return self.num_image + self.num_text

    @property
    def group_sizes(self) -> List[int]:
        return [self.num_image, self.num_text]

    @property
    def equal_len(self) -> bool:
        return len(set(self.image_lens)) == 1 and len(set(self.text_lens)) == 1

    def sample_index_lists(self, device) -> List[torch.Tensor]:
        """Row indices of each sample in the packed order (its image tokens, then its text tokens)."""
        index = []
        image_offset, text_offset = 0, self.num_image
        for n_image, n_text in zip(self.image_lens, self.text_lens):
            index.append(torch.cat([torch.arange(image_offset, image_offset + n_image, device=device),
                                    torch.arange(text_offset, text_offset + n_text, device=device)]))
            image_offset += n_image
            text_offset += n_text
        return index

    def to_batched(self, x: torch.Tensor) -> torch.Tensor:
        """``[S, ...]`` packed -> ``[B, L, ...]`` (equal lengths only)."""
        assert self.equal_len
        b, n_image, n_text = self.batch_size, self.image_lens[0], self.text_lens[0]
        image = x[: self.num_image].reshape(b, n_image, *x.shape[1:])
        text = x[self.num_image:].reshape(b, n_text, *x.shape[1:])
        return torch.cat([image, text], dim=1)

    def from_batched(self, xb: torch.Tensor) -> torch.Tensor:
        """``[B, L, ...]`` -> ``[S, ...]`` packed (equal lengths only)."""
        assert self.equal_len
        b, n_image, n_text = self.batch_size, self.image_lens[0], self.text_lens[0]
        image = xb[:, :n_image].reshape(b * n_image, *xb.shape[2:])
        text = xb[:, n_image:].reshape(b * n_text, *xb.shape[2:])
        return torch.cat([image, text], dim=0)
