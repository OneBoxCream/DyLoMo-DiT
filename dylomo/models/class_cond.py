"""DyLoMo-DiT for class-conditional latent diffusion (ImageNet 256x256).

Patchified VAE latents; timestep and class embeddings are summed into a conditioning vector that every sub-layer
receives through adaLN-Zero; rotary positions inside attention; the output layer predicts the noise and the variance
(8 channels for a 4-channel latent). The middle ``loop_span`` layers share their weights across the passes of the loop
and the number of passes depends on the timestep (:class:`dylomo.loop.PassSchedule`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..attention import Attention
from ..layers import FourierRoPE, RMSNorm, TokenLayout, grid_coords
from ..loop import PassSchedule, layer_schedule
from ..moe import DenseFFN, MultiHeadSparseFFN

DEFAULTS = dict(
    model_type="dylomo-dit-class-cond",
    in_channels=4,
    patch_size=2,
    num_classes=1000,
    attention_head_dim=128,
    route_scale=4.9,
    timestep_embedding_size=256,
)


# --------------------------------------------------------------------------- conditioning
class TimestepEmbedder(nn.Module):
    """Sinusoidal features of the integer timestep (cos | sin over geometrically spaced frequencies, as in
    GLIDE / ADM by OpenAI, MIT licence) followed by a two-layer MLP. ``mlp.0`` / ``mlp.2`` are the checkpoint names.
    """

    def __init__(self, hidden: int, frequency_size: int = 256, max_period: float = 10000.0):
        super().__init__()
        if frequency_size % 2:
            raise ValueError("frequency_size must be even")
        self.mlp = nn.Sequential(nn.Linear(frequency_size, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        half = frequency_size // 2
        # evaluated once on the CPU, so that every device uses the same frequency values
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device="cpu") / half)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, None].float() * self.frequencies[None]
        return self.mlp(torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1))


class ClassEmbedder(nn.Module):
    """Class embedding table; index ``num_classes`` is the null class used for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden: int):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


def _zero_modulation(hidden: int, outputs: int) -> nn.Sequential:
    module = nn.Sequential(nn.SiLU(), nn.Linear(hidden, outputs, bias=True))
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x``: ``[B * T, C]`` tokens, ``shift`` / ``scale``: ``[B, C]`` -> ``x * (1 + scale) + shift`` per sample."""
    return (x.reshape(shift.shape[0], -1, x.shape[-1]) * (1 + scale[:, None, :]) + shift[:, None, :]).reshape_as(x)


def _gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.reshape(gate.shape[0], -1, x.shape[-1]) * gate[:, None, :]).reshape_as(x)


def _residual_add(h: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """``h + update`` accumulated in fp32 (the residual stream stays fp32 under mixed precision)."""
    return (h.float() + update.float()).to(h.dtype)


# --------------------------------------------------------------------------- block
class DyLoMoBlock(nn.Module):
    """Attention and feed-forward sub-layers, each ``h <- h + scale * g(c) * F(adaLN(h, c))``.

    ``scale`` is ``1 / r`` inside the looped segment (``r`` passes) and 1 elsewhere. The feed-forward is a dense
    gated network in the first and the last layer and multi-head sparse experts in all other layers.
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: nn.Module):
        super().__init__()
        self.attention = Attention(hidden, heads, head_dim)
        self.mlp = ffn
        # one linear layer produces (shift, scale, gate) for the attention and for the feed-forward
        self.adaLN_modulation = _zero_modulation(hidden, 6 * hidden)

    def forward(self, h: torch.Tensor, rope: torch.Tensor, layout: TokenLayout, scale: float,
                c: torch.Tensor) -> torch.Tensor:
        """``h``: ``[B * T, hidden]`` tokens, ``c``: ``[B, hidden]`` conditioning, ``scale``: residual factor."""
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, dim=1)

        a = _modulate(self.attention.pre_norm(h, layout.group_sizes), shift_a, scale_a)
        out = _gate(self.attention(a, rope, layout), gate_a)
        h = _residual_add(h, out * scale)

        a = _modulate(self.mlp.pre_norm(h, [h.shape[0]]), shift_m, scale_m)
        out = _gate(self.mlp(a, [a.shape[0]]), gate_m)
        return _residual_add(h, out * scale)


# --------------------------------------------------------------------------- model
class DyLoMoDiT(nn.Module):
    """Class-conditional DyLoMo-DiT. See ``configs/dylomo_dit_{b,l,xl}.json`` for the released sizes."""

    def __init__(self, config: Dict):
        super().__init__()
        cfg = dict(DEFAULTS)
        cfg.update(config)
        if cfg["model_type"] != DEFAULTS["model_type"]:
            raise ValueError(f"not a class-conditional DyLoMo-DiT config: {cfg['model_type']}")
        if cfg["patch_size"] != 2 or cfg["in_channels"] != 4:
            raise ValueError("the released models use 4 latent channels and 2x2 patches")
        self.config = cfg
        hidden, num_layers, head_dim = cfg["hidden_size"], cfg["num_layers"], cfg["attention_head_dim"]
        if hidden % head_dim:
            raise ValueError("hidden_size must be a multiple of attention_head_dim")
        self.hidden, self.num_layers = hidden, num_layers
        self.in_channels, self.patch_size = cfg["in_channels"], cfg["patch_size"]
        self.out_channels = 2 * self.in_channels  # noise and variance
        self.num_classes = cfg["num_classes"]
        self.loop_span = tuple(cfg["loop_span"])
        self.pass_schedule = PassSchedule.from_config(cfg["pass_schedule"])
        layer_schedule(num_layers, self.loop_span, self.pass_schedule.max_passes)  # validates the span

        patch_dim = self.in_channels * self.patch_size ** 2
        self.x_embedder = nn.Linear(patch_dim, hidden)
        self.y_embedder = ClassEmbedder(self.num_classes, hidden)
        self.t_embedder = TimestepEmbedder(hidden, cfg["timestep_embedding_size"])
        self.rope = FourierRoPE(head_dim)

        def ffn(index: int) -> nn.Module:
            if index in (0, num_layers - 1):
                return DenseFFN(hidden, cfg["dense_ffn_hidden_size"])
            return MultiHeadSparseFFN(hidden, cfg["routing_heads"], cfg["experts_per_head"], cfg["top_k"],
                                      cfg["expert_hidden_size"], cfg["shared_expert_hidden_size"],
                                      route_scale=cfg["route_scale"])

        self.layers = nn.ModuleList([DyLoMoBlock(hidden, hidden // head_dim, head_dim, ffn(i))
                                     for i in range(num_layers)])
        self.final_norm = RMSNorm(hidden, out_dtype=torch.float32)
        self.final = nn.Linear(hidden, patch_dim * 2)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        self.final_modulation = _zero_modulation(hidden, 2 * hidden)

    # ------------------------------------------------------------------ patches
    @staticmethod
    def patchify(x: torch.Tensor) -> torch.Tensor:
        """``[B, 4, H, W]`` -> ``[B, (H/2) * (W/2), 16]``."""
        b, c, h, w = x.shape
        if c != 4 or h % 2 or w % 2:
            raise ValueError("expected a four-channel latent of even size")
        return x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 3, 5, 1).reshape(b, (h // 2) * (w // 2), 16)

    @staticmethod
    def unpatchify(x: torch.Tensor, h: int, w: int, channels: int = 8) -> torch.Tensor:
        b = x.shape[0]
        return x.reshape(b, h // 2, w // 2, 2, 2, channels).permute(0, 5, 1, 3, 2, 4).reshape(b, channels, h, w)

    # ------------------------------------------------------------------ forward
    def passes_for(self, t: torch.Tensor) -> List[int]:
        """Number of passes of every sample, from its timestep."""
        return [self.pass_schedule.passes_at(int(v)) for v in t.tolist()]

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                passes: Optional[int] = None) -> torch.Tensor:
        """``x``: ``[B, 4, H, W]`` noisy latents, ``t``: ``[B]`` integer timesteps (0 .. 999), ``y``: ``[B]`` class
        labels (``num_classes`` = null class). Returns ``[B, 8, H, W]``: predicted noise and variance channels.

        ``passes`` overrides the pass schedule (analysis only; the released weights expect their own schedule).
        """
        per_sample = [passes] * x.shape[0] if passes is not None else self.passes_for(t)
        distinct = sorted(set(per_sample))
        if len(distinct) == 1:
            return self._forward(x, t, y, distinct[0])
        # samples whose noise levels ask for different numbers of passes are run group by group
        out = None
        for r in distinct:
            index = torch.tensor([i for i, v in enumerate(per_sample) if v == r], device=x.device)
            part = self._forward(x.index_select(0, index), t.index_select(0, index), y.index_select(0, index), r)
            if out is None:
                out = part.new_empty((x.shape[0],) + tuple(part.shape[1:]))
            out.index_copy_(0, index, part)
        return out

    def _forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, passes: int) -> torch.Tensor:
        b, _, h, w = x.shape
        tokens = self.patchify(x)
        num_tokens = tokens.shape[1]
        c = self.t_embedder(t) + self.y_embedder(y)
        z = self.x_embedder(tokens).to(c.dtype).reshape(b * num_tokens, -1)
        layout = TokenLayout([num_tokens] * b, [0] * b)
        coords = grid_coords((1, h // 2, w // 2), (1, h // 2, w // 2), device=x.device)
        rope = self.rope(coords).repeat(b, 1)
        for index, scale in layer_schedule(self.num_layers, self.loop_span, passes):
            z = self.layers[index](z, rope, layout, scale, c)
        with torch.autocast(x.device.type, enabled=False):
            out = self.final_norm(z.float())
            shift, scale = self.final_modulation(c.float()).chunk(2, dim=1)
            out = self.final(_modulate(out, shift, scale))
        return self.unpatchify(out.view(b, num_tokens, 2 * 16), h, w)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float,
                         guided_channels: int = 3) -> torch.Tensor:
        """Classifier-free guidance in the DiT protocol.

        ``x``, ``t``, ``y`` hold the conditional samples in the first half and the null-class samples in the second
        half; both halves are evaluated on the latents of the first half, and guidance is applied to the first
        ``guided_channels`` noise channels only (the remaining noise channel and the variance are left as predicted).
        """
        n = x.shape[0] // 2
        out = self.forward(torch.cat([x[:n], x[:n]], dim=0), t, y)
        cond, uncond = out[:n, :guided_channels], out[n:, :guided_channels]
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([torch.cat([guided, guided], dim=0), out[:, guided_channels:]], dim=1)

    # ------------------------------------------------------------------ bookkeeping
    def num_parameters(self) -> Dict[str, int]:
        """Total parameters and the parameters one token uses (routed experts counted as ``top_k / experts``)."""
        total, active = 0, 0.0
        for name, p in self.named_parameters():
            total += p.numel()
            if ".core.w_" in name:
                core = self.get_submodule(name.rsplit(".", 1)[0])
                active += p.numel() * core.top_k / core.experts
            else:
                active += p.numel()
        return dict(total=total, active=round(active))
