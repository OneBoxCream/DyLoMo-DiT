"""DyLoMo-DiT for text-to-image generation with rectified flow.

Text tokens and image tokens form one sequence. There is no adaLN: the diffusion time is written into the first
``time_channels`` input channels of every token as a sinusoidal code (text tokens use time 0). The residual is a
4-stream residual (:class:`dylomo.residual.HyperConnection`). The first and the last layer are dense, with separate
weights for image and text tokens; all layers in between use multi-head sparse experts. The middle ``loop_span``
layers share their weights and are run ``passes`` times with residual factor ``1 / passes``.

The network predicts the velocity ``v = noise - x0`` for ``x_t = (1 - sigma) * x0 + sigma * noise``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from ..attention import Attention
from ..layers import (FourierRoPE, GroupLinear, RMSNorm, TokenLayout, grid_coords, sinusoidal_embedding,
                      text_coords)
from ..loop import PassSchedule, layer_schedule
from ..moe import DenseFFN, MultiHeadSparseFFN
from ..residual import HyperConnection

DEFAULTS = dict(
    model_type="dylomo-dit-t2i",
    attention_head_dim=128,
    route_scale=4.9,
    residual_streams=4,
    time_channels=64,
    rope_temperature=10000.0,
)

IMAGE, TEXT = 0, 1


class InputAdapter(nn.Module):
    """Projects image latents and text features to the residual streams and writes the time code."""

    def __init__(self, latent_channels: int, text_channels: int, width: int, time_channels: int, head_dim: int,
                 rope_temperature: float):
        super().__init__()
        self.time_channels = time_channels
        self.video_embedder = nn.Linear(latent_channels, width, bias=True)
        self.text_embedder = nn.Linear(text_channels, width, bias=True)
        self.rope = FourierRoPE(head_dim, rope_temperature)

    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor, coords: torch.Tensor,
                times: torch.Tensor):
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            x = torch.cat([self.video_embedder(image_tokens.float()), self.text_embedder(text_tokens.float())], dim=0)
            if self.time_channels > 0:
                code = sinusoidal_embedding(self.time_channels, times)
                x = torch.cat([code.to(x.dtype), x[:, self.time_channels:]], dim=-1)
            rope = self.rope(coords)
        return x, rope


class OutputAdapter(nn.Module):
    def __init__(self, latent_channels: int, width: int, eps: float = 1e-6):
        super().__init__()
        self.final_norm = RMSNorm(width, eps=eps, out_dtype=torch.float32)
        self.final_linear = nn.Linear(width, latent_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.final_linear(self.final_norm(x.float()))


class T2IBlock(nn.Module):
    """Attention and feed-forward sub-layers around the multi-stream residual."""

    def __init__(self, hidden: int, heads: int, head_dim: int, streams: int, ffn: nn.Module, attention_groups: int):
        super().__init__()
        self.hidden, self.streams = hidden, streams
        self.attention = Attention(hidden, heads, head_dim, groups=attention_groups)
        self.mlp = ffn
        self.hc_attn = HyperConnection(hidden, streams, mode="ohc")
        self.hc_mlp = HyperConnection(hidden, streams, mode="ohc")
        self.hc_norm = RMSNorm(hidden * streams, groups=attention_groups, out_dtype=torch.float32)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, layout: TokenLayout, scale: float) -> torch.Tensor:
        num_tokens = x.shape[0]
        gs = layout.group_sizes
        s = x.reshape(num_tokens, self.streams, self.hidden)

        a, cache = self.hc_attn.read(s, self.hc_norm(s.reshape(num_tokens, -1), gs))
        out = self.attention(self.attention.pre_norm(a, gs), rope, layout)
        if scale != 1.0:
            out = out * scale
        s = self.hc_attn.write(s, out, cache)

        a, cache = self.hc_mlp.read(s, self.hc_norm(s.reshape(num_tokens, -1), gs))
        out = self.mlp(self.mlp.pre_norm(a, gs), gs)
        if scale != 1.0:
            out = out * scale
        s = self.hc_mlp.write(s, out, cache)
        return s.reshape(num_tokens, -1)


class DyLoMoT2I(nn.Module):
    """Text-to-image DyLoMo-DiT. See ``configs/dylomo_dit_t2i.json``."""

    def __init__(self, config: Dict):
        super().__init__()
        cfg = dict(DEFAULTS)
        cfg.update(config)
        if cfg["model_type"] != DEFAULTS["model_type"]:
            raise ValueError(f"not a text-to-image DyLoMo-DiT config: {cfg['model_type']}")
        self.config = cfg
        hidden, num_layers, head_dim = cfg["hidden_size"], cfg["num_layers"], cfg["attention_head_dim"]
        streams = cfg["residual_streams"]
        if hidden % head_dim or head_dim % 8:
            raise ValueError("hidden_size must be a multiple of attention_head_dim, which must be a multiple of 8")
        if not 0 <= cfg["time_channels"] <= hidden:
            raise ValueError("time_channels must not exceed hidden_size")
        self.hidden, self.num_layers, self.streams = hidden, num_layers, streams
        self.latent_channels, self.text_channels = cfg["latent_channels"], cfg["text_channels"]
        self.loop_span = tuple(cfg["loop_span"]) if cfg.get("loop_span") is not None else None
        schedule = PassSchedule.from_config(cfg["pass_schedule"])
        if schedule.thresholds:
            raise ValueError("the text-to-image model uses a fixed number of passes")
        self.passes = schedule.passes[0]
        layer_schedule(num_layers, self.loop_span, self.passes)  # validates the span

        width = hidden * streams
        self.pre_adapter = InputAdapter(self.latent_channels, self.text_channels, width, cfg["time_channels"],
                                        head_dim, cfg["rope_temperature"])

        def block(index: int) -> T2IBlock:
            dense = index in (0, num_layers - 1)
            if dense:
                ffn = DenseFFN(hidden, cfg["dense_ffn_hidden_size"], groups=2)
            else:
                ffn = MultiHeadSparseFFN(hidden, cfg["routing_heads"], cfg["experts_per_head"], cfg["top_k"],
                                         cfg["expert_hidden_size"], cfg["shared_expert_hidden_size"],
                                         route_scale=cfg["route_scale"], norm_groups=2)
            return T2IBlock(hidden, hidden // head_dim, head_dim, streams, ffn, attention_groups=2 if dense else 1)

        self.layers = nn.ModuleList([block(i) for i in range(num_layers)])
        self.post_adapter = OutputAdapter(self.latent_channels, width)

    def forward(self, latent: torch.Tensor, text: torch.Tensor, sigma: torch.Tensor,
                text_lens: Optional[Sequence[int]] = None, passes: Optional[int] = None) -> torch.Tensor:
        """``latent``: ``[B, C, H, W]`` noisy latents, ``text``: ``[B, L, D]`` text features, ``sigma``: ``[B]``
        diffusion times in ``[0, 1]``. ``text_lens`` gives the number of valid text tokens per sample (default ``L``);
        a length of 0 is the unconditional (empty-text) branch. Returns the velocity ``[B, C, H, W]``.
        """
        if latent.ndim != 4:
            raise ValueError("expected latents of shape [B, C, H, W]")
        b, c, h, w = latent.shape
        length = text.shape[1]
        device = latent.device
        text_lens = list(text_lens) if text_lens is not None else [length] * b
        if c != self.latent_channels or text.shape[0] != b or text.shape[-1] != self.text_channels:
            raise ValueError("latent / text dimensions do not match the model")
        if len(text_lens) != b or any(not isinstance(n, int) or not 0 <= n <= length for n in text_lens):
            raise ValueError("text_lens must hold one valid length per sample")
        if tuple(sigma.shape) != (b,):
            raise ValueError("sigma must have shape [B]")

        image_tokens = latent.permute(0, 2, 3, 1).reshape(-1, c)
        text_tokens = torch.cat([text[i, : text_lens[i]] for i in range(b)], dim=0)
        layout = TokenLayout([h * w] * b, text_lens)
        image_coords = grid_coords((1, h, w), (1, h, w), device=device)
        coords = torch.cat([image_coords.repeat(b, 1)] + [text_coords(n, device=device) for n in text_lens], dim=0)
        times = torch.cat([sigma.reshape(b, 1, 1).expand(b, h, w).reshape(-1).float(),
                           torch.zeros(sum(text_lens), device=device)], dim=0)

        x, rope = self.pre_adapter(image_tokens, text_tokens, coords, times)
        for index, scale in layer_schedule(self.num_layers, self.loop_span, self.passes if passes is None else passes):
            x = self.layers[index](x, rope, layout, scale)
        out = self.post_adapter(x[: layout.num_image])
        return out.reshape(b, h, w, -1).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------ bookkeeping
    def num_parameters(self) -> Dict[str, int]:
        """Total parameters and the parameters used by one image token in one visit of every layer: the routers
        in full, ``top_k`` experts per head, the image half of the per-group weights and the image adapters."""
        total = sum(p.numel() for p in self.parameters())
        active = 0
        for name, module in self.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                full = f"{name}.{pname}" if name else pname
                if full.startswith("pre_adapter.text_embedder"):
                    continue
                if ".core.w_" in full:
                    core = module
                    active += p.numel() * core.top_k // core.experts
                elif isinstance(module, (GroupLinear, RMSNorm)) and module.groups > 1:
                    active += p.numel() // module.groups
                else:
                    active += p.numel()
        return dict(total=total, active_per_image_token=active)
