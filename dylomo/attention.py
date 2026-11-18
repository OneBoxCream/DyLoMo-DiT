"""Multi-head self-attention with QK normalisation, rotary positions, a per-head output gate and an attention sink.

The attention sink is one extra softmax column per head whose logit is a learnable scalar and whose
value is zero, so a head can attend to "nothing". It is realised inside
``F.scaled_dot_product_attention`` by appending a few channels to Q / K / V.

The layer layout (QK normalisation, attention sink, per-head gate) follows MAGI-2 preview.

Portions of this file are adapted from MAGI-2 preview (Copyright (c) 2025-2026 SandAI, Apache License 2.0,
https://github.com/SandAI-org/MAGI-2-preview/tree/073c84f2102ec3c9287623113a103c14402770ad) and have been modified;
see NOTICE.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GroupLinear, RMSNorm, TokenLayout, apply_rope

_SINK_PAD = 8


def sdpa_with_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: Optional[torch.Tensor],
                   scale: float) -> torch.Tensor:
    """``q, k, v``: ``[B, H, L, D]``; ``sink``: ``[H]`` fp32 logits (or ``None``). Returns ``[B, H, L, D]``.

    Augmentation: ``q' = [q, 1, 1, 0...]``, ``k'_j = [k_j, 0, 0, 0...]``, ``k'_sink = [0..., hi, lo, 0...]`` with
    ``hi + lo = sink / scale`` (the split keeps precision when the compute dtype is bf16) and ``v'_sink = 0``.
    """
    if sink is None:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    b, h, length, d = q.shape
    dtype = q.dtype
    s = sink.float() / scale
    hi = s.to(dtype)
    lo = (s - hi.float()).to(dtype)
    q_aug = torch.cat([q, q.new_zeros(b, h, length, _SINK_PAD)], dim=-1)
    q_aug[..., d] = 1.0
    q_aug[..., d + 1] = 1.0
    k_aug = torch.cat([k, k.new_zeros(b, h, k.shape[2], _SINK_PAD)], dim=-1)
    k_sink = k.new_zeros(b, h, 1, d + _SINK_PAD)
    k_sink[..., d] = hi.view(1, h, 1)
    k_sink[..., d + 1] = lo.view(1, h, 1)
    k_aug = torch.cat([k_sink, k_aug], dim=2)
    v_aug = torch.cat([v.new_zeros(b, h, 1, d), v], dim=2)
    v_aug = F.pad(v_aug, (0, _SINK_PAD))  # fused attention kernels need equal Q / K / V head sizes
    return F.scaled_dot_product_attention(q_aug, k_aug, v_aug, scale=scale)[..., :d]


class Attention(nn.Module):
    """Self-attention over a packed token sequence.

    The pre-normalisation is a submodule (``pre_norm``) but is applied by the caller: the class-conditional block
    modulates the normalised tokens with adaLN before they enter the attention, the text-to-image block uses them
    directly. ``forward`` therefore expects *normalised* tokens.
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, groups: int = 1, eps: float = 1e-6):
        super().__init__()
        self.hidden, self.heads, self.head_dim, self.groups = hidden, heads, head_dim, groups
        self.scale = 1.0 / math.sqrt(head_dim)
        self.sinks = nn.Parameter(torch.zeros(1, heads, dtype=torch.float32))
        self.pre_norm = RMSNorm(hidden, groups=groups, eps=eps)
        self.linear_qkv = GroupLinear(hidden, 3 * heads * head_dim, groups)
        self.linear_proj = GroupLinear(heads * head_dim, hidden, groups)
        self.linear_g = GroupLinear(hidden, heads, groups)
        self.q_norm = RMSNorm(head_dim, groups=groups, eps=eps, out_dtype=torch.float32)
        self.k_norm = RMSNorm(head_dim, groups=groups, eps=eps, out_dtype=torch.float32)

    def forward(self, n: torch.Tensor, rope: torch.Tensor, layout: TokenLayout) -> torch.Tensor:
        """``n``: ``[S, hidden]`` normalised tokens; ``rope``: ``[S, rope_dim]`` fp32."""
        num_tokens = n.shape[0]
        gs = layout.group_sizes
        heads, head_dim = self.heads, self.head_dim
        qkv = self.linear_qkv(n, gs)
        q, k, v = torch.split(qkv, [heads * head_dim] * 3, dim=-1)
        q = q.reshape(num_tokens, heads, head_dim)
        k = k.reshape(num_tokens, heads, head_dim)
        v = v.reshape(num_tokens, heads, head_dim)
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else v.dtype
        q = self.q_norm(q, gs)
        k = self.k_norm(k, gs)
        q = apply_rope(q.float(), rope)
        k = apply_rope(k.float(), rope)
        q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)

        sink = self.sinks[0]
        if layout.equal_len:
            qb, kb, vb = (layout.to_batched(t).transpose(1, 2) for t in (q, k, v))  # [B, H, L, D]
            ob = sdpa_with_sink(qb, kb, vb, sink, self.scale)
            out = layout.from_batched(ob.transpose(1, 2))  # [S, H, D]
        else:  # samples of different length: one attention call per sample
            out = torch.empty_like(v)
            for index in layout.sample_index_lists(n.device):
                qb, kb, vb = (t.index_select(0, index).unsqueeze(0).transpose(1, 2) for t in (q, k, v))
                ob = sdpa_with_sink(qb, kb, vb, sink, self.scale)
                out.index_copy_(0, index, ob.transpose(1, 2).squeeze(0))

        gate = self.linear_g(n, gs).reshape(num_tokens, heads, 1)
        out = out * torch.sigmoid(gate)
        out = out.reshape(num_tokens, heads * head_dim).to(compute_dtype)
        return self.linear_proj(out, gs)
