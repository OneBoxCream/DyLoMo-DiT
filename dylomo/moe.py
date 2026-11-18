"""Multi-head sparse experts (the feed-forward network of all layers except the first and the last).

    y = W_merge [ sum_{e in top-k of head j} w_e^(j) E_e^(j)(u^(j)) ]_{j=1..H} + E_shared(x),   u = W_split x

Each of the ``H`` heads owns a router and ``E`` experts; a token selects ``k`` experts per head. The router gives its
experts sigmoid scores, selects the top ``k`` after adding a per-expert routing bias, and weights the selected experts
by their scores, L1-normalised over the ``k`` and multiplied by ``route_scale``. The routing bias takes part in the
selection only. It is a plain buffer that is loaded with the weights and never changes at inference time.

The multi-head layout and the routing rule follow MAGI-2 preview.

Portions of this file are adapted from MAGI-2 preview (Copyright (c) 2025-2026 SandAI, Apache License 2.0,
https://github.com/SandAI-org/MAGI-2-preview/tree/073c84f2102ec3c9287623113a103c14402770ad) and have been modified;
see NOTICE.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GroupLinear, RMSNorm, clamped_swiglu, clamped_swiglu_interleaved


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Rows ``ends[g-1]:ends[g]`` of ``x`` ``[N, in]`` are multiplied by ``weight[g]`` ``[in, out]``.

    Uses ``torch._grouped_mm`` when the installed PyTorch provides it and a per-group loop otherwise.
    """
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(x, weight, offs=ends)
    return grouped_matmul_loop(x, weight, ends)


def grouped_matmul_loop(x: torch.Tensor, weight: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Reference implementation of :func:`grouped_matmul`: one matrix product per non-empty group."""
    outputs, start = [], 0
    for group, end in enumerate(ends.tolist()):
        if end > start:
            outputs.append(x[start:end] @ weight[group])
        start = end
    if not outputs:
        return x.new_zeros(0, weight.shape[-1])
    return torch.cat(outputs, dim=0)


class SparseExperts(nn.Module):
    """Routers and routed experts of all heads. Input / output: ``[S, heads * head_dim]``.

    Parameters (``HE = heads * experts`` rows, head-major):
      ``gate``        ``[HE, d]``        router weights (fp32)
      ``w_gate_up``   ``[HE, d, 2 f]``   expert up-projection; columns ``[:f]`` gate branch, ``[f:]`` up branch
      ``w_down``      ``[HE, f, d]``     expert down-projection
      ``expert_bias`` ``[HE]``           routing bias (buffer; added to the scores for the top-k selection only)
    """

    def __init__(self, hidden: int, heads: int, experts: int, top_k: int, expert_hidden: int,
                 route_scale: float = 4.9, chunk_size: int = 65536):
        super().__init__()
        assert hidden % heads == 0 and 1 <= top_k <= experts
        self.heads, self.experts, self.top_k = heads, experts, top_k
        self.head_dim = hidden // heads
        self.expert_hidden = expert_hidden
        self.route_scale = route_scale
        self.chunk_size = chunk_size  # assignments processed per expert call (bounds the intermediate activations)
        total = heads * experts
        self.gate = nn.Parameter(torch.empty(total, self.head_dim, dtype=torch.float32))
        self.w_gate_up = nn.Parameter(torch.empty(total, self.head_dim, 2 * expert_hidden, dtype=torch.float32))
        self.w_down = nn.Parameter(torch.empty(total, expert_hidden, self.head_dim, dtype=torch.float32))
        self.register_buffer("expert_bias", torch.zeros(total, dtype=torch.float32), persistent=True)
        with torch.no_grad():
            nn.init.normal_(self.gate, 0.0, 0.02)
            nn.init.normal_(self.w_gate_up, 0.0, self.head_dim ** -0.5)
            nn.init.normal_(self.w_down, 0.0, expert_hidden ** -0.5)

    # ------------------------------------------------------------------ routing
    def route(self, x_heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x_heads`` ``[S, H, d]`` -> routing weights ``[H, S, k]`` (fp32) and expert indices ``[H, S, k]``."""
        with torch.autocast(device_type=x_heads.device.type, enabled=False):
            heads, experts = self.heads, self.experts
            gate = self.gate.view(heads, experts, self.head_dim).float()
            logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
            scores = torch.sigmoid(logits)
            selection = scores + self.expert_bias.view(heads, 1, experts)  # the bias only affects the selection
            _, index = torch.topk(selection, self.top_k, dim=-1)
            weights = scores.gather(-1, index)
            weights = F.normalize(weights, p=1, dim=-1, eps=1e-12)
            weights = weights * self.route_scale
        return weights, index

    # ------------------------------------------------------------------ experts
    def _expert_ffn(self, x: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """``x``: assignments sorted by expert ``[N, d]``; ``ends``: cumulative assignment count per expert."""
        dtype = x.dtype
        hidden = self.expert_hidden
        outputs = []
        for start in range(0, max(1, x.shape[0]), self.chunk_size):
            part = x[start:start + self.chunk_size]
            part_ends = (ends - start).clamp(0, part.shape[0]).to(torch.int32)
            gate_up = grouped_matmul(part.contiguous(), self.w_gate_up.to(dtype), part_ends)
            act = clamped_swiglu(gate_up[:, :hidden], gate_up[:, hidden:], out_dtype=dtype)
            outputs.append(grouped_matmul(act.contiguous(), self.w_down.to(dtype), part_ends))
        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        heads, experts, top_k, head_dim = self.heads, self.experts, self.top_k, self.head_dim
        device = x.device
        x_heads = x.view(num_tokens, heads, head_dim)
        weights, index = self.route(x_heads)
        compute_dtype = (torch.get_autocast_dtype(x.device.type)
                         if torch.is_autocast_enabled(x.device.type) else self.w_gate_up.dtype)

        # one row per (head, token, selected expert): the expert it goes to and the row of x it reads
        expert_id = (index + torch.arange(heads, device=device).view(heads, 1, 1) * experts).reshape(-1)
        rows = (torch.arange(num_tokens, device=device).view(1, num_tokens, 1) * heads
                + torch.arange(heads, device=device).view(heads, 1, 1))
        rows = rows.expand(heads, num_tokens, top_k).reshape(-1)
        order = torch.argsort(expert_id, stable=True)
        expert_sorted, rows_sorted = expert_id[order], rows[order]
        weights_sorted = weights.reshape(-1)[order]
        counts = torch.bincount(expert_sorted, minlength=heads * experts)
        ends = counts.cumsum(0).to(torch.int32)

        x_flat = x_heads.reshape(num_tokens * heads, head_dim)
        x_sorted = x_flat.to(compute_dtype).index_select(0, rows_sorted).contiguous()
        y = self._expert_ffn(x_sorted, ends)
        y = y.float() * weights_sorted.unsqueeze(-1)
        out = torch.zeros(num_tokens * heads, head_dim, dtype=torch.float32, device=device).index_add_(0, rows_sorted, y)
        return out.view(num_tokens, heads * head_dim).to(x.dtype)

    @torch.no_grad()
    def selected_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Expert indices ``[H, S, k]`` chosen for the tokens ``x`` ``[S, heads * head_dim]`` (for analysis)."""
        return self.route(x.view(x.shape[0], self.heads, self.head_dim))[1]


class MultiHeadSparseFFN(nn.Module):
    """``pre_norm -> split -> SparseExperts -> merge``, plus an always-on shared expert on the normalised input.

    As for the attention, ``pre_norm`` is applied by the caller; ``forward`` expects normalised tokens.
    """

    is_dense = False

    def __init__(self, hidden: int, heads: int, experts: int, top_k: int, expert_hidden: int, shared_hidden: int,
                 route_scale: float = 4.9, norm_groups: int = 1, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = RMSNorm(hidden, groups=norm_groups, eps=eps)
        self.split_linear = GroupLinear(hidden, hidden)
        self.merge_linear = GroupLinear(hidden, hidden)
        self.core = SparseExperts(hidden, heads, experts, top_k, expert_hidden, route_scale=route_scale)
        self.shared_fc1 = GroupLinear(hidden, 2 * shared_hidden)
        self.shared_fc2 = GroupLinear(shared_hidden, hidden)

    def forward(self, n: torch.Tensor, group_sizes: Optional[Sequence[int]] = None) -> torch.Tensor:
        out = self.merge_linear(self.core(self.split_linear(n)))
        return out + self.shared_fc2(clamped_swiglu_interleaved(self.shared_fc1(n)))


class DenseFFN(nn.Module):
    """Gated feed-forward network of the first and the last layer (one set of weights per token group)."""

    is_dense = True

    def __init__(self, hidden: int, ffn_hidden: int, groups: int = 1, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = RMSNorm(hidden, groups=groups, eps=eps)
        self.up_gate = GroupLinear(hidden, 2 * ffn_hidden, groups)
        self.down = GroupLinear(ffn_hidden, hidden, groups)

    def forward(self, n: torch.Tensor, group_sizes: Optional[Sequence[int]] = None) -> torch.Tensor:
        return self.down(clamped_swiglu_interleaved(self.up_gate(n, group_sizes)), group_sizes)
