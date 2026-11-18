# Parameter names and shapes

The released safetensors files use exactly the tensor names of the training checkpoints (for the class-conditional
models: the EMA copy of the weights). No tensor is renamed, reshaped or dropped; `convert_checkpoint.py` only removes
what is not needed for inference (raw weights, optimizer state, random-number state, training record).

Notation: `d` hidden width, `A` attention heads, `D` attention head size (`d = A * D`), `H` routing heads,
`E` experts per head, `d_h = d / H`, `f` expert width, `f_s` shared-expert width, `F` width of the dense feed-forward
of the first / last layer, `G` number of token groups of a weight (1 = shared by all tokens; 2 = one copy for image
tokens and one for text tokens), `n` residual streams (text-to-image: 4).

## Common to both model families (`layers.<i>.`)

| Tensor | Shape | Role |
|---|---|---|
| `attention.pre_norm.weight` | `[G, d]` | RMSNorm gain minus one of the attention input |
| `attention.linear_qkv.weight` | `[G, 3 d, d]` | query / key / value projection |
| `attention.q_norm.weight`, `attention.k_norm.weight` | `[G, D]` | RMSNorm of queries / keys (per head) |
| `attention.sinks` | `[1, A]` | attention-sink logit of every head |
| `attention.linear_g.weight` | `[G, A, d]` | per-head sigmoid output gate (reads the normalised input) |
| `attention.linear_proj.weight` | `[G, d, d]` | output projection |
| `mlp.pre_norm.weight` | `[G, d]` | RMSNorm of the feed-forward input |
| `mlp.split_linear.weight` | `[1, d, d]` | `W_split`: input of the routed experts (sparse-expert layers) |
| `mlp.core.gate` | `[H E, d_h]` | router weights, head-major (rows `h E .. (h+1) E - 1` belong to head `h`) |
| `mlp.core.expert_bias` | `[H E]` | routing bias (buffer): added to the scores for the top-k selection only; the files hold its value at the end of training |
| `mlp.core.w_gate_up` | `[H E, d_h, 2 f]` | expert up-projection; columns `[:f]` gate branch, `[f:]` up branch |
| `mlp.core.w_down` | `[H E, f, d_h]` | expert down-projection |
| `mlp.merge_linear.weight` | `[1, d, d]` | `W_merge`: output of the routed experts |
| `mlp.shared_fc1.weight` | `[1, 2 f_s, d]` | shared expert, up-projection (gate / up interleaved: even / odd rows) |
| `mlp.shared_fc2.weight` | `[1, d, f_s]` | shared expert, down-projection |
| `mlp.up_gate.weight` | `[G, 2 F, d]` | dense feed-forward of the first / last layer (gate / up interleaved) |
| `mlp.down.weight` | `[G, d, F]` | dense feed-forward, down-projection |

Linear weights are stored as `[G, out, in]` and are applied without bias. RMSNorm multiplies by `1 + weight`.
The layers of the looped segment appear once; all passes use the same tensors.

## Class-conditional models

| Tensor | Shape | Role |
|---|---|---|
| `x_embedder.weight`, `.bias` | `[d, 16]`, `[d]` | patch embedding (2 x 2 patches of the 4-channel latent) |
| `t_embedder.mlp.0.*`, `t_embedder.mlp.2.*` | `[d, 256]`, `[d, d]` (+ bias) | timestep embedding MLP |
| `y_embedder.embedding_table.weight` | `[1001, d]` | class embedding; row 1000 is the null class |
| `layers.<i>.adaLN_modulation.1.weight`, `.bias` | `[6 d, d]`, `[6 d]` | adaLN-Zero: shift, scale, gate of the attention and of the feed-forward |
| `final_norm.weight` | `[1, d]` | RMSNorm before the output layer |
| `final_modulation.1.weight`, `.bias` | `[2 d, d]`, `[2 d]` | adaLN shift and scale of the output layer |
| `final.weight`, `.bias` | `[32, d]`, `[32]` | output layer: 2 x 2 patch of noise (4 channels) and variance (4 channels) |

All tensors of these models have `G = 1`.

## Text-to-image model

| Tensor | Shape | Role |
|---|---|---|
| `pre_adapter.video_embedder.weight`, `.bias` | `[n d, 48]`, `[n d]` | projection of the image latent (one token per latent position) |
| `pre_adapter.text_embedder.weight`, `.bias` | `[n d, 3584]`, `[n d]` | projection of the text features |
| `layers.<i>.hc_attn.*`, `layers.<i>.hc_mlp.*` | | 4-stream residual around the attention / the feed-forward |
| `...alpha_pre`, `...alpha_post`, `...alpha_res` | `[1]` | scales of the read, write and mixing logits |
| `...bias_pre`, `...bias_post` | `[n]` | biases of the read / write gates |
| `...bias_res` | `[n, n]` | bias of the mixing logits (the first 8 entries parameterise two quaternions) |
| `...phi` | `[n d, 2 n + n^2]` | projection of the normalised streams to the gate and mixing logits |
| `layers.<i>.hc_norm.weight` | `[G, n d]` | RMSNorm over the concatenated streams |
| `post_adapter.final_norm.weight` | `[1, n d]` | RMSNorm before the output layer |
| `post_adapter.final_linear.weight` | `[48, n d]` | output layer (velocity) |

`G = 2` in the first and the last layer (attention projections, norms and the dense feed-forward have separate
weights for image and text tokens) and for `mlp.pre_norm.weight` of every layer; `G = 1` elsewhere. Row 0 of a
`G = 2` tensor belongs to the image tokens, row 1 to the text tokens. The embedder of the image latent keeps its
checkpoint name `video_embedder`.
