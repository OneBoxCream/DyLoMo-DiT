"""Count the forward compute of the released models the way the paper does.

GMAC = 10^9 multiply-accumulates of ONE forward pass of the denoiser for one image, measured on the instantiated
model: matrix products are counted by PyTorch's operator-level flop counter (linear layers, the router scoring and
the routed expert products), attention is counted from the shapes of the actual attention calls (the QK^T and PV
products, including the attention-sink column). Normalisation, activations, top-k selection and other element-wise
work are not counted. For a model whose number of passes depends on the noise level, the paper reports the mean over
the 250 sampling steps.

    python tools/count_gmac.py --paper --skip-t2i            # B / L / XL
    python tools/count_gmac.py --paper                       # also the text-to-image model
    python tools/count_gmac.py --config configs/dylomo_dit_xl.json

The count does not depend on the weight values, so the models are instantiated with zero weights and no checkpoint is
needed. The text-to-image model is large; ``--device cuda`` stores its expert weights in bf16.

Self-check. The script fails instead of printing a number if no attention product was counted, or if a released
configuration with the settings of the paper does not reproduce the published value (23.1 / 80.1 / 117.8 GMAC for
DyLoMo-DiT-B / L / XL, 8,333 GMAC for the text-to-image model at 1024x1024 with 13 text tokens).
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dylomo  # noqa: E402
from dylomo.diffusion import SpacedDDPM  # noqa: E402

# GMAC columns of the paper for the released configurations (class-conditional: 250 sampling steps, 256x256;
# text-to-image: 1024x1024, 13 text tokens), with the number of decimals they are published with.
PAPER_GMAC = {"dylomo_dit_b": (23.1, 1), "dylomo_dit_l": (80.1, 1), "dylomo_dit_xl": (117.8, 1),
              "dylomo_dit_t2i": (8333, 0)}
PAPER_STEPS, PAPER_T2I_SIZE, PAPER_T2I_TEXT_TOKENS = 250, (1024, 1024), 13


class SelfCheckError(RuntimeError):
    """The count is not trustworthy on this installation."""


def _shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else tuple(value)


def _grouped_mm_flops(a, b, *args, **kwargs):  # [N, in] x [groups, in, out]: every row meets one group
    a, b = _shape(a), _shape(b)
    return 2 * a[0] * a[1] * b[2]


_grouped_mm_flops._get_raw = True  # the flop counter may pass tensors or shapes; both are handled


def build_zero_model(config: dict, device: str = "cpu") -> torch.nn.Module:
    """The model with all weights zero (no random initialisation). Derived constants keep their values."""
    with torch.device("meta"):
        model = dylomo.build_model(config)
        if device != "cpu":
            dylomo.io.cast_expert_weights(model, torch.bfloat16)
    for module in model.modules():
        for name, parameter in module._parameters.items():
            if parameter is not None:
                zeros = torch.zeros(parameter.shape, dtype=parameter.dtype, device=device)
                module._parameters[name] = torch.nn.Parameter(zeros, requires_grad=False)
        for name, buffer in module._buffers.items():
            if buffer is None:
                continue
            if name in module._non_persistent_buffers_set:  # computed in the constructor, never on "meta"
                module._buffers[name] = buffer.to(device)
            else:
                module._buffers[name] = torch.zeros(buffer.shape, dtype=buffer.dtype, device=device)
    return model.eval()


def count_macs(forward) -> dict:
    """Multiply-accumulates executed by ``forward()``: ``{"matmul": ..., "attention": ..., "total": ...}``.

    Attention is taken out of the hands of the flop counter, whose treatment of the fused attention operators
    differs between PyTorch versions: every attention call is recorded with its shapes and replaced by a stand-in
    (the values do not matter for counting), and its two products are counted as
    ``batch * heads * L_q * L_k * (D_q + D_v)``.
    """
    from torch.utils.flop_counter import FlopCounterMode

    attention_macs = 0

    def recorded_attention(query, key, value, *args, **kwargs):
        nonlocal attention_macs
        batch_and_heads = 1
        for size in query.shape[:-2]:
            batch_and_heads *= size
        attention_macs += batch_and_heads * query.shape[-2] * key.shape[-2] * (query.shape[-1] + value.shape[-1])
        return value.new_zeros(*query.shape[:-1], value.shape[-1])

    mapping = {}
    grouped_mm = getattr(torch.ops.aten, "_grouped_mm", None) if hasattr(torch, "_grouped_mm") else None
    if grouped_mm is not None:
        mapping[grouped_mm] = _grouped_mm_flops
    with torch.no_grad(), mock.patch.object(F, "scaled_dot_product_attention", recorded_attention), \
            FlopCounterMode(display=False, custom_mapping=mapping) as counter:
        forward()
    matmul_macs = counter.get_total_flops() // 2  # the flop counter counts a multiply-add as two
    if attention_macs <= 0:
        raise SelfCheckError("no attention product was counted: the result would be too small")
    if matmul_macs <= 0:
        raise SelfCheckError("no matrix product was counted")
    return dict(matmul=matmul_macs, attention=attention_macs, total=matmul_macs + attention_macs)


def class_cond_report(config: dict, steps: int = 250, latent_size: int = 32, device: str = "cpu") -> dict:
    model = build_zero_model(config, device)
    schedule = model.pass_schedule
    x = torch.zeros(1, model.in_channels, latent_size, latent_size, device=device)
    y = torch.zeros(1, dtype=torch.long, device=device)
    t = torch.full((1,), 500, dtype=torch.long, device=device)
    counts = {r: count_macs(lambda r=r: model(x, t, y, passes=r)) for r in sorted(set(schedule.passes))}
    by_passes = {r: c["total"] / 1e9 for r, c in counts.items()}
    timesteps = SpacedDDPM(steps).timestep_map
    mean = sum(by_passes[schedule.passes_at(step)] for step in timesteps) / len(timesteps)
    attention = sum(counts[schedule.passes_at(step)]["attention"] for step in timesteps) / len(timesteps) / 1e9
    span = model.loop_span
    return dict(parameters=model.num_parameters(), gmac_by_passes=by_passes, gmac=mean, gmac_attention=attention,
                mean_passes=schedule.mean_passes(timesteps), sampling_steps=steps,
                layer_visits={r: model.num_layers + (r - 1) * (span[1] - span[0]) for r in by_passes})


def t2i_report(config: dict, height: int, width: int, text_tokens: int, device: str = "cpu") -> dict:
    model = build_zero_model(config, device)
    h, w = height // 16, width // 16
    latent = torch.zeros(1, model.latent_channels, h, w, device=device)
    text = torch.zeros(1, max(text_tokens, 1), model.text_channels, device=device)
    sigma = torch.full((1,), 0.5, device=device)
    counts = count_macs(lambda: model(latent, text, sigma, text_lens=[text_tokens]))
    return dict(parameters=model.num_parameters(), gmac=counts["total"] / 1e9,
                gmac_attention=counts["attention"] / 1e9, image_tokens=h * w, text_tokens=text_tokens,
                height=height, width=width)


def released_name(config: dict):
    """Name of the released configuration ``config`` is identical to, or ``None``."""
    for name in PAPER_GMAC:
        if dylomo.load_config(ROOT / "configs" / f"{name}.json") == config:
            return name
    return None


def check_against_paper(name, gmac: float) -> None:
    """Raise if a released configuration with the settings of the paper does not give the published GMAC."""
    expected, decimals = PAPER_GMAC[name]
    if round(gmac, decimals) != expected:
        raise SelfCheckError(f"{name}: counted {gmac:.3f} GMAC, the paper has {expected}. The count is wrong on this "
                             f"installation (PyTorch {torch.__version__}); do not use it.")


def report(path: Path, args) -> dict:
    """Count one configuration, check it, print one line and return the numbers."""
    config = dylomo.load_config(path)
    label = path.stem if path.is_file() else path.name
    released = released_name(config)
    if config["model_type"] == "dylomo-dit-t2i":
        result = t2i_report(config, args.height, args.width, args.text_tokens, args.device)
        paper_settings = (args.height, args.width) == PAPER_T2I_SIZE and args.text_tokens == PAPER_T2I_TEXT_TOKENS
        if released and paper_settings:
            check_against_paper(released, result["gmac"])
        counts = result["parameters"]
        print(f"{label}: total {counts['total'] / 1e9:.2f}B, active per image token "
              f"{counts['active_per_image_token'] / 1e9:.2f}B; {result['gmac']:,.1f} GMAC for one forward pass at "
              f"{args.width}x{args.height} ({result['image_tokens']} image + {result['text_tokens']} text tokens)")
    else:
        result = class_cond_report(config, steps=args.steps, device=args.device)
        paper_settings = args.steps == PAPER_STEPS
        if released and paper_settings:
            check_against_paper(released, result["gmac"])
        counts = result["parameters"]
        per_pass = ", ".join(f"{r} pass{'es' if r > 1 else ''}: {g:.3f} ({result['layer_visits'][r]} layer visits)"
                             for r, g in result["gmac_by_passes"].items())
        print(f"{label}: total {counts['total'] / 1e6:.1f}M, active {counts['active'] / 1e6:.1f}M; GMAC {per_pass}; "
              f"mean over {args.steps} sampling steps ({result['mean_passes']:.2f} passes): {result['gmac']:.3f}")
    result["checked_against_paper"] = bool(released and paper_settings)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="model config (JSON file or model directory)")
    parser.add_argument("--paper", action="store_true", help="all released models with the settings of the paper")
    parser.add_argument("--steps", type=int, default=PAPER_STEPS,
                        help="sampling steps the pass schedule is averaged over")
    parser.add_argument("--height", type=int, default=PAPER_T2I_SIZE[0])
    parser.add_argument("--width", type=int, default=PAPER_T2I_SIZE[1])
    parser.add_argument("--text-tokens", type=int, default=PAPER_T2I_TEXT_TOKENS,
                        help="text tokens fed to the text-to-image denoiser (unpadded prompt)")
    parser.add_argument("--skip-t2i", action="store_true", help="with --paper: skip the text-to-image model")
    parser.add_argument("--json", help="also write the results to this file")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if not args.paper and not args.config:
        parser.error("give --config or --paper")

    if args.paper:
        names = [name for name in PAPER_GMAC if not (args.skip_t2i and name == "dylomo_dit_t2i")]
        paths = [ROOT / "configs" / f"{name}.json" for name in names]
    else:
        paths = [Path(args.config)]
    results = {}
    try:
        for path in paths:
            results[path.stem if path.is_file() else path.name] = report(path, args)
    except SelfCheckError as error:
        raise SystemExit(f"self-check failed: {error}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
