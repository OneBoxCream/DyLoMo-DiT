"""Bit-exact comparison of this repository with the original implementation the released weights were trained with.

Maintainer test. The original code base is not part of this repository, so the test is skipped unless the
environment variable ``DYLOMO_REF_ADAPTER`` names a Python file that exposes

    build_class_cond(config) -> reference      build_t2i(config) -> reference

where ``config`` is one of the public configs and ``reference`` provides ``.module`` (an ``nn.Module`` whose
``state_dict`` uses the same names as the public model), ``.to(device)`` and

    class-conditional:  forward(x, t, y, passes), forward_with_cfg(x, t, y, cfg_scale, passes),
                        sample(ids, seed, label_seed, steps, cfg_scale, device, pass_of_timestep)
    text-to-image:      forward(latent, text, sigma, text_lens)

All locations of the original code are read from the environment by the adapter; none is stored here.

Comparison protocol: fp32, TF32 disabled, every parameter and buffer randomised (also those that are
zero-initialised), state dict copied with ``strict=True``, outputs compared with ``torch.equal``.

On CUDA the comparison runs with ``torch.use_deterministic_algorithms(True)``: the scatter-add that sums the
selected experts of a token (``index_add_``) accumulates in a non-deterministic order on CUDA, so without it two
forward passes of the *same* model already differ in the last bits (``test_cuda_scatter_add_is_not_deterministic``
records this for the original implementation).
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic cuBLAS, before CUDA starts

import pytest
import torch

import dylomo
from dylomo.diffusion import SpacedDDPM, sample_class_conditional

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = os.environ.get("DYLOMO_REF_ADAPTER")
pytestmark = pytest.mark.skipif(not ADAPTER, reason="DYLOMO_REF_ADAPTER is not set (maintainer test)")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

T2I_SMALL = dict(model_type="dylomo-dit-t2i", hidden_size=256, num_layers=6, attention_head_dim=64, loop_span=[2, 4],
                 pass_schedule=dict(thresholds=[], passes=[2]), routing_heads=2, experts_per_head=8, top_k=3,
                 expert_hidden_size=96, shared_expert_hidden_size=160, dense_ffn_hidden_size=384, route_scale=4.9,
                 residual_streams=4, latent_channels=48, text_channels=96, time_channels=64)
T2I_MEDIUM = dict(T2I_SMALL, hidden_size=512, num_layers=10, attention_head_dim=128, loop_span=[3, 7], routing_heads=4,
                  experts_per_head=24, top_k=6, expert_hidden_size=320, shared_expert_hidden_size=320,
                  dense_ffn_hidden_size=1280, text_channels=3584)
CLASS_SMALL = dict(model_type="dylomo-dit-class-cond", hidden_size=128, num_layers=6, attention_head_dim=32,
                   loop_span=[2, 4], pass_schedule=dict(thresholds=[500], passes=[3, 1]), routing_heads=2,
                   experts_per_head=5, top_k=2, expert_hidden_size=48, shared_expert_hidden_size=48,
                   dense_ffn_hidden_size=256)


def adapter():
    spec = importlib.util.spec_from_file_location("dylomo_reference_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def no_tf32(deterministic=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def randomise(module: torch.nn.Module, seed: int) -> None:
    """Give every parameter and buffer non-trivial values (zero-initialised tensors would hide differences)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            if not tensor.is_floating_point():
                continue
            if float(tensor.abs().max()) == 0.0 or tensor.numel() <= 16:
                tensor.add_(torch.randn(tensor.shape, generator=generator) * 0.1)


def pair(builder, config, seed, device):
    torch.manual_seed(seed)
    reference = builder(config)
    randomise(reference.module, seed + 1)
    model = dylomo.build_model(config).eval()
    result = model.load_state_dict(reference.module.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert sorted(model.state_dict()) == sorted(reference.module.state_dict())
    return reference.to(device), model.to(device)


def report(name, a, b):
    diff = (a.float() - b.float()).abs().max().item()
    print(f"{name}: equal={torch.equal(a, b)} max_abs_diff={diff:.3e} |ref|max={a.abs().max().item():.3e}")
    return diff


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tier", ["b", "l", "xl"])
def test_class_conditional_forward(tier, device):
    no_tf32()
    config = dylomo.load_config(ROOT / "configs" / f"dylomo_dit_{tier}.json")
    reference, model = pair(adapter().build_class_cond, config, seed=10, device=device)
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(4, 4, 32, 32, generator=generator).to(device)
    y = torch.tensor([1, 207, 999, 1000], device=device)
    with torch.no_grad():
        for timestep in (3, 499, 500, 999):
            t = torch.full((4,), timestep, device=device, dtype=torch.long)
            passes = model.pass_schedule.passes_at(timestep)
            assert passes == (3 if timestep < 500 else 1)
            ref = reference.forward(x, t, y, passes)
            out = model(x, t, y)
            assert ref.abs().max() > 0
            report(f"{tier}/{device}/t={timestep}/passes={passes}", ref, out)
            assert torch.equal(ref, out)
        # a fixed two-pass forward (override), and classifier-free guidance
        t = torch.full((4,), 250, device=device, dtype=torch.long)
        assert torch.equal(reference.forward(x, t, y, 2), model(x, t, y, passes=2))
        y_cfg = torch.tensor([5, 77, 1000, 1000], device=device)
        ref = reference.forward_with_cfg(x, t, y_cfg, 1.5, 3)
        out = model.forward_with_cfg(x, t, y_cfg, 1.5)
        report(f"{tier}/{device}/cfg", ref, out)
        assert torch.equal(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast comparison needs CUDA")
@pytest.mark.parametrize("tier", ["b", "xl"])
def test_class_conditional_forward_bf16_autocast(tier):
    """The mixed-precision path (fp32 weights, bf16 matrix products) of ``--precision bf16``."""
    no_tf32()
    config = dylomo.load_config(ROOT / "configs" / f"dylomo_dit_{tier}.json")
    reference, model = pair(adapter().build_class_cond, config, seed=10, device="cuda")
    x = torch.randn(4, 4, 32, 32, generator=torch.Generator().manual_seed(123)).cuda()
    y = torch.tensor([1, 207, 999, 1000], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timestep in (100, 800):
            t = torch.full((4,), timestep, device="cuda", dtype=torch.long)
            ref = reference.forward(x, t, y, model.pass_schedule.passes_at(timestep))
            out = model(x, t, y)
            report(f"{tier}/cuda/bf16-autocast/t={timestep}", ref, out)
            assert out.dtype == torch.float32 and torch.equal(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_scatter_add_is_not_deterministic():
    """Documents why the CUDA comparisons need deterministic algorithms: without them the original model does not
    reproduce its own output bit for bit (the differences are rounding-level)."""
    no_tf32(deterministic=False)
    config = dylomo.load_config(ROOT / "configs" / "dylomo_dit_b.json")
    reference, model = pair(adapter().build_class_cond, config, seed=10, device="cuda")
    x = torch.randn(4, 4, 32, 32, generator=torch.Generator().manual_seed(123)).cuda()
    y = torch.tensor([1, 207, 999, 1000], device="cuda")
    t = torch.full((4,), 3, device="cuda", dtype=torch.long)
    with torch.no_grad():
        first = reference.forward(x, t, y, 3)
        repeats = [reference.forward(x, t, y, 3) for _ in range(3)]
        ours = model(x, t, y)
    self_diff = max(report("reference vs itself (default CUDA mode)", first, other) for other in repeats)
    cross_diff = report("reference vs this repository (default CUDA mode)", first, ours)
    scale = first.abs().max().item()
    assert self_diff / scale < 1e-4 and cross_diff / scale < 1e-4
    no_tf32(deterministic=True)


@pytest.mark.parametrize("device", DEVICES)
def test_class_conditional_mixed_timesteps(device):
    """A batch whose samples need different pass counts equals the per-group forwards of the original."""
    no_tf32()
    reference, model = pair(adapter().build_class_cond, CLASS_SMALL, seed=20, device=device)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(4, 4, 16, 16, generator=generator).to(device)
    t = torch.tensor([10, 800, 499, 500], device=device)
    y = torch.tensor([0, 1, 2, 1000], device=device)
    with torch.no_grad():
        out = model(x, t, y)
        low, high = torch.tensor([0, 2], device=device), torch.tensor([1, 3], device=device)
        assert torch.equal(out[low], reference.forward(x[low], t[low], y[low], 3))
        assert torch.equal(out[high], reference.forward(x[high], t[high], y[high], 1))


@pytest.mark.parametrize("device", DEVICES)
def test_class_conditional_sampler(device):
    """DDPM with guidance and the 3 | 1 pass schedule: final latents equal the original sampler's.

    The full 250-step protocol on CUDA; 12 steps (which still use both pass counts) on the CPU to keep it short.
    """
    no_tf32()
    reference, model = pair(adapter().build_class_cond, CLASS_SMALL, seed=30, device=device)
    ids = [0, 1, 49999]
    steps = 250 if device == "cuda" else 12
    ref_latents, ref_labels, calls = reference.sample(ids, 0, 42, steps, 1.5, device, model.pass_schedule.passes_at)
    latents, labels = sample_class_conditional(model, ids, seed=0, label_seed=42, steps=steps, cfg_scale=1.5,
                                               device=device, latent_size=32)
    assert len(calls) == steps and {r for _, r in calls} == {1, 3}
    if steps == 250:
        assert sum(r for _, r in calls) == 500  # two passes on average
    assert torch.equal(labels.cpu(), ref_labels.cpu())
    assert torch.isfinite(ref_latents).all()
    report(f"sampler/{device}/{steps} steps", ref_latents, latents)
    assert torch.equal(ref_latents, latents)


def test_ddpm_schedule_matches_reference():
    reference = adapter().build_class_cond(CLASS_SMALL).diffusion(250)
    ddpm = SpacedDDPM(250)
    assert list(reference.timestep_map) == ddpm.timestep_map
    import numpy as np
    for name, mine in (("betas", ddpm.betas), ("posterior_log_variance_clipped", ddpm.posterior_log_variance_clipped),
                       ("sqrt_recip_alphas_cumprod", ddpm.sqrt_recip_alphas_cumprod),
                       ("sqrt_recipm1_alphas_cumprod", ddpm.sqrt_recipm1_alphas_cumprod),
                       ("posterior_mean_coef1", ddpm.posterior_mean_coef1),
                       ("posterior_mean_coef2", ddpm.posterior_mean_coef2)):
        assert np.array_equal(getattr(reference, name), mine), name


def t2i_inputs(config, device, text_len, batch, height, width, seed):
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, config["latent_channels"], height, width, generator=generator).to(device)
    text = torch.randn(batch, text_len, config["text_channels"], generator=generator).to(device)
    sigma = torch.rand(batch, generator=generator).to(device)
    return latent, text, sigma


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["small", "medium"])
def test_text_to_image_forward(name, device):
    no_tf32()
    config = dict(T2I_SMALL if name == "small" else T2I_MEDIUM)
    reference, model = pair(adapter().build_t2i, config, seed=40, device=device)
    with torch.no_grad():
        cases = [("equal lengths", 2, [9, 9]), ("unequal lengths", 3, [9, 4, 7]), ("no text", 2, [0, 0]),
                 ("one sample", 1, [5])]
        for label, batch, text_lens in cases:
            latent, text, sigma = t2i_inputs(config, device, 9, batch, 6, 10, seed=7)
            ref = reference.forward(latent, text, sigma, text_lens)
            out = model(latent, text, sigma, text_lens)
            assert ref.abs().max() > 0
            report(f"t2i-{name}/{device}/{label}", ref, out)
            assert torch.equal(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast comparison needs CUDA")
def test_text_to_image_forward_bf16_autocast():
    """The mixed-precision path used for sampling (fp32 weights, bf16 matmuls) is bit-identical as well, and
    storing the expert weights in bf16 does not change it."""
    no_tf32()
    config = dict(T2I_MEDIUM)
    reference, model = pair(adapter().build_t2i, config, seed=50, device="cuda")
    latent, text, sigma = t2i_inputs(config, "cuda", 11, 2, 8, 8, seed=9)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = reference.forward(latent, text, sigma, [11, 6])
        out = model(latent, text, sigma, [11, 6])
        report("t2i-medium/cuda/bf16-autocast", ref, out)
        assert torch.equal(ref, out)
        dylomo.io.cast_expert_weights(model, torch.bfloat16)
        assert torch.equal(ref, model(latent, text, sigma, [11, 6]))


@pytest.mark.skipif(not os.environ.get("DYLOMO_REF_T2I_SAMPLER"), reason="DYLOMO_REF_T2I_SAMPLER is not set")
@pytest.mark.parametrize("device", DEVICES)
def test_text_to_image_sampler(device):
    """Euler sampling with APG: final latents equal the original sampler's."""
    from dylomo.flow import sample_euler
    no_tf32()
    config = dict(T2I_SMALL)
    reference, model = pair(adapter().build_t2i, config, seed=60, device=device)
    generator = torch.Generator().manual_seed(3)
    text = torch.randn(1, 7, config["text_channels"], generator=generator).to(device)
    noise = torch.randn(1, config["latent_channels"], 4, 6, generator=generator).to(device)
    apg = dict(momentum=-0.5, eta=0.15, norm_threshold=0.0)
    autocast = device == "cuda"
    ref = reference.sample(text, [7], noise, 12, 1.6, 5.0, apg, autocast)
    out = sample_euler(model, text, [7], noise, steps=12, shift=1.6, guidance_scale=5.0, guidance="apg",
                       apg_momentum=-0.5, apg_eta=0.15, autocast=autocast)
    report(f"t2i-sampler/{device}", ref, out)
    assert torch.equal(ref, out)
    ref = reference.sample(text, [7], noise, 5, 3.0, 4.0, None, autocast)
    out = sample_euler(model, text, [7], noise, steps=5, shift=3.0, guidance_scale=4.0, guidance="cfg",
                       autocast=autocast)
    assert torch.equal(ref, out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
