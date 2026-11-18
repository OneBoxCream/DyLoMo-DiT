"""Replays inputs and outputs recorded from the original implementation (the one the released weights were trained
with) on two tiny random models. The fixtures were computed on a CPU in fp32; on another machine the results agree
up to floating-point rounding, hence the tolerance.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import dylomo
from dylomo.diffusion import sample_class_conditional
from dylomo.flow import sample_euler

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RTOL, ATOL = 2e-4, 2e-5


def load_fixture(name):
    path = FIXTURES / name
    with safe_open(str(path), framework="pt") as file:
        config = json.loads(file.metadata()["config"])
    data = load_file(str(path))
    model = dylomo.build_model(config).eval()
    model.load_state_dict({k[len("weights."):]: v for k, v in data.items() if k.startswith("weights.")}, strict=True)
    return model, data


def assert_close(actual, expected, what):
    assert actual.shape == expected.shape, what
    scale = expected.abs().max().item()
    error = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL * max(scale, 1.0)), f"{what}: max error {error:.3e} (scale {scale:.3e})"


@torch.no_grad()
def test_class_conditional_forward():
    model, data = load_fixture("class_cond_tiny.safetensors")
    x, y = data["input.x"], data["input.y"]
    for timestep in (120, 499, 500, 870):  # three passes below 500, one from 500 on
        t = torch.full((x.shape[0],), timestep, dtype=torch.long)
        assert_close(model(x, t, y), data[f"output.forward.t{timestep}"], f"forward at t={timestep}")
    t = torch.full((4,), 300, dtype=torch.long)
    out = model.forward_with_cfg(data["input.cfg.x"], t, data["input.cfg.y"], 1.5)
    assert_close(out, data["output.cfg.t300"], "forward with guidance")


@torch.no_grad()
def test_class_conditional_pass_count_matters():
    """Guards the fixture itself: running the wrong number of passes must not reproduce the recorded output."""
    model, data = load_fixture("class_cond_tiny.safetensors")
    x, y = data["input.x"], data["input.y"]
    t = torch.full((x.shape[0],), 120, dtype=torch.long)
    expected = data["output.forward.t120"]
    assert not torch.allclose(model(x, t, y, passes=1), expected, rtol=1e-2, atol=1e-3)
    assert not torch.allclose(model(x, t, y, passes=2), expected, rtol=1e-2, atol=1e-3)


@torch.no_grad()
def test_class_conditional_sampler():
    model, data = load_fixture("class_cond_tiny.safetensors")
    latents, labels = sample_class_conditional(model, [0, 1, 2], seed=0, label_seed=42, steps=20, cfg_scale=1.5,
                                               device="cpu", latent_size=32)
    assert torch.equal(labels, data["output.sample.labels"])
    assert_close(latents, data["output.sample.latents"], "20-step guided DDPM sample")


@torch.no_grad()
def test_text_to_image_forward():
    model, data = load_fixture("t2i_tiny.safetensors")
    latent, text, sigma = data["input.latent"], data["input.text"], data["input.sigma"]
    assert_close(model(latent, text, sigma), data["output.forward.full_text"], "all text tokens")
    assert_close(model(latent, text, sigma, text_lens=[7, 3, 5]), data["output.forward.mixed_text"], "mixed lengths")
    assert_close(model(latent, text, sigma, text_lens=[0, 0, 0]), data["output.forward.no_text"], "no text")


@torch.no_grad()
def test_text_to_image_sampler():
    model, data = load_fixture("t2i_tiny.safetensors")
    text, noise = data["input.text"][:1], data["input.sample.noise"]
    apg = sample_euler(model, text, [7], noise, steps=8, shift=1.6, guidance_scale=5.0, guidance="apg",
                       apg_momentum=-0.5, apg_eta=0.15, autocast=False)
    assert_close(apg, data["output.sample.apg"], "8-step Euler sample with APG")
    cfg = sample_euler(model, text, [7], noise, steps=8, shift=1.6, guidance_scale=5.0, guidance="cfg", autocast=False)
    assert_close(cfg, data["output.sample.cfg"], "8-step Euler sample with CFG")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
