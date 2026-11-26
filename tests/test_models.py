"""Self-contained checks of the model definitions, the schedules and the weight I/O (no external files needed)."""

import json
from pathlib import Path

import pytest
import torch

import dylomo
from dylomo.diffusion import SpacedDDPM
from dylomo.loop import PassSchedule, layer_schedule
from dylomo.moe import SparseExperts, grouped_matmul, grouped_matmul_loop

ROOT = Path(__file__).resolve().parents[1]

TINY_CLASS = dict(model_type="dylomo-dit-class-cond", hidden_size=64, num_layers=4, attention_head_dim=32,
                  loop_span=[1, 3], pass_schedule=dict(thresholds=[500], passes=[3, 1]), routing_heads=2,
                  experts_per_head=4, top_k=2, expert_hidden_size=16, shared_expert_hidden_size=24,
                  dense_ffn_hidden_size=64)
TINY_T2I = dict(model_type="dylomo-dit-t2i", hidden_size=64, num_layers=4, attention_head_dim=32, loop_span=[1, 3],
                pass_schedule=dict(thresholds=[], passes=[2]), routing_heads=2, experts_per_head=8, top_k=2,
                expert_hidden_size=16, shared_expert_hidden_size=24, dense_ffn_hidden_size=64, residual_streams=4,
                latent_channels=48, text_channels=32, time_channels=16)

# total / active parameters of the released class-conditional models (the "params" columns of the paper)
PAPER_COUNTS = dict(b=(128_515_664, 75_800_144), l=(451_293_312, 278_278_272), xl=(686_558_368, 411_307_168))


@pytest.mark.parametrize("tier", ["b", "l", "xl"])
def test_released_class_conditional_configs(tier):
    config = dylomo.load_config(ROOT / "configs" / f"dylomo_dit_{tier}.json")
    with torch.device("meta"):  # shapes only
        model = dylomo.build_model(config)
    counts = model.num_parameters()
    assert (counts["total"], counts["active"]) == PAPER_COUNTS[tier]
    assert model.pass_schedule == PassSchedule((500,), (3, 1))
    lo, hi = model.loop_span
    visits = {r: len(layer_schedule(model.num_layers, model.loop_span, r)) for r in (1, 3)}
    assert visits == ({1: 16, 3: 30} if tier == "xl" else {1: 12, 3: 24})
    assert hi - lo == (7 if tier == "xl" else 6)


def test_released_text_to_image_config():
    config = dylomo.load_config(ROOT / "configs" / "dylomo_dit_t2i.json")
    with torch.device("meta"):
        model = dylomo.build_model(config)
    counts = model.num_parameters()
    assert counts["total"] == 15_481_848_256            # 15.48B
    assert counts["active_per_image_token"] == 1_050_494_912  # 1.05B
    assert len(layer_schedule(model.num_layers, model.loop_span, model.passes)) == 8 + 16 * 2 + 8


def test_repository_index_is_not_a_model_config(tmp_path):
    # the top-level config.json of the ImageNet weights repository only lists the model folders
    (tmp_path / "config.json").write_text(json.dumps({"models": ["DyLoMo-DiT-B", "DyLoMo-DiT-XL"]}))
    with pytest.raises(ValueError, match="DyLoMo-DiT-B, DyLoMo-DiT-XL"):
        dylomo.load_config(tmp_path)


def test_pass_schedule_and_sampling_grid():
    schedule = PassSchedule((500,), (3, 1))
    assert [schedule.passes_at(t) for t in (0, 499, 500, 999)] == [3, 3, 1, 1]
    grid = SpacedDDPM(250).timestep_map
    assert len(grid) == 250 and grid[0] == 0 and grid[-1] == 999
    assert schedule.mean_passes(grid) == 2.0  # two passes on average over the 250 sampling steps
    bands = PassSchedule((400, 600), (3, 2, 1))
    assert [bands.passes_at(t) for t in (399, 400, 599, 600)] == [3, 2, 2, 1]
    assert PassSchedule.from_config(2) == PassSchedule((), (2,))
    with pytest.raises(ValueError):
        PassSchedule((500,), (3,))
    with pytest.raises(ValueError):
        PassSchedule((600, 400), (3, 2, 1))


def test_layer_schedule():
    assert layer_schedule(5, (1, 3), 1) == [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)]
    assert layer_schedule(5, (1, 3), 2) == [(0, 1.0), (1, 0.5), (2, 0.5), (1, 0.5), (2, 0.5), (3, 1.0), (4, 1.0)]
    assert [i for i, _ in layer_schedule(4, (1, 3), 3)] == [0, 1, 2, 1, 2, 1, 2, 3]
    with pytest.raises(ValueError):
        layer_schedule(4, (3, 3), 2)


def test_forward_shapes_and_pass_grouping():
    torch.manual_seed(0)
    model = dylomo.build_model(TINY_CLASS).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.final.weight, std=0.1)  # the output layer is zero-initialised
        for layer in model.layers:
            torch.nn.init.normal_(layer.adaLN_modulation[1].weight, std=0.1)
        x = torch.randn(4, 4, 16, 16)
        y = torch.tensor([1, 2, 3, 1000])
        t = torch.tensor([10, 900, 499, 500])
        out = model(x, t, y)
        assert out.shape == (4, 8, 16, 16)
        # samples with different pass counts in one batch equal the separate forward passes
        low, high = [0, 2], [1, 3]
        assert torch.allclose(out[low], model(x[low], t[low], y[low]), atol=1e-5)
        assert torch.allclose(out[high], model(x[high], t[high], y[high]), atol=1e-5)
        assert not torch.allclose(model(x[low], t[low], y[low], passes=1), out[low], atol=1e-4)

    t2i = dylomo.build_model(TINY_T2I).eval()
    with torch.no_grad():
        latent, text, sigma = torch.randn(2, 48, 4, 6), torch.randn(2, 5, 32), torch.tensor([0.3, 0.8])
        assert t2i(latent, text, sigma).shape == (2, 48, 4, 6)
        assert t2i(latent, text, sigma, text_lens=[0, 0]).shape == (2, 48, 4, 6)
        # every sample of a batch is processed independently, whatever the text lengths of the others
        mixed = t2i(latent, text, sigma, text_lens=[5, 2])
        single = t2i(latent[1:], text[1:], sigma[1:], text_lens=[2])
        assert torch.allclose(mixed[1:], single, atol=1e-4)


def test_guidance_scale_one_is_the_conditional_prediction():
    """Sampling without guidance evaluates the conditional branch only; with scale 1 the guided combination of
    the two branches is the same prediction up to rounding."""
    torch.manual_seed(0)
    model = dylomo.build_model(TINY_CLASS).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.final.weight, std=0.1)
        x, y = torch.randn(3, 4, 16, 16), torch.tensor([1, 2, 3])
        t = torch.full((6,), 400)
        single = model(x, t[:3], y)
        both = model.forward_with_cfg(torch.cat([x, x]), t, torch.cat([y, torch.full_like(y, 1000)]), 1.0)[:3]
        assert single.abs().max() > 0 and torch.allclose(single, both, atol=1e-5)
        guided = model.forward_with_cfg(torch.cat([x, x]), t, torch.cat([y, torch.full_like(y, 1000)]), 1.5)
        assert torch.equal(guided[:3, :3], guided[3:, :3])            # both halves carry the guided noise
        assert torch.allclose(guided[:3, 3:], single[:, 3:], atol=1e-5)  # unguided channels: conditional output


def test_routing():
    torch.manual_seed(0)
    core = SparseExperts(hidden=32, heads=2, experts=6, top_k=2, expert_hidden=8)
    x = torch.randn(10, 32)
    weights, index = core.route(x.view(10, 2, 16))
    assert weights.shape == (2, 10, 2) and index.shape == (2, 10, 2)
    assert torch.allclose(weights.sum(-1), torch.full((2, 10), core.route_scale))  # L1-normalised, times alpha
    # the routing bias is a buffer and takes part in the selection only
    assert "expert_bias" in dict(core.named_buffers()) and "expert_bias" not in dict(core.named_parameters())
    with torch.no_grad():
        core.expert_bias[3] = 10.0
    _, biased = core.route(x.view(10, 2, 16))
    assert (biased[0] == 3).any(-1).all() and not (index[0] == 3).any(-1).all()
    assert core(x).shape == (10, 32)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_grouped_matmul_loop_matches(device):
    torch.manual_seed(0)
    groups, d_in, d_out, rows = 7, 12, 20, 300
    weight = torch.randn(groups, d_in, d_out, device=device)
    group = torch.randint(0, groups - 1, (rows,), device=device).sort().values  # the last group stays empty
    ends = torch.bincount(group, minlength=groups).cumsum(0).to(torch.int32)
    x = torch.randn(rows, d_in, device=device)
    expected = torch.stack([x[i] @ weight[group[i]] for i in range(rows)])
    assert torch.allclose(grouped_matmul_loop(x, weight, ends), expected, atol=1e-5)
    assert torch.allclose(grouped_matmul(x, weight, ends), expected, atol=1e-5)


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    model = dylomo.build_model(TINY_T2I)
    dylomo.save_model(model, model.config, tmp_path / "single")
    dylomo.save_model(model, model.config, tmp_path / "sharded", max_shard_bytes=200_000)
    assert (tmp_path / "single" / "model.safetensors").is_file()
    assert len(list((tmp_path / "sharded").glob("model-*-of-*.safetensors"))) > 1
    reference = model.state_dict()
    for name in ("single", "sharded"):
        loaded = dylomo.load_model(tmp_path / name)
        assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
        state = loaded.state_dict()
        assert state.keys() == reference.keys() and all(torch.equal(state[k], reference[k]) for k in reference)
    half = dylomo.load_model(tmp_path / "single", expert_dtype=torch.bfloat16)
    dtypes = {name: p.dtype for name, p in half.named_parameters()}
    assert all((dtype == torch.bfloat16) == (".core.w_" in name) for name, dtype in dtypes.items())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("sharded", [False, True])
def test_mixed_storage_bf16_inference(tmp_path, device, sharded):
    """Load real safetensors and compare nonzero predictions and Euler/APG trajectories bit for bit."""
    from dylomo.flow import sample_euler
    from dylomo.io import is_expert_weight
    from safetensors.torch import load_file

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for seed in (7, 42):
            torch.manual_seed(seed)
            source = dylomo.build_model(TINY_T2I)
            # Exercise all layers, including any initially zero output projections.
            with torch.no_grad():
                for tensor in source.state_dict().values():
                    if tensor.is_floating_point():
                        tensor.add_(0.03 * torch.randn_like(tensor))
            fp32 = source.state_dict()
            mixed = {name: tensor.to(torch.bfloat16) if is_expert_weight(name) else tensor
                     for name, tensor in fp32.items()}
            assert any(not torch.equal(mixed[n].float(), t) for n, t in fp32.items() if is_expert_weight(n))
            limit = 200_000 if sharded else None
            for label, state in (("fp32", fp32), ("mixed", mixed)):
                folder = tmp_path / str(seed) / label
                dylomo.save_model(state, TINY_T2I, folder, max_shard_bytes=limit)
                assert (folder / "model.safetensors.index.json").exists() == sharded
                for file in dylomo.io.weight_files(folder):
                    for name, tensor in load_file(str(file)).items():
                        expected = torch.bfloat16 if label == "mixed" and is_expert_weight(name) else torch.float32
                        assert tensor.dtype == expected, name
            baseline = dylomo.load_model(tmp_path / str(seed) / "fp32", device=device)
            latent = torch.randn(2, 48, 4, 6, device=device)
            text = torch.randn(2, 5, 32, device=device)
            sigma = torch.tensor([0.2, 0.8], device=device)

            def infer(model):
                with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                    predictions = [model(latent, text, sigma, text_lens=lens) for lens in ([5, 2], [0, 0])]
                    trajectory = []
                    sample_euler(model, text, [5, 2], latent, steps=3, autocast=True,
                                 step_callback=lambda x, _: trajectory.append(x.clone()))
                return predictions + trajectory

            expected = infer(baseline)
            assert all(torch.isfinite(t).all() and t.abs().max() > 0 for t in expected)
            for override in (None, torch.bfloat16, torch.float32):
                loaded = dylomo.load_model(tmp_path / str(seed) / "mixed", device=device, expert_dtype=override)
                for name, tensor in loaded.state_dict().items():
                    want = (override or torch.bfloat16) if is_expert_weight(name) else torch.float32
                    assert tensor.dtype == want, name
                actual = infer(loaded)
                for reference, output in zip(expected, actual):
                    assert reference.dtype == output.dtype
                    assert torch.equal(reference.contiguous().view(torch.uint8),
                                       output.contiguous().view(torch.uint8)), (device, sharded, seed, override)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_text_prompt_template():
    from dylomo import text_encoder
    assert len(text_encoder.SYSTEM_PROMPT) == 305  # the hash of the exact string is asserted at import time


def test_state_dict_has_no_training_state():
    model = dylomo.build_model(TINY_CLASS)
    buffers = [name for name, _ in model.named_buffers()]
    constants = {"rope.bands", "t_embedder.frequencies"}  # derived in the constructor, not stored
    assert all(name.endswith("core.expert_bias") or name in constants for name in buffers)
    assert not constants & model.state_dict().keys()
    json.dumps(model.config)  # the config is plain JSON
