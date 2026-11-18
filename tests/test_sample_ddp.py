"""Resuming ``sample_ddp.py``: finished shards are reused only by a run with the same weights, VAE and settings."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import dylomo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sample_ddp  # noqa: E402

TINY = dict(model_type="dylomo-dit-class-cond", hidden_size=64, num_layers=4, attention_head_dim=32, loop_span=[1, 3],
            pass_schedule=dict(thresholds=[500], passes=[3, 1]), routing_heads=2, experts_per_head=4, top_k=2,
            expert_hidden_size=16, shared_expert_hidden_size=24, dense_ffn_hidden_size=64)


class StandInVAE(torch.nn.Module):
    """Has weights (so that it has an identity) and decodes a latent to a flat image."""

    config = {"scaling_factor": 0.18215}

    def __init__(self, seed: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, generator=torch.Generator().manual_seed(seed)))


def save_model(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = dylomo.build_model(TINY)
    with torch.no_grad():
        torch.nn.init.normal_(model.final.weight, std=0.1)
    dylomo.save_model(model, TINY, path)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs ``sample_ddp.main`` with a tiny model and a stand-in VAE; returns the number of sampler calls."""
    save_model(tmp_path / "model", seed=1)
    calls = []
    sampler = sample_ddp.sample_class_conditional

    def counting_sampler(*args, **kwargs):
        calls.append(list(args[1]))
        return sampler(*args, **kwargs)

    def decode(vae, latents):
        value = (latents.float().mean(dim=(1, 2, 3)) * 50 + 128 + vae.weight.sum()).clamp(0, 255).to(torch.uint8)
        return value.view(-1, 1, 1, 1).expand(-1, 256, 256, 3).cpu().numpy().copy()

    monkeypatch.setattr(sample_ddp, "sample_class_conditional", counting_sampler)
    monkeypatch.setattr(sample_ddp, "load_sd_vae", lambda name, device: StandInVAE(int(name)))
    monkeypatch.setattr(sample_ddp, "decode_sd_latents", decode)
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    def invoke(*extra, vae="1", output="out", samples="6"):
        calls.clear()
        sample_ddp.main(["--model", str(tmp_path / "model"), "--vae", vae, "--output", str(tmp_path / output),
                         "--num-samples", samples, "--batch-size", "2", "--steps", "2", "--device", "cpu", *extra])
        return len(calls)

    invoke.root = tmp_path
    return invoke


def test_resume_reuses_only_its_own_shards(run):
    out = run.root / "out"
    assert run() == 3  # six samples in shards of two
    first = np.load(out / "samples.npz")
    images, labels = first["arr_0"], first["arr_1"]
    assert images.shape == (6, 256, 256, 3) and images.dtype == np.uint8 and labels.shape == (6,)
    record = json.loads((out / "sampling.json").read_text())
    assert set(record["weights_sha256"]) == {"model.safetensors"} and len(record["vae"]["weights_sha256"]) == 64

    assert run() == 0  # nothing is sampled again ...
    again = np.load(out / "samples.npz")
    assert np.array_equal(again["arr_0"], images) and np.array_equal(again["arr_1"], labels)

    (out / "shard-000001.npz").unlink()  # ... and an interrupted run only fills the gap
    assert run() == 1
    assert np.array_equal(np.load(out / "samples.npz")["arr_0"], images)
    sample_ddp.main(["--output", str(out), "--pack-only"])
    assert np.array_equal(np.load(out / "samples.npz")["arr_0"], images)


def test_other_weights_of_the_same_size_are_refused(run):
    assert run() == 3
    before = (run.root / "out" / "samples.npz").read_bytes()
    size = (run.root / "model" / "model.safetensors").stat().st_size
    save_model(run.root / "other", seed=2)
    assert (run.root / "other" / "model.safetensors").stat().st_size == size  # same architecture, same file size
    shutil.copy(run.root / "other" / "model.safetensors", run.root / "model" / "model.safetensors")
    with pytest.raises(SystemExit, match="weights_sha256"):
        run()
    assert (run.root / "out" / "samples.npz").read_bytes() == before
    assert run(output="fresh") == 3  # a new directory works


def test_other_vae_or_settings_are_refused(run):
    assert run() == 3
    with pytest.raises(SystemExit, match="vae"):
        run(vae="2")
    with pytest.raises(SystemExit, match="num_samples"):
        run(samples="8")
    with pytest.raises(SystemExit, match="cfg_scale"):
        run("--cfg-scale", "4.0")
    with pytest.raises(SystemExit, match="tf32"):
        run("--no-tf32")
    assert run() == 0  # the original run is still intact


def test_foreign_or_damaged_shards_are_refused(run):
    assert run() == 3
    out = run.root / "out"
    shard = out / "shard-000002.npz"
    data = dict(np.load(shard))

    np.savez(shard, **{**data, "ids": data["ids"] + 100})  # other sample IDs
    with pytest.raises(SystemExit, match="other sample IDs"):
        run()
    np.savez(shard, **{**data, "record": np.array("0" * 64)})  # written under another record
    with pytest.raises(SystemExit, match="different sampling.json"):
        run()
    with pytest.raises(ValueError, match="different sampling.json"):
        sample_ddp.pack(out, json.loads((out / "sampling.json").read_text()))
    np.savez(shard, **data)
    assert run() == 0

    (out / "shard-r0000-b000000.npz").write_bytes(b"")  # a shard this run would never write
    with pytest.raises(SystemExit, match="do not belong"):
        run()


def test_shards_without_a_record_are_refused(run):
    assert run() == 3
    (run.root / "out" / "sampling.json").unlink()
    with pytest.raises(SystemExit, match="no sampling.json"):
        run()
