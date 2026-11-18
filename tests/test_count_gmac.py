"""``tools/count_gmac.py``: the counts of the released configurations and the self-check that guards them."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import count_gmac  # noqa: E402

TINY_T2I = dict(model_type="dylomo-dit-t2i", hidden_size=64, num_layers=4, attention_head_dim=32, loop_span=[1, 3],
                pass_schedule=dict(thresholds=[], passes=[2]), routing_heads=2, experts_per_head=8, top_k=2,
                expert_hidden_size=16, shared_expert_hidden_size=24, dense_ffn_hidden_size=64, residual_streams=4,
                latent_channels=48, text_channels=32, time_channels=16)


def test_released_class_conditional_models_match_the_paper():
    results = count_gmac.main(["--paper", "--skip-t2i", "--threads", "2"])
    assert {name: round(result["gmac"], 1) for name, result in results.items()} == {
        "dylomo_dit_b": 23.1, "dylomo_dit_l": 80.1, "dylomo_dit_xl": 117.8}
    for result in results.values():
        assert result["checked_against_paper"] and result["gmac_attention"] > 0
        assert result["mean_passes"] == 2.0
        low, high = result["gmac_by_passes"][3], result["gmac_by_passes"][1]
        assert abs((low + high) / 2 - result["gmac"]) < 1e-9  # three passes for half of the steps, one for the rest


def test_attention_is_counted_from_the_call_shapes():
    """One image of 4 x 6 latent positions and 5 text tokens: 29 tokens, 2 attention heads of size 32, six layer
    visits (1 + 2 x 2 + 1). Every visit computes QK^T and PV against 29 keys plus the sink, with 8 extra channels."""
    result = count_gmac.t2i_report(TINY_T2I, height=64, width=96, text_tokens=5)
    assert result["image_tokens"] == 24
    assert round(result["gmac_attention"] * 1e9) == 6 * 2 * 29 * 30 * (40 + 40)
    assert result["gmac"] > result["gmac_attention"] > 0


def test_self_check_fails_on_a_wrong_total(monkeypatch, capsys):
    monkeypatch.setitem(count_gmac.PAPER_GMAC, "dylomo_dit_b", (23.2, 1))
    with pytest.raises(SystemExit, match="self-check failed"):
        count_gmac.main(["--config", str(ROOT / "configs" / "dylomo_dit_b.json"), "--threads", "2"])
    assert "dylomo_dit_b:" not in capsys.readouterr().out  # no number is printed when the check fails


def test_self_check_fails_without_attention():
    with pytest.raises(count_gmac.SelfCheckError, match="attention"):
        count_gmac.count_macs(lambda: torch.zeros(2, 3) @ torch.zeros(3, 4))


def test_other_settings_are_not_compared_with_the_paper():
    result = count_gmac.main(["--config", str(ROOT / "configs" / "dylomo_dit_b.json"), "--steps", "50", "--threads", "2"])
    assert not result["dylomo_dit_b"]["checked_against_paper"]
