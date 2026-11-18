"""Check an installation: package versions and the loading paths that the unit tests do not reach.

    python tools/check_install.py            # class-conditional path: model code and the sd-vae-ft-ema decoder
    python tools/check_install.py --t2i      # also the Wan2.2 VAE and the Qwen2.5-VL text encoder (large downloads)

``--vae``, ``--wan-vae`` and ``--text-encoder`` accept local directories instead of hub IDs. The script uses random
denoiser weights; it checks that everything loads and runs, not image quality. It exits with an error if a step
fails or if an installed package is outside the tested range of ``requirements.txt``.
"""

import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dylomo  # noqa: E402

TINY_CLASS = dict(model_type="dylomo-dit-class-cond", hidden_size=64, num_layers=4, attention_head_dim=32,
                  loop_span=[1, 3], pass_schedule=dict(thresholds=[500], passes=[3, 1]), routing_heads=2,
                  experts_per_head=4, top_k=2, expert_hidden_size=16, shared_expert_hidden_size=24,
                  dense_ffn_hidden_size=64)
TINY_T2I = dict(model_type="dylomo-dit-t2i", hidden_size=64, num_layers=4, attention_head_dim=32, loop_span=[1, 3],
                pass_schedule=dict(thresholds=[], passes=[2]), routing_heads=2, experts_per_head=8, top_k=2,
                expert_hidden_size=16, shared_expert_hidden_size=24, dense_ffn_hidden_size=64, residual_streams=4,
                latent_channels=48, text_channels=3584, time_channels=64)


def check_versions() -> bool:
    """Print the installed versions and compare them with the ranges of requirements.txt."""
    from packaging.requirements import Requirement

    ok = True
    print(f"python {sys.version.split()[0]}")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        installed = metadata.version(requirement.name)
        base = installed.split("+")[0]  # local build tags such as +cpu are not part of the requirement
        inside = requirement.specifier.contains(base, prereleases=True)
        ok &= inside
        print(f"{requirement.name} {installed}  [{'ok' if inside else 'OUTSIDE ' + str(requirement.specifier)}]")
    return ok


def check_models(device) -> None:
    from dylomo.diffusion import sample_class_conditional

    torch.manual_seed(0)
    model = dylomo.build_model(TINY_CLASS).to(device).eval()
    latents, labels = sample_class_conditional(model, ids=[0, 1], steps=3, cfg_scale=1.5, device=device, latent_size=8)
    assert latents.shape == (2, 4, 8, 8) and torch.isfinite(latents).all()
    print(f"model code: DDPM sampling runs on {device} (grouped product: "
          f"{'torch._grouped_mm' if hasattr(torch, '_grouped_mm') else 'per-expert loop'})")


def check_sd_vae(name: str, device) -> None:
    from dylomo.vae import decode_sd_latents, load_sd_vae

    vae = load_sd_vae(name, device=device)
    images = decode_sd_latents(vae, torch.zeros(1, 4, 32, 32, device=device))
    assert images.shape == (1, 256, 256, 3) and images.dtype.name == "uint8"
    print(f"sd-vae-ft-ema: loaded with diffusers {metadata.version('diffusers')}, decodes to {images.shape[1:]}")


def check_t2i(wan_vae: str, subfolder, text_encoder: str, device) -> None:
    from dylomo.flow import sample_euler
    from dylomo.text_encoder import QwenTextEncoder
    from dylomo.vae import decode_wan_latents, load_wan_vae

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = load_wan_vae(wan_vae, device=device, dtype=dtype, subfolder=subfolder)
    encoder = QwenTextEncoder(text_encoder, device=device, dtype=dtype, pad_to_max_length=False)
    features = encoder(["a photo of a cat"])[0]
    assert features.ndim == 2 and features.shape[1] == 3584 and torch.isfinite(features).all()
    print(f"Qwen2.5-VL text encoder: loaded with transformers {metadata.version('transformers')}, "
          f"features {tuple(features.shape)}, instruction prefix of {encoder.crop_start} tokens removed")

    torch.manual_seed(0)
    denoiser = dylomo.build_model(TINY_T2I).to(device).eval()
    noise = torch.randn(1, 48, 4, 6, generator=torch.Generator().manual_seed(0)).to(device)
    text = features.unsqueeze(0).to(device)
    latent = sample_euler(denoiser, text, [text.shape[1]], noise, steps=2, autocast=device.type == "cuda")
    images = decode_wan_latents(vae, latent)
    assert images.shape == (1, 64, 96, 3) and images.dtype.name == "uint8"
    print(f"Wan2.2 VAE: loaded with diffusers {metadata.version('diffusers')}, decodes to {images.shape[1:]}; "
          f"text encoder -> Euler sampling -> VAE runs end to end")


def main(argv=None):
    from dylomo.text_encoder import TEXT_ENCODER
    from dylomo.vae import SD_VAE, WAN_VAE

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t2i", action="store_true", help="also check the text-to-image components")
    parser.add_argument("--vae", default=SD_VAE, help="sd-vae-ft-ema: hub ID or local directory")
    parser.add_argument("--wan-vae", default=WAN_VAE, help="Wan2.2 VAE: hub ID or local directory")
    parser.add_argument("--wan-vae-subfolder", default="vae", help="sub-folder of --wan-vae ('' = none)")
    parser.add_argument("--text-encoder", default=TEXT_ENCODER, help="Qwen2.5-VL-7B-Instruct: hub ID or local directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    versions_ok = check_versions()
    check_models(device)
    check_sd_vae(args.vae, device)
    if args.t2i:
        check_t2i(args.wan_vae, args.wan_vae_subfolder or None, args.text_encoder, device)
    if not versions_ok:
        raise SystemExit("an installed package is outside the tested range of requirements.txt")
    print("installation check passed")


if __name__ == "__main__":
    main()
