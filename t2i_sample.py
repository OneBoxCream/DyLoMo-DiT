"""Text-to-image sampling with the DyLoMo-DiT text-to-image checkpoint.

    python t2i_sample.py --model checkpoints/dylomo-dit-t2i --prompt "a photo of a red fox in the snow" \
        --height 1024 --width 1024 --seed 0 1 --output outputs/

NOTE: the released text-to-image weights are an intermediate checkpoint, not a final model. Defaults are the
sampling settings of the paper (50 Euler steps, shift 1.6, APG with guidance scale 5).

Components (downloaded from the Hugging Face Hub unless local paths are given):
  text encoder  Qwen/Qwen2.5-VL-7B-Instruct (language model only)
  VAE           Wan2.2 VAE (the `vae` sub-folder of Wan-AI/Wan2.2-TI2V-5B-Diffusers)
"""

import argparse
import re
from pathlib import Path

import torch

from dylomo.pipeline_t2i import DyLoMoT2IPipeline
from dylomo.text_encoder import TEXT_ENCODER
from dylomo.vae import WAN_VAE


def slug(text: str, limit: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "prompt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="directory with config.json and the safetensors weights")
    parser.add_argument("--prompt", nargs="+", required=True, help="one or more prompts")
    parser.add_argument("--height", type=int, default=1024, help="multiple of 16")
    parser.add_argument("--width", type=int, default=1024, help="multiple of 16")
    parser.add_argument("--seed", type=int, nargs="+", default=[0], help="one image per prompt and seed")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=1.6)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--guidance", choices=["apg", "cfg"], default="apg")
    parser.add_argument("--apg-momentum", type=float, default=-0.5)
    parser.add_argument("--apg-eta", type=float, default=0.15)
    parser.add_argument("--text-encoder", default=TEXT_ENCODER, help="hub ID or local directory")
    parser.add_argument("--vae", default=WAN_VAE, help="hub ID or local directory of the Wan2.2 VAE")
    parser.add_argument("--vae-subfolder", default="vae", help="sub-folder of --vae that holds the VAE ('' = none)")
    parser.add_argument("--expert-dtype", choices=["bf16", "fp32"], default="bf16",
                        help="storage dtype of the routed-expert weights on the GPU (bf16 takes half the space and "
                             "gives the same result under bf16 autocast)")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = DyLoMoT2IPipeline.from_pretrained(
        args.model, text_encoder=args.text_encoder, vae=args.vae, device=args.device,
        expert_dtype=torch.bfloat16 if args.expert_dtype == "bf16" else None, vae_subfolder=args.vae_subfolder or None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(args.prompt):
        images = pipeline(prompt, height=args.height, width=args.width, steps=args.steps, shift=args.shift,
                          guidance_scale=args.guidance_scale, guidance=args.guidance,
                          apg_momentum=args.apg_momentum, apg_eta=args.apg_eta, seeds=args.seed)
        for seed, image in zip(args.seed, images):
            path = output / f"{index:03d}_{slug(prompt)}_seed{seed}.png"
            image.save(path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
