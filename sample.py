"""Sample class-conditional images from a DyLoMo-DiT checkpoint and save them as a grid.

    python sample.py --model checkpoints/dylomo-dit-xl --classes 360 207 88 279 839 967 --cfg-scale 4.0

Uses the evaluation sampler (250-step DDPM, classifier-free guidance on the first three latent channels); the number
of passes through the looped segment follows the pass schedule stored in the model config.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dylomo
from dylomo.diffusion import sample_class_conditional
from dylomo.vae import SD_VAE, decode_sd_latents, load_sd_vae


def make_grid(images: np.ndarray, columns: int) -> Image.Image:
    count, height, width, _ = images.shape
    rows = (count + columns - 1) // columns
    grid = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)
    for i, image in enumerate(images):
        row, column = divmod(i, columns)
        grid[row * height:(row + 1) * height, column * width:(column + 1) * width] = image
    return Image.fromarray(grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="directory with config.json and model.safetensors")
    parser.add_argument("--vae", default=SD_VAE, help="sd-vae-ft-ema: hub ID or local directory")
    parser.add_argument("--classes", type=int, nargs="+", default=[360, 207, 88, 279, 839, 967],
                        help="ImageNet class indices (0 .. 999), one image each (default: otter, golden retriever, "
                             "macaw, arctic fox, suspension bridge, espresso)")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="guidance scale (1.5 = evaluation setting)")
    parser.add_argument("--steps", type=int, default=250, help="DDPM sampling steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--output", default="sample.png")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="TF32 for matrix products on CUDA (on by default, as in the evaluation of the paper); "
                             "--no-tf32 turns it off for matrix products only, the cuDNN setting is not touched")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    model = dylomo.load_model(args.model, device=args.device)
    vae = load_sd_vae(args.vae, device=args.device)
    latents, _ = sample_class_conditional(model, ids=list(range(len(args.classes))), seed=args.seed, steps=args.steps,
                                          cfg_scale=args.cfg_scale, device=args.device,
                                          latent_size=args.image_size // 8, labels=args.classes)
    images = decode_sd_latents(vae, latents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_grid(images, args.columns).save(output)
    print(f"saved {len(images)} images to {output}")


if __name__ == "__main__":
    main()
