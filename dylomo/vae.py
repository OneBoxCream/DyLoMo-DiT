"""VAE helpers. Both VAEs are public models loaded through ``diffusers``.

* class-conditional models: ``stabilityai/sd-vae-ft-ema`` (4 latent channels, 8x downsampling, scaling 0.18215)
* text-to-image model: the Wan2.2 VAE (48 latent channels, 16x downsampling), e.g. the ``vae`` sub-folder of
  ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``
"""

from __future__ import annotations

import numpy as np
import torch

SD_VAE = "stabilityai/sd-vae-ft-ema"
SD_VAE_SCALING = 0.18215
WAN_VAE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_VAE_DOWNSAMPLE = 16


def load_sd_vae(name_or_path: str = SD_VAE, device="cuda"):
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(name_or_path).eval().requires_grad_(False).to(device=device, dtype=torch.float32)


@torch.no_grad()
def decode_sd_latents(vae, latents: torch.Tensor) -> np.ndarray:
    """Latents ``[N, 4, h, w]`` (as sampled) -> images ``[N, 8h, 8w, 3]`` uint8.

    Pixel conversion of the FID protocol: ``[-1, 1]`` is mapped linearly to ``[0.5, 255.5]``, clipped to
    ``[0, 255]`` and truncated to integers.
    """
    images = vae.decode(latents.float() / SD_VAE_SCALING).sample
    pixels = images.mul(127.5).add(128.0).clamp(0, 255)
    return pixels.permute(0, 2, 3, 1).cpu().to(torch.uint8).numpy()


def load_wan_vae(name_or_path: str = WAN_VAE, device="cuda", dtype: torch.dtype = torch.bfloat16, subfolder="vae"):
    """Load the Wan2.2 VAE. ``name_or_path`` may be a hub ID / pipeline directory (with ``subfolder``) or a
    directory that directly contains the VAE (``subfolder=None``)."""
    from diffusers import AutoencoderKLWan

    kwargs = dict(subfolder=subfolder) if subfolder else {}
    vae = AutoencoderKLWan.from_pretrained(name_or_path, torch_dtype=dtype, **kwargs)
    if vae.config.z_dim != 48:
        raise ValueError(f"expected the 48-channel Wan2.2 VAE, got z_dim={vae.config.z_dim}")
    return vae.eval().requires_grad_(False).to(device)


@torch.no_grad()
def decode_wan_latents(vae, latents: torch.Tensor) -> np.ndarray:
    """Normalised latents ``[N, 48, h, w]`` -> images ``[N, 16h, 16w, 3]`` uint8."""
    device, dtype = vae.device, vae.dtype
    channels = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, channels, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device)).view(1, channels, 1, 1, 1)
    images = []
    for latent in latents:  # one image at a time keeps the memory of 1024x1024 decoding bounded
        z = latent.to(device=device, dtype=dtype)[None, :, None]  # [1, 48, 1, h, w]: a single latent frame
        z = z / inv_std.to(dtype) + mean.to(dtype)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            image = vae.decode(z).sample
        image = image.float().clamp_(-1, 1)[0, :, 0]  # [3, H, W]
        pixels = image.permute(1, 2, 0).add(1).mul(0.5).clamp(0, 1).mul(255).round().to(torch.uint8)
        images.append(pixels.cpu().numpy())
    return np.stack(images)
