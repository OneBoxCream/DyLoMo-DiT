"""Minimal text-to-image pipeline: Qwen2.5-VL text features -> DyLoMo-DiT denoiser (Euler, APG) -> Wan2.2 VAE."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from . import io
from .flow import sample_euler
from .text_encoder import TEXT_ENCODER, QwenTextEncoder
from .vae import WAN_VAE, WAN_VAE_DOWNSAMPLE, decode_wan_latents, load_wan_vae


def make_noise(seed: int, channels: int, height: int, width: int) -> torch.Tensor:
    """Initial noise ``[1, C, height / 16, width / 16]``: fp32, drawn on the CPU so that it is device-independent."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, channels, height // WAN_VAE_DOWNSAMPLE, width // WAN_VAE_DOWNSAMPLE),
                       generator=generator, dtype=torch.float32)


class DyLoMoT2IPipeline:
    """``pipeline(prompt, height=1024, width=1024)`` returns a list of PIL images.

    Defaults are the sampling settings of the paper: 50 Euler steps, shift 1.6, APG with guidance scale 5,
    momentum -0.5 and eta 0.15, the empty text as the unconditional branch, no negative prompt.
    """

    def __init__(self, denoiser, text_encoder, vae):
        self.denoiser, self.text_encoder, self.vae = denoiser, text_encoder, vae

    @classmethod
    def from_pretrained(cls, model_dir: str, text_encoder: str = TEXT_ENCODER, vae: str = WAN_VAE, device="cuda",
                        expert_dtype: Optional[torch.dtype] = torch.bfloat16, vae_subfolder: Optional[str] = "vae"):
        """``expert_dtype=torch.bfloat16`` keeps the routed-expert weights (most of the parameters) in bf16, which
        halves their storage and leaves the bf16-autocast result unchanged; pass ``None`` to keep them in fp32."""
        device = torch.device(device)
        denoiser = io.load_model(model_dir, device=device, expert_dtype=expert_dtype if device.type == "cuda" else None)
        if denoiser.config["model_type"] != "dylomo-dit-t2i":
            raise ValueError("not a text-to-image DyLoMo-DiT checkpoint")
        return cls(denoiser, QwenTextEncoder(text_encoder, device=device),
                   load_wan_vae(vae, device=device, subfolder=vae_subfolder))

    @torch.no_grad()
    def generate_latents(self, prompt: str, height: int = 1024, width: int = 1024, steps: int = 50, shift: float = 1.6,
                         guidance_scale: float = 5.0, guidance: str = "apg", apg_momentum: float = -0.5,
                         apg_eta: float = 0.15, seeds: Sequence[int] = (0,)) -> torch.Tensor:
        if height % WAN_VAE_DOWNSAMPLE or width % WAN_VAE_DOWNSAMPLE or min(height, width) < WAN_VAE_DOWNSAMPLE:
            raise ValueError("height and width must be multiples of 16")
        device = next(self.denoiser.parameters()).device
        text = self.text_encoder([prompt])[0].unsqueeze(0).to(device)  # [1, L, D]
        latents = []
        for seed in seeds:  # one image per call: the APG running average is kept per image
            noise = make_noise(seed, self.denoiser.latent_channels, height, width).to(device)
            latents.append(sample_euler(self.denoiser, text, [text.shape[1]], noise, steps=steps, shift=shift,
                                        guidance_scale=guidance_scale, guidance=guidance, apg_momentum=apg_momentum,
                                        apg_eta=apg_eta, autocast=device.type == "cuda"))
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def __call__(self, prompt: Union[str, Sequence[str]], **kwargs) -> List:
        from PIL import Image

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        images = []
        for text in prompts:
            for pixels in decode_wan_latents(self.vae, self.generate_latents(text, **kwargs)):
                images.append(Image.fromarray(pixels))
        return images
