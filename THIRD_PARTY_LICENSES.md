# Third-party material and licenses

This file separates what is part of this repository from what is only downloaded or installed when the code is used.

## 1. Material contained in this repository

| Material | Where | Origin | License |
|---|---|---|---|
| Adapted model components (clamped SwiGLU, Fourier rotary embedding and coordinate layout, routing rule and layout of the multi-head sparse experts, attention sink and per-head gate, gate form of the multi-stream residual) | `dylomo/layers.py`, `dylomo/attention.py`, `dylomo/moe.py`, `dylomo/residual.py` | [MAGI-2 preview](https://github.com/SandAI-org/MAGI-2-preview/tree/073c84f2102ec3c9287623113a103c14402770ad), Copyright (c) 2025-2026 SandAI | Apache License 2.0 |
| DDPM sampling rules (timestep respacing, learned-range variance) and the sinusoidal timestep embedding | `dylomo/diffusion.py`, `dylomo/models/class_cond.py` | [improved-diffusion](https://github.com/openai/improved-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), GLIDE; Copyright (c) 2021 OpenAI | MIT, full text in [LICENSES/MIT-OpenAI.txt](LICENSES/MIT-OpenAI.txt) |
| System instruction of the text encoder (prompt template `li-dit-encode-image-json`) | `dylomo/text_encoder.py` | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | [License of HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE) |

See [NOTICE](NOTICE) for the attribution statements. The class-conditional protocol follows
[DiT](https://github.com/facebookresearch/DiT); no source file of DiT is included.

## 2. Models and tools downloaded by the user (not distributed here)

| Component | Used for | License stated by its publisher |
|---|---|---|
| [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) | VAE of the class-conditional models | MIT |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | text encoder of the text-to-image model | Apache-2.0 |
| [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (`vae` sub-folder) | VAE of the text-to-image model | Apache-2.0 |
| [ADM evaluator](https://github.com/openai/guided-diffusion/tree/main/evaluations) and its reference batch | FID / sFID / IS / precision / recall | MIT |

## 3. Python packages installed from `requirements.txt` (not distributed here)

The table lists every package of the tested environment, direct and indirect dependencies alike: the packages that
`pip install -r requirements.txt` resolved to on Linux x86-64 with Python 3.12 (the same set as
`requirements.lock.txt`). The license column repeats what each package declares in its own metadata; consult the
package for the authoritative text. The PyTorch wheel for Linux pulls in the NVIDIA CUDA runtime libraries, which are
proprietary; a CPU-only PyTorch build does not need them.

<!-- BEGIN PACKAGE TABLE -->
| Package | Version | License (as declared by the package) | Required | Project |
|---|---|---|---|---|
| anyio | 4.16.0 | MIT | indirect (httpx) | https://github.com/agronholm/anyio |
| certifi | 2026.10.23 | MPL-2.0 | indirect (httpcore, httpx, requests) | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | MIT | indirect (requests) | https://charset-normalizer.readthedocs.io/ |
| cuda-bindings | 13.5.0 | Apache-2.0 | indirect (torch) | https://github.com/NVIDIA/cuda-python |
| cuda-pathfinder | 1.9.0 | Apache-2.0 | indirect (cuda-bindings) | https://github.com/NVIDIA/cuda-python |
| cuda-toolkit | 13.0.2 | not declared in the package metadata | indirect (torch) | https://developer.nvidia.com/cuda-toolkit |
| diffusers | 0.38.0 | Apache 2.0 License | direct | https://github.com/huggingface/diffusers |
| filelock | 4.2.12 | MIT | indirect (diffusers, huggingface-hub, torch, transformers) | https://github.com/tox-dev/filelock |
| fsspec | 2026.9.0 | BSD-3-Clause | indirect (huggingface-hub, torch) | https://github.com/fsspec/filesystem_spec |
| h11 | 0.16.0 | MIT | indirect (httpcore) | https://github.com/python-hyper/h11 |
| hf-xet | 1.8.0 | Apache-2.0 | indirect (huggingface-hub) | https://github.com/huggingface/xet-core |
| httpcore | 1.0.9 | BSD-3-Clause | indirect (httpx) | https://www.encode.io/httpcore/ |
| httpx | 0.28.1 | BSD-3-Clause | indirect (diffusers) | https://github.com/encode/httpx |
| huggingface_hub | 0.36.2 | Apache | indirect (diffusers, tokenizers, transformers) | https://github.com/huggingface/huggingface_hub |
| idna | 3.20 | BSD-3-Clause | indirect (anyio, httpx, requests) | https://github.com/kjd/idna |
| importlib_metadata | 10.2.0 | Apache-2.0 | indirect (diffusers) | https://github.com/python/importlib_metadata |
| Jinja2 | 3.1.6 | BSD License | indirect (torch) | https://github.com/pallets/jinja/ |
| lazy-importer | 0.1.0 | Apache-2.0 | indirect (importlib-metadata) |  |
| MarkupSafe | 3.0.4 | BSD-3-Clause | indirect (jinja2) | https://github.com/pallets/markupsafe/ |
| mpmath | 1.3.0 | BSD | indirect (sympy) | https://github.com/fredrik-johansson/mpmath |
| networkx | 3.7 | BSD-3-Clause | indirect (torch) | https://networkx.org/ |
| numpy | 2.1.0 | BSD License | direct | https://numpy.org |
| nvidia-cublas | 13.1.0.3 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit, nvidia-cudnn-cu13, nvidia-cusolver) | https://developer.nvidia.com/cuda-zone |
| nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-cuda-runtime | 13.0.96 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-cudnn-cu13 | 9.19.0.56 | LicenseRef-NVIDIA-Proprietary | indirect (torch) | https://developer.nvidia.com/cuda-zone |
| nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit, nvidia-cusolver) | https://developer.nvidia.com/cuda-zone |
| nvidia-cusparselt-cu13 | 0.8.0 | NVIDIA Proprietary Software | indirect (torch) | https://developer.nvidia.com/cusparselt |
| nvidia-nccl-cu13 | 2.28.9 | LicenseRef-NVIDIA-Proprietary | indirect (torch) | https://developer.nvidia.com/cuda-zone |
| nvidia-nvjitlink | 13.0.88 | LicenseRef-NVIDIA-Proprietary | indirect (cuda-toolkit, nvidia-cufft, nvidia-cusolver, nvidia-cusparse) | https://developer.nvidia.com/cuda-zone |
| nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary | indirect (torch) | https://developer.nvidia.com/cuda-zone |
| nvidia-nvtx | 13.0.85 | Apache 2.0 | indirect (cuda-toolkit) | https://developer.nvidia.com/cuda-zone |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | indirect (huggingface-hub, transformers) | https://github.com/pypa/packaging |
| pillow | 12.1.1 | MIT-CMU | direct | https://python-pillow.github.io |
| PyYAML | 6.0.3 | MIT | indirect (huggingface-hub, transformers) | https://github.com/yaml/pyyaml |
| regex | 2026.11.18 | Apache-2.0 AND CNRI-Python | indirect (diffusers, transformers) | https://github.com/mrabarnett/mrab-regex |
| requests | 2.34.2 | Apache-2.0 | indirect (diffusers, huggingface-hub, transformers) | https://github.com/psf/requests |
| safetensors | 0.8.0 | Apache Software License | direct | https://github.com/huggingface/safetensors |
| setuptools | 81.0.0 | MIT | indirect (torch) | https://github.com/pypa/setuptools |
| sympy | 1.14.0 | BSD | indirect (torch) | https://github.com/sympy/sympy |
| tokenizers | 0.22.2 | Apache Software License | indirect (transformers) | https://github.com/huggingface/tokenizers |
| torch | 2.11.0 | BSD-3-Clause | direct | https://pytorch.org |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | indirect (huggingface-hub, transformers) | https://tqdm.github.io |
| transformers | 4.57.1 | Apache 2.0 License | direct | https://github.com/huggingface/transformers |
| triton | 3.6.0 | MIT License | indirect (torch) | https://github.com/triton-lang/triton/ |
| typing_extensions | 4.16.0 | PSF-2.0 | indirect (anyio, huggingface-hub, torch) | https://github.com/python/typing_extensions |
| urllib3 | 2.8.0 | MIT | indirect (requests) | https://urllib3.readthedocs.io |
| zipp | 4.1.2 | MIT | indirect (importlib-metadata) | https://github.com/jaraco/zipp |
<!-- END PACKAGE TABLE -->
