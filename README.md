# DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers

Inference code and model definitions for **DyLoMo-DiT**.

[Paper](PAPER_URL_PLACEHOLDER) | [Project page](PROJECT_PAGE_URL_PLACEHOLDER) | [ImageNet weights](HF_IMAGENET_REPO_PLACEHOLDER) | [Text-to-image weights](HF_T2I_REPO_PLACEHOLDER)

DyLoMo-DiT is a diffusion transformer that makes three resources separately configurable: the parameters a token
uses (multi-head sparse experts in the feed-forward networks), the depth it traverses (a weight-shared loop over the
middle layers), and how that depth is spent across noise levels (more passes through the loop at low noise, fewer at
high noise). At the parameter and compute budget of DiT it lowers FID on ImageNet 256x256 by 11-24%.

This repository contains

* the model definitions (forward pass only) of the class-conditional models and of the text-to-image model,
* the ImageNet sampler and the 50k-sample script of the FID protocol (`sample.py`, `sample_ddp.py`),
* a minimal text-to-image pipeline (`t2i_sample.py`),
* tools for the compute accounting of the paper (`tools/count_gmac.py`) and an installation check
  (`tools/check_install.py`).

Training code is not part of this release.

## Released models

### Class-conditional, ImageNet 256x256

All models: 400k training steps unless noted, EMA weights, 250-step DDPM, classifier-free guidance 1.5, FID-50K with
the ADM evaluator.

| Model | Layers (looped) | Experts: heads x experts (selected) | Passes | Params total / active (M) | GMAC | FID |
|---|---|---|---|---|---|---|
| DyLoMo-DiT-B | 12 (6) | 2 x 17 (6) | 3 for t < 500, 1 for t >= 500 | 128.5 / 75.8 | 23.1 | 19.16 |
| DyLoMo-DiT-L | 12 (6) | 4 x 17 (6) | 3 for t < 500, 1 for t >= 500 | 451.3 / 278.3 | 80.1 | 6.25 |
| DyLoMo-DiT-XL | 16 (7) | 4 x 16 (6) | 3 for t < 500, 1 for t >= 500 | 686.6 / 411.3 | 117.8 | 4.34 |
| DyLoMo-DiT-XL, 1.6M steps | 16 (7) | 4 x 16 (6) | 3 for t < 500, 1 for t >= 500 | 686.6 / 411.3 | 117.8 | 2.18 |

For reference, the DiT baselines trained under the same protocol reach FID 21.19 (DiT-B/2), 7.42 (DiT-L/2) and
5.57 (DiT-XL/2) at 400k steps. GMAC is 10^9 multiply-accumulates of one forward pass, averaged over the 250 sampling
steps (see [Compute accounting](#compute-accounting)).

**All released ImageNet checkpoints use training seed 42.** The FIDs above describe those individual weights.
Table 2 of the paper reports means over two training seeds: B 18.82, L 6.41 and XL 4.22.
The 1.6M-step model continues the seed-42 XL run. The released checkpoints' full metrics use EMA weights,
250-step DDPM, guidance scale 1.5 and 50,000 generated images:

| Released checkpoint | Training seed | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|
| DyLoMo-DiT-B (400k) | 42 | 19.16 | 5.56 | 70.2 | 0.68 | 0.55 |
| DyLoMo-DiT-L (400k) | 42 | 6.25 | 4.46 | 142.0 | 0.80 | 0.53 |
| DyLoMo-DiT-XL (400k) | 42 | 4.34 | 4.30 | 172.4 | 0.82 | 0.53 |
| DyLoMo-DiT-XL (1.6M) | 42 | 2.18 | 4.28 | 253.0 | 0.83 | 0.58 |

### Text-to-image

> **The text-to-image weights are an intermediate checkpoint, not a final model.** They are released to make the
> text-to-image result of the paper reproducible.

| Model | Params total / active per image token | Looped layers | Experts | GenEval (1024x1024) |
|---|---|---|---|---|
| DyLoMo-DiT text-to-image | 15.48B / 1.05B | 32 layers, middle 16 run twice | 4 x 128 (6) | 0.714 |

Text encoder: Qwen2.5-VL-7B (frozen); VAE: Wan2.2 VAE (48 channels, 16x downsampling); training objective: rectified
flow; sampling: 50 Euler steps, shift 1.6, APG with guidance scale 5.

## Installation

```bash
git clone CODE_REPO_URL_PLACEHOLDER dylomo-dit && cd dylomo-dit
pip install -r requirements.txt
```

`requirements.txt` fixes the versions the code was tested with: PyTorch 2.5.1 to 2.11.0, diffusers 0.38.0,
transformers 4.57.1, safetensors 0.8.0, numpy 2.1.0 and pillow 12.1.1, on Python 3.12 (the class-conditional part
also on Python 3.10). Other combinations are not covered: PyTorch 2.4 cannot import this diffusers release, and newer
diffusers and transformers releases need a newer PyTorch. `requirements.lock.txt` lists the complete tested
environment, including indirect dependencies.

Check an installation with

```bash
python tools/check_install.py          # package versions, model code, sd-vae-ft-ema loading and decoding
python tools/check_install.py --t2i    # also the Wan2.2 VAE and the Qwen2.5-VL text encoder (large downloads)
```

The routed experts use `torch._grouped_mm` when the installed PyTorch provides it and fall back to a per-expert loop
otherwise; both give the same result.

Download a complete model folder from the weight repositories above, for example into
`checkpoints/dylomo-dit-xl`. ImageNet folders contain `config.json` + `model.safetensors`. The text-to-image folder
contains `config.json`, `model.safetensors.index.json` and every `model-*-of-*.safetensors` shard.
`dylomo.load_model()` reads either format and preserves the stored expert dtype by default.

## Class-conditional sampling

```bash
python sample.py --model checkpoints/dylomo-dit-xl --classes 360 207 88 279 839 967 --cfg-scale 4.0 --output sample.png
```

In Python:

```python
import torch, dylomo
from dylomo.diffusion import sample_class_conditional
from dylomo.vae import load_sd_vae, decode_sd_latents

model = dylomo.load_model("checkpoints/dylomo-dit-xl", device="cuda")
vae = load_sd_vae(device="cuda")
latents, labels = sample_class_conditional(model, ids=range(4), labels=[360, 207, 88, 279], cfg_scale=4.0)
images = decode_sd_latents(vae, latents)        # uint8, [4, 256, 256, 3]
```

The number of passes through the looped segment is part of the model: `model(x, t, y)` reads it from the timestep
through the pass schedule in `config.json` (three passes for `t < 500`, one for `t >= 500`, i.e. two on average over
the sampling steps). Nothing has to be set at sampling time, and the released weights should not be run under a
different schedule.

## Reproducing FID on ImageNet 256x256

Protocol of the paper: EMA weights, 250-step DDPM with the learned variance, classifier-free guidance scale 1.5
applied to the first three latent channels (as in DiT), fp32, `sd-vae-ft-ema` decoder, 50,000 samples with uniformly
drawn class labels, FID / sFID / IS / precision / recall computed with the TensorFlow evaluator of
[openai/guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations) against its ImageNet 256x256
reference batch.

1. Generate 50,000 images (one GPU, or `torchrun` for several):

   ```bash
   python sample_ddp.py --model checkpoints/dylomo-dit-xl --output runs/xl
   torchrun --nproc_per_node=8 sample_ddp.py --model checkpoints/dylomo-dit-xl --output runs/xl
   ```

   This writes `runs/xl/samples.npz` (`arr_0`: `[50000, 256, 256, 3]` uint8). The noise and the label of every image
   depend only on its sample ID, so the image set does not depend on the number of GPUs (up to floating-point
   rounding). Interrupted runs resume with the same command. The settings of a run, the SHA-256 of the weight files
   and a digest of the VAE weights are recorded in `runs/xl/sampling.json`; a run with other weights, another VAE or
   other settings refuses to reuse the directory, so use one output directory per model and setting.

2. Evaluate with the ADM evaluator (a separate TensorFlow environment is the easiest way):

   ```bash
   git clone https://github.com/openai/guided-diffusion
   pip install -r guided-diffusion/evaluations/requirements.txt
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   python guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz runs/xl/samples.npz
   ```

`--cfg-scale 1.0` samples without guidance. `--precision bf16` samples under bf16 autocast, which is not the
evaluation protocol.
On CUDA, TF32 is enabled for matrix products by default (`torch.backends.cuda.matmul.allow_tf32`), as in the
evaluation of the paper and in the sampling scripts of DiT. `--no-tf32` turns it off for matrix products only; the
cuDNN setting used by the convolutions of the VAE decoder is left at the PyTorch default.

## Text-to-image sampling

```bash
python t2i_sample.py --model checkpoints/dylomo-dit-t2i \
    --prompt "a red fox sitting in fresh snow, golden hour" --height 1024 --width 1024 --seed 0 1 --output outputs/
```

```python
from dylomo.pipeline_t2i import DyLoMoT2IPipeline

pipeline = DyLoMoT2IPipeline.from_pretrained("checkpoints/dylomo-dit-t2i", device="cuda")
image = pipeline("a red fox sitting in fresh snow, golden hour", height=1024, width=1024, seeds=[0])[0]
```

* The text encoder ([Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), language model
  only) and the VAE ([Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), `vae` sub-folder) are
  downloaded from the Hugging Face Hub; `--text-encoder` / `--vae` accept local paths.
* Defaults are the settings behind the GenEval number: 50 Euler steps, shift 1.6, APG with guidance scale 5,
  momentum -0.5 and eta 0.15, the empty text as unconditional branch, no negative prompt, prompts not rewritten.
  The gallery of the paper additionally uses a larger shift at high resolution (3.88 at 1024x1024, 4.87 at
  1808x1024 and 1024x1808); pass it with `--shift`.
* Height and width must be multiples of 16. The checkpoint comes from the training stage with short side 1024.
* The evaluation of the paper used fp32 weights under bf16 autocast. The released safetensors store routed-expert
  parameters in bf16 and all remaining parameters in fp32 (about 32 GB of weights, split into shards; this is not a
  peak-memory estimate). Loading preserves that mixed storage, and sampling uses bf16 autocast. Random-weight
  checks on a reduced configuration give bit-identical outputs to all-fp32 files under bf16 autocast on CPU and
  CUDA, using deterministic algorithms. `--expert-dtype fp32` widens the stored expert values; it cannot recover
  precision discarded when the files were saved.
* The GenEval number of the paper follows the official protocol: 553 prompts with 4 images each (2,212 images) at
  1024x1024, scored with the official implementation. `--seed` seeds a standard PyTorch generator for the initial
  noise; it does not reproduce the individual images of that evaluation.
* Every token only uses 6 of the 128 experts of each routing head, so the compute of a forward pass corresponds to the
  1.05B active parameters, not to the 15.48B stored ones.

This is a research checkpoint. It can produce inaccurate, biased or otherwise objectionable images; it has no safety
filter and should not be deployed without one.

## Model overview

```
x_t --patchify--> tokens --[ front layers | looped middle layers x r | back layers ]--> noise (+ variance)
                                              shared weights, residual factor 1/r
```

* **Multi-head sparse experts** (`dylomo/moe.py`). Except in the first and the last layer, the feed-forward network
  splits its input into `H` heads; each head has its own router and `E` experts and selects `k` of them per token
  (sigmoid scores, top-k after adding a routing bias, weights L1-normalised over the selected experts), plus one
  always-on shared expert. The routing bias is a buffer (`*.core.expert_bias`) that is loaded with the weights and
  is constant at inference time; the released files hold its value at the end of training, which is the one the
  evaluation used.
* **Weight-shared loop** (`dylomo/loop.py`). The middle `loop_span` layers are executed `r` times with the same
  weights; inside the loop every sub-layer writes to the residual with factor `1/r`. Routing is recomputed in every
  pass.
* **Passes by noise level.** `r` depends on the timestep through `pass_schedule` in the config:
  `{"thresholds": [500], "passes": [3, 1]}` means three passes for `t < 500` and one for `t >= 500`.
* **Class-conditional models** (`dylomo/models/class_cond.py`): adaLN-Zero conditioning on timestep + class, RMSNorm,
  rotary positions, attention with QK normalisation, a per-head output gate and an attention sink; the output layer
  predicts noise and variance (DDPM objective of DiT).
* **Text-to-image model** (`dylomo/models/t2i.py`): text and image tokens in one sequence, the diffusion time written
  into the first 64 input channels instead of adaLN, a 4-stream residual, separate weights for image and text tokens
  in the first and last layer, rectified-flow velocity prediction.

Configs of the released models are in `configs/`; parameter names and shapes are documented in
[docs/PARAMETERS.md](docs/PARAMETERS.md).

## Compute accounting

`tools/count_gmac.py` recomputes the GMAC columns of the paper: multiply-accumulates of one denoiser forward pass,
measured on the instantiated model (matrix products and expert products, including the router scoring, counted by
PyTorch's flop counter; attention products counted from the shapes of the attention calls).

```bash
python tools/count_gmac.py --paper --skip-t2i        # B / L / XL: 23.1, 80.1, 117.8 GMAC
python tools/count_gmac.py --config configs/dylomo_dit_t2i.json --height 1024 --width 1024 --text-tokens 13 --device cuda
                                                     # text-to-image at 1024x1024: 8,333 GMAC
```

The count does not depend on the weight values, so no checkpoint is needed. The script checks itself: it stops with
an error instead of printing a number if no attention product was counted, or if a released configuration does not
reproduce the value of the paper.

## Numerical notes

* The forward passes in this repository were checked bit for bit (fp32, TF32 off) against the implementation the
  weights were trained with, for the B / L / XL configurations and for the text-to-image architecture, on CPU and on
  CUDA, and so were the two samplers; `tests/test_reference_equivalence.py` is that check (it needs the original
  code and is skipped otherwise). `tests/test_golden.py` compares against outputs recorded from the original
  implementation and runs anywhere.
* On CUDA the scatter-add that sums the selected experts of a token is not deterministic, so two runs of the same
  model differ in the last bits. In our check (DyLoMo-DiT-B with random weights), the largest difference between two
  forward passes was about 2e-5 at an output magnitude of about 20. `sample_ddp.py --deterministic` switches PyTorch
  to deterministic algorithms, with which repeated runs on the same GPU gave bit-identical results.
* All of these checks used random weights. FID and GenEval of the released checkpoints have not been re-measured
  with this repository, and the effect of the non-determinism above, of `--deterministic` and of `--no-tf32` on the
  metrics has not been measured.
* The text-to-image evaluation in the paper was run with a multi-GPU implementation of the same computation (fp32
  weights, bf16 autocast); this repository provides the single-GPU reference path.
* Text features are computed with every prompt padded to the full length of 1,000 tokens, as during training
  (`QwenTextEncoder(pad_to_max_length=False)` skips the padding; its features differ by bf16 rounding).

## Tests

```bash
pip install pytest
pytest tests
python tools/check_install.py
```

## License

The code is released under **Apache-2.0**; see [LICENSE](LICENSE). Third-party attributions are in [NOTICE](NOTICE) and
[LICENSES/](LICENSES); the licenses of the Python packages the code depends on are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
Both the ImageNet and text-to-image weights are released under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). The class-conditional models are trained
on ImageNet-1k, whose terms of access restrict it to non-commercial research and educational use.

## Citation

```bibtex
@article{dylomodit,
  title   = {DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers},
  author  = {AUTHORS_PLACEHOLDER},
  journal = {arXiv preprint arXiv:ARXIV_ID_PLACEHOLDER},
  year    = {YEAR_PLACEHOLDER}
}
```
