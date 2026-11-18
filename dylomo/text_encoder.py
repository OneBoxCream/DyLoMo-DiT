"""Text conditioning of the text-to-image model: features of a frozen Qwen2.5-VL-7B language model.

The prompt is wrapped in a chat template with a fixed system instruction; the features are the hidden states of the
third-to-last layer (before the final normalisation) at the positions that follow the user marker, i.e. the prompt
tokens and the closing template tokens. System-instruction positions are dropped. At most ``max_length`` (1,000)
positions are kept.

The model was trained with this system instruction, so it must be reproduced exactly (including its whitespace).
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import torch

TEXT_ENCODER = "Qwen/Qwen2.5-VL-7B-Instruct"
TEXT_CHANNELS = 3584

_GAP = " " * 9
# Prompt template "li-dit-encode-image-json" of HunyuanVideo-1.5 (https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5).
SYSTEM_PROMPT = _GAP.join([
    "You are a helpful assistant. Describe the image by detailing the following aspects:",
    "1. The main content and theme of the image.",
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.",
    "3. The background environment, light, style and atmosphere.",
])
SYSTEM_PROMPT_SHA256 = "6a366a253ab1d9010dccb59802980fae74e737f81a8c63af79a22a18f139aed5"
assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == SYSTEM_PROMPT_SHA256
USER_MARKER = "<|im_start|>user\n"


def _language_model(model):
    """The text decoder of a Qwen2.5-VL checkpoint (the vision tower is not used)."""
    for path in ("language_model", "model.language_model"):
        module = model
        try:
            for name in path.split("."):
                module = getattr(module, name)
            return module
        except AttributeError:
            continue
    return model


class QwenTextEncoder:
    def __init__(self, name_or_path: str = TEXT_ENCODER, device="cuda", dtype: torch.dtype = torch.bfloat16,
                 max_length: int = 1000, skip_layers: int = 2, pad_to_max_length: bool = True):
        from transformers import AutoModel, AutoTokenizer

        if not 1 <= max_length <= 1000:
            raise ValueError("max_length must be in 1 .. 1000")
        self.device, self.max_length, self.skip_layers = torch.device(device), max_length, skip_layers
        # Padding every prompt to the full length reproduces the training-time tokenisation; with a causal language
        # model the padding does not change the kept positions, so ``False`` only saves time.
        self.pad_to_max_length = pad_to_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, padding_side="right")
        # the checkpoint is stored in bf16; loading it in the target dtype avoids an fp32 copy in host memory
        model = AutoModel.from_pretrained(name_or_path, low_cpu_mem_usage=True, dtype=dtype)
        self.model = _language_model(model).to(dtype=dtype).eval().requires_grad_(False).to(self.device)
        del model
        self.crop_start = self._find_crop_start()

    def _conversation(self, prompt: str):
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt if prompt else " "}]

    def _tokenize(self, prompts: Sequence[str], max_length: int, padding):
        return self.tokenizer.apply_chat_template([self._conversation(p) for p in prompts], add_generation_prompt=True,
                                                  tokenize=True, return_dict=True, truncation=True,
                                                  max_length=max_length, padding=padding, return_tensors="pt")

    def _find_crop_start(self) -> int:
        """Index of the first position after the user marker (constant: the system instruction is fixed)."""
        input_ids = self._tokenize(["a photo of a cat"], 256, "max_length")["input_ids"][0].tolist()
        marker = self.tokenizer(USER_MARKER, add_special_tokens=False)["input_ids"]
        for i in range(len(input_ids) - len(marker) + 1):
            if input_ids[i:i + len(marker)] == marker:
                return i + len(marker)
        raise ValueError("the chat template of this tokenizer does not contain the expected user marker")

    @torch.no_grad()
    def __call__(self, prompts: Sequence[str]) -> List[torch.Tensor]:
        """One feature matrix ``[L_i, 3584]`` per prompt (``L_i`` <= ``max_length``)."""
        padding = "max_length" if self.pad_to_max_length else True
        tokens = self._tokenize(list(prompts), self.max_length + self.crop_start, padding)
        attention_mask = tokens["attention_mask"].to(self.device)
        outputs = self.model(input_ids=tokens["input_ids"].to(self.device), attention_mask=attention_mask,
                             output_hidden_states=True)
        hidden = outputs.hidden_states[-(self.skip_layers + 1)][:, self.crop_start:]
        mask = attention_mask[:, self.crop_start:].bool()
        features = [hidden[i][mask[i]].detach() for i in range(hidden.shape[0])]
        for feature in features:
            if feature.ndim != 2 or feature.shape[-1] != TEXT_CHANNELS or not 0 < feature.shape[0] <= self.max_length:
                raise ValueError(f"unexpected text feature shape {tuple(feature.shape)}")
            if not torch.isfinite(feature).all():
                raise FloatingPointError("non-finite text features")
        return features
