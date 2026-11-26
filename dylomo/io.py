"""Configs and weights: ``config.json`` next to ``model.safetensors`` (or shards listed in an index file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from .models import DyLoMoDiT, DyLoMoT2I

MODEL_TYPES = {"dylomo-dit-class-cond": DyLoMoDiT, "dylomo-dit-t2i": DyLoMoT2I}
WEIGHTS_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"


def load_config(path: Union[str, Path]) -> Dict:
    """Read a config from a JSON file or from the ``config.json`` of a model directory."""
    path = Path(path)
    if path.is_dir():
        path = path / CONFIG_NAME
    config = json.loads(path.read_text())
    if "model_type" not in config and isinstance(config.get("models"), list):  # top-level index of a weights repo
        raise ValueError(f"{path} lists the models of a repository, it is not a model config; load one of its "
                         f"model folders instead: {', '.join(map(str, config['models']))}")
    if config.get("model_type") not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {config.get('model_type')!r}; expected one of {sorted(MODEL_TYPES)}")
    return config


def build_model(config: Dict) -> torch.nn.Module:
    """Instantiate a randomly initialised model from a config dictionary."""
    return MODEL_TYPES[config["model_type"]](config)


def weight_files(model_dir: Union[str, Path]):
    """The safetensors files of a model directory, in loading order."""
    model_dir = Path(model_dir)
    index = model_dir / INDEX_NAME
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / name for name in names]
    return [model_dir / WEIGHTS_NAME]


def is_expert_weight(name: str) -> bool:
    """Routed-expert tensors (``*.core.w_gate_up`` / ``*.core.w_down``): the bulk of the parameters."""
    return name.endswith((".core.w_gate_up", ".core.w_down"))


def load_state_dict(model_dir: Union[str, Path], device: Union[str, torch.device] = "cpu",
                    expert_dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """Read single-file or sharded weights, preserving each routed expert's stored dtype by default.

    Mixed storage (bf16 routed experts, fp32 remaining tensors) needs no dtype override. Routers, shared experts,
    routing biases and other floating-point tensors stay fp32, including when ``expert_dtype`` is specified.
    """
    from safetensors.torch import load_file

    state: Dict[str, torch.Tensor] = {}
    for file in weight_files(model_dir):
        if not file.is_file():
            raise FileNotFoundError(f"missing weight file: {file}")
        part = load_file(str(file), device="cpu")
        overlap = state.keys() & part.keys()
        if overlap:
            raise ValueError(f"tensor stored twice: {sorted(overlap)[:3]}")
        for name in list(part):
            tensor = part.pop(name)
            if is_expert_weight(name):
                if expert_dtype is not None:
                    tensor = tensor.to(expert_dtype)
            elif tensor.is_floating_point():
                tensor = tensor.to(torch.float32)
            state[name] = tensor.to(device)
    return state


def load_model(model_dir: Union[str, Path], device: Union[str, torch.device] = "cpu",
               expert_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """Build the model described by ``model_dir/config.json`` and load its weights (strictly).

    ``expert_dtype`` (e.g. ``torch.bfloat16``) stores the routed-expert weight tensors in a smaller dtype. Under
    bf16 autocast this does not change the result, because the expert weights are cast to the compute dtype
    before use; all other weights always stay fp32. With ``None`` the experts keep the dtype of the file.
    Casting bf16 files back to fp32 does not recover the original fp32 precision.

    The model is created without allocating or initialising its weights and the tensors are moved to ``device``
    file by file, so the peak host memory is one weight file, not the whole model.
    """
    config = load_config(model_dir)
    with torch.device("meta"):
        model = build_model(config)
    state = load_state_dict(model_dir, device=device, expert_dtype=expert_dtype)
    model.load_state_dict(state, strict=True, assign=True)  # names and shapes must match exactly
    del state
    for module in model.modules():  # derived constants are not stored in the files
        for name, buffer in list(module.named_buffers(recurse=False)):
            if name in module._non_persistent_buffers_set:
                module.register_buffer(name, buffer.to(device), persistent=False)
    return model.eval().requires_grad_(False)


def cast_expert_weights(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast the routed-expert tensors (``*.core.w_gate_up`` / ``*.core.w_down``) in place."""
    for name, parameter in model.named_parameters():
        if is_expert_weight(name):
            parameter.data = parameter.data.to(dtype)


def save_model(model_or_state, config: Dict, output_dir: Union[str, Path], max_shard_bytes: Optional[int] = None,
               metadata: Optional[Dict[str, str]] = None) -> None:
    """Write ``config.json`` and safetensors weights (one file, or shards of at most ``max_shard_bytes``)."""
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model_or_state.state_dict() if isinstance(model_or_state, torch.nn.Module) else model_or_state
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    metadata = {"format": "pt", **(metadata or {})}
    (output_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")

    total = sum(t.numel() * t.element_size() for t in state.values())
    if max_shard_bytes is None or total <= max_shard_bytes:
        save_file(state, str(output_dir / WEIGHTS_NAME), metadata=metadata)
        return
    shards, current, size = [], {}, 0
    for name, tensor in state.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current and size + nbytes > max_shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = tensor
        size += nbytes
    shards.append(current)
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(output_dir / filename), metadata=metadata)
        weight_map.update({name: filename for name in shard})
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (output_dir / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
