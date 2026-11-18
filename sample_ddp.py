"""Generate the 50,000 images of the FID protocol and pack them into one ``.npz`` for the ADM evaluator.

    # one GPU
    python sample_ddp.py --model checkpoints/dylomo-dit-xl --output runs/xl
    # N GPUs
    torchrun --nproc_per_node=N sample_ddp.py --model checkpoints/dylomo-dit-xl --output runs/xl

Protocol of the paper: EMA weights, 250-step DDPM, classifier-free guidance 1.5 on the first three latent channels,
fp32 weights and activations, sd-vae-ft-ema decoder, 50,000 images with class labels drawn uniformly. The result,
``<output>/samples.npz`` (``arr_0``: uint8 images ``[50000, 256, 256, 3]``, ``arr_1``: labels), is the input of
``evaluator.py`` from openai/guided-diffusion together with ``VIRTUAL_imagenet256_labeled.npz``.

The noise and the label of a sample depend only on its sample ID (``--seed``, ``--label-seed``), so the image set does
not depend on the number of GPUs (up to floating-point rounding).

Resuming. Finished shards are kept and an interrupted run continues with the same command. The settings of a run,
the SHA-256 of the weight files and a digest of the VAE weights are recorded in ``<output>/sampling.json``; every shard
carries the digest of that record and its sample IDs. A run whose settings, weights or VAE differ from the record
is refused, and so is a shard that does not belong to the record: existing images are never mixed with new ones.
Use a new output directory for a different model or setting.
"""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

import dylomo
from dylomo.diffusion import sample_class_conditional, sample_labels
from dylomo.vae import SD_VAE, decode_sd_latents, load_sd_vae

IMAGE_SIZE = 256
RECORD = "sampling.json"


# --------------------------------------------------------------------------- identity of a run
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_sha256(module) -> str:
    """Digest of the weights a module was loaded with (names, dtypes, shapes and values, in name order)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_record(args, config: dict, vae) -> dict:
    """Everything the images depend on. Two runs may share shards only if their records are identical."""
    vae_config = {key: value for key, value in dict(getattr(vae, "config", {})).items() if not key.startswith("_")}
    return dict(
        schema=2,
        model_config=config,
        weights_sha256={file.name: file_sha256(file) for file in dylomo.io.weight_files(args.model)},
        vae=dict(config=json.loads(json.dumps(vae_config, default=str)), weights_sha256=module_sha256(vae)),
        num_samples=args.num_samples, batch_size=args.batch_size, steps=args.steps, cfg_scale=args.cfg_scale,
        guided_channels=[0, 1, 2], seed=args.seed, label_seed=args.label_seed, precision=args.precision,
        tf32=args.tf32, image_size=IMAGE_SIZE,
    )


def check_record(root: Path, record: dict):
    """Compare with the record of the output directory; write it if there is none. Returns an error text or None."""
    path = root / RECORD
    if not path.exists():
        stray = sorted(p.name for p in root.glob("shard-*.npz"))
        if stray:
            return f"{root} contains image shards but no {RECORD}; use a new output directory"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return None
    stored = json.loads(path.read_text())
    if stored == record:
        return None
    changed = sorted(key for key in set(stored) | set(record) if stored.get(key) != record.get(key))
    return (f"{path} was written by a run with different {', '.join(changed)}; its images cannot be reused. "
            f"Use a new output directory.")


# --------------------------------------------------------------------------- shards
def shard_plan(num_samples: int, batch_size: int):
    """``[(shard name, sample IDs)]``: consecutive IDs in groups of ``batch_size``, independent of the GPU count."""
    return [(f"shard-{index:06d}.npz", list(range(start, min(start + batch_size, num_samples))))
            for index, start in enumerate(range(0, num_samples, batch_size))]


def load_shard(path: Path, ids, record_sha256: str, label_seed: int, num_classes: int):
    """Images and labels of a finished shard, after checking that it belongs to this run."""
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if not {"images", "labels", "ids", "record"} <= names:
            raise ValueError(f"{path.name}: not a shard of this script")
        if str(data["record"]) != record_sha256:
            raise ValueError(f"{path.name}: written under a different {RECORD}")
        if data["ids"].tolist() != list(ids):
            raise ValueError(f"{path.name}: holds other sample IDs than expected")
        images, labels = data["images"], data["labels"]
    if images.shape != (len(ids), IMAGE_SIZE, IMAGE_SIZE, 3) or images.dtype != np.uint8:
        raise ValueError(f"{path.name}: unexpected image array {images.shape} {images.dtype}")
    if labels.tolist() != sample_labels(ids, label_seed, num_classes):
        raise ValueError(f"{path.name}: labels do not match the label seed")
    return images, labels


def pack(root: Path, record: dict) -> Path:
    """Merge all shards into ``arr_0`` (images) / ``arr_1`` (labels) of an uncompressed ``samples.npz``."""
    count, digest = record["num_samples"], canonical_sha256(record)
    image_path, label_path = root / "arr_0.npy", root / "arr_1.npy"
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=(count, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64, shape=(count,))
    try:
        for name, ids in shard_plan(count, record["batch_size"]):
            if not (root / name).exists():
                raise ValueError(f"{name} is missing: sampling is not finished")
            images[ids[0]:ids[-1] + 1], labels[ids[0]:ids[-1] + 1] = load_shard(
                root / name, ids, digest, record["label_seed"], record["model_config"].get("num_classes", 1000))
        images.flush()
        labels.flush()
        del images, labels
        output = root / "samples.npz"
        temporary = output.with_name(output.name + ".part")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(image_path, "arr_0.npy")
            archive.write(label_path, "arr_1.npy")
        os.replace(temporary, output)
    finally:
        for path in (image_path, label_path):
            if path.exists():
                path.unlink()
    return output


# --------------------------------------------------------------------------- processes
def setup_distributed(device: str):
    """(device, rank, world size); a process group is created only under torchrun."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return torch.device(device), 0, 1
    import torch.distributed as dist

    dist.init_process_group("gloo")  # only used to agree on the run record and to wait before packing
    rank, world = dist.get_rank(), dist.get_world_size()
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
        return torch.device("cuda", torch.cuda.current_device()), rank, world
    return torch.device("cpu"), rank, world


def agree(value, world: int):
    """Give every process the value of rank 0."""
    if world == 1:
        return value
    import torch.distributed as dist

    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="directory with config.json and model.safetensors")
    parser.add_argument("--vae", default=SD_VAE, help="sd-vae-ft-ema: hub ID or local directory")
    parser.add_argument("--output", required=True, help="output directory (one per model and setting)")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32, help="images per shard and per forward batch")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                        help="fp32 is the evaluation protocol; bf16 samples under bf16 autocast")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="TF32 for matrix products on CUDA (torch.backends.cuda.matmul.allow_tf32). On by "
                             "default, as in the evaluation of the paper. --no-tf32 turns it off for matrix "
                             "products only; the cuDNN setting used by the VAE convolutions is not touched")
    parser.add_argument("--deterministic", action="store_true",
                        help="use deterministic PyTorch algorithms (repeatable results on a given GPU); see README")
    parser.add_argument("--pack-only", action="store_true", help="only merge the finished shards into samples.npz")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.pack_only:
        if not (root / RECORD).exists():
            parser.error(f"{root / RECORD} not found: nothing to pack")
        print(f"wrote {pack(root, json.loads((root / RECORD).read_text()))}")
        return
    if not args.model:
        parser.error("--model is required")
    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("--num-samples and --batch-size must be positive")

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    device, rank, world = setup_distributed(args.device)

    config = dylomo.load_config(args.model)
    model = dylomo.load_model(args.model, device=device)
    vae = load_sd_vae(args.vae, device=device)

    # rank 0 decides whether this run may use the output directory; every rank must have loaded the same files
    local_record = build_record(args, config, vae)
    verdict = dict(record=local_record, error=check_record(root, local_record)) if rank == 0 else None
    verdict = agree(verdict, world)
    if verdict["error"]:
        raise SystemExit(verdict["error"])
    record = verdict["record"]
    if local_record != record:
        raise SystemExit(f"rank {rank} loaded other weights or settings than rank 0")
    digest = canonical_sha256(record)

    plan = shard_plan(args.num_samples, args.batch_size)
    expected = {name for name, _ in plan}
    if rank == 0:
        stray = sorted(p.name for p in root.glob("shard-*.npz") if p.name not in expected)
        if stray:
            raise SystemExit(f"{root} contains shards that do not belong to this run ({stray[0]}, ...); "
                             f"use a new output directory")
    mine = [(name, ids) for index, (name, ids) in enumerate(plan) if index % world == rank]
    for done, (name, ids) in enumerate(mine, start=1):
        shard = root / name
        if shard.exists():
            try:
                load_shard(shard, ids, digest, args.label_seed, model.num_classes)
            except ValueError as error:
                raise SystemExit(f"{error}; refusing to reuse or overwrite it. Use a new output directory.")
            continue
        latents, labels = sample_class_conditional(model, ids, seed=args.seed, label_seed=args.label_seed,
                                                   steps=args.steps, cfg_scale=args.cfg_scale, device=device,
                                                   latent_size=IMAGE_SIZE // 8, precision=args.precision)
        images = decode_sd_latents(vae, latents)
        temporary = shard.with_name(shard.name + ".part")
        with temporary.open("wb") as file:
            np.savez(file, images=images, labels=labels.cpu().numpy().astype(np.int64),
                     ids=np.array(ids, dtype=np.int64), record=np.array(digest))
        os.replace(temporary, shard)
        print(f"rank {rank}: shard {done}/{len(mine)}", flush=True)

    if world > 1:
        import torch.distributed as dist

        dist.barrier()
    if rank == 0:
        print(f"wrote {pack(root, record)}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
