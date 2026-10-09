from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import require
from ..data.arrays import BucketBatchSampler, TripletDataset, collate_arrays
from ..diffusion.losses import compute_generator_loss
from ..diffusion.process import GaussianDiffusion, cosine_beta_schedule
from ..models.diffusion_unet import build_diffusion_model
from ..variants import get_variant
from .common import (
    append_csv,
    autocast_context,
    build_scheduler,
    move_tensors,
    resolve_device,
    seed_everything,
    write_json,
)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "config": config,
        },
        path,
    )


def train_generator(config: dict[str, Any]) -> Path:
    require(
        config,
        [
            "project.seed",
            "project.output_dir",
            "data.train_manifest",
            "data.workers",
            "model.variant",
            "model.base_channels",
            "model.channel_multipliers",
            "model.attention_heads",
            "model.time_embedding_multiplier",
            "model.gradient_checkpointing",
            "diffusion.timesteps",
            "diffusion.cosine_offset",
            "training.optimizer",
            "training.scheduler",
            "training.scheduler_power",
            "training.learning_rate",
            "training.weight_decay",
            "training.batch_size_by_roi",
            "training.gradient_accumulation",
            "training.total_steps",
            "training.checkpoint_interval",
            "training.mixed_precision",
            "training.background_noise_weight",
            "training.slice_difference_weight",
        ],
    )
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device()
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    dataset = TripletDataset(config["data"]["train_manifest"], seed=seed)
    sampler = BucketBatchSampler(
        dataset,
        batch_sizes={int(k): int(v) for k, v in training["batch_size_by_roi"].items()},
        shuffle=True,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["data"]["workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=collate_arrays,
    )
    model = build_diffusion_model(config).to(device)
    variant = get_variant(config["model"]["variant"])
    diffusion = GaussianDiffusion(
        cosine_beta_schedule(
            int(config["diffusion"]["timesteps"]),
            float(config["diffusion"]["cosine_offset"]),
        )
    ).to(device)
    optimizer_name = str(training["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError("The reference training entry point supports AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = int(training["total_steps"])
    scheduler = build_scheduler(
        optimizer,
        str(training["scheduler"]),
        total_steps,
        float(training["scheduler_power"]),
    )
    accumulation = int(training["gradient_accumulation"])
    precision = str(training["mixed_precision"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and precision.lower() == "fp16"
    )
    optimizer.zero_grad(set_to_none=True)
    step = 0
    epoch = 0
    started = time.perf_counter()
    log_path = output_dir / "training_log.csv"
    while step < total_steps:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            batch = move_tensors(batch, device)
            with autocast_context(device, precision):
                loss, metrics = compute_generator_loss(
                    model=model,
                    diffusion=diffusion,
                    batch=batch,
                    variant=variant,
                    background_noise_weight=float(training["background_noise_weight"]),
                    slice_difference_weight=float(training["slice_difference_weight"]),
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % accumulation:
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            append_csv(
                log_path,
                {
                    "step": step,
                    "epoch": epoch,
                    "total_loss": float(metrics["total_loss"]),
                    "noise_loss": float(metrics["noise_loss"]),
                    "slice_difference_loss": float(metrics["slice_difference_loss"]),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "gpu_peak_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                },
            )
            if step % int(training["checkpoint_interval"]) == 0:
                _save_checkpoint(
                    output_dir / f"checkpoint_{step}.pt", model, optimizer, scheduler, step, config
                )
            if step >= total_steps:
                break
        epoch += 1
    last_path = output_dir / "last.pt"
    _save_checkpoint(last_path, model, optimizer, scheduler, step, config)
    write_json(
        output_dir / "training_summary.json",
        {
            "variant": variant.name,
            "steps": step,
            "seed": seed,
            "device": str(device),
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "checkpoint": str(last_path.resolve()),
        },
    )
    return last_path
