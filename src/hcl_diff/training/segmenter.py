from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import require
from ..data.arrays import BucketBatchSampler, SegmentationDataset, collate_arrays
from ..models.segmentation import build_segmentation_model, parameter_count
from .common import (
    append_csv,
    autocast_context,
    build_scheduler,
    move_tensors,
    resolve_device,
    seed_everything,
    write_json,
)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return soft_dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits, target)


def train_segmenter(config: dict[str, Any]) -> Path:
    require(
        config,
        [
            "project.seed",
            "project.output_dir",
            "data.train_manifest",
            "data.workers",
            "model.backbone",
            "model.input_channels",
            "model.output_channels",
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
        ],
    )
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device()
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    dataset = SegmentationDataset(config["data"]["train_manifest"])
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
    model = build_segmentation_model(config).to(device)
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
    while step < total_steps:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            batch = move_tensors(batch, device)
            with autocast_context(device, precision):
                logits = model(batch["ct"])
                loss = segmentation_loss(logits, batch["mask"])
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
                output_dir / "training_log.csv",
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "gpu_peak_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                },
            )
            if step % int(training["checkpoint_interval"]) == 0:
                torch.save(
                    {"model": model.state_dict(), "step": step, "config": config},
                    output_dir / f"checkpoint_{step}.pt",
                )
            if step >= total_steps:
                break
        epoch += 1
    last_path = output_dir / "last.pt"
    torch.save({"model": model.state_dict(), "step": step, "config": config}, last_path)
    write_json(
        output_dir / "training_summary.json",
        {
            "backbone": config["model"]["backbone"],
            "steps": step,
            "seed": seed,
            "device": str(device),
            "parameter_count": parameter_count(model),
            "checkpoint": str(last_path.resolve()),
        },
    )
    return last_path
