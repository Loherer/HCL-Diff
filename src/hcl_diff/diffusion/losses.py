from __future__ import annotations

from collections.abc import Mapping

import torch

from ..variants import VariantSpec
from .process import GaussianDiffusion


def select_slices(tensor: torch.Tensor, variant: VariantSpec, center_index: int = 1) -> torch.Tensor:
    if variant.slices == 1:
        return tensor[:, center_index : center_index + 1]
    if tensor.shape[1] != variant.slices:
        raise ValueError(
            f"Expected {variant.slices} slices for {variant.name}, got {tensor.shape[1]}"
        )
    return tensor


def adjacent_slice_difference_loss(
    predicted_clean: torch.Tensor,
    clean: torch.Tensor,
    edit_mask: torch.Tensor,
) -> torch.Tensor:
    predicted_delta = predicted_clean[:, 1:] - predicted_clean[:, :-1]
    target_delta = clean[:, 1:] - clean[:, :-1]
    pair_mask = torch.maximum(edit_mask[:, 1:], edit_mask[:, :-1])
    denominator = pair_mask.sum().clamp_min(1.0)
    return ((predicted_delta - target_delta).abs() * pair_mask).sum() / denominator


def compute_generator_loss(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    variant: VariantSpec,
    background_noise_weight: float,
    slice_difference_weight: float,
    center_index: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean = select_slices(batch["ct"], variant, center_index)
    masked = select_slices(batch["masked_ct"], variant, center_index)
    target = select_slices(batch["target_mask"], variant, center_index)
    edit = select_slices(batch["edit_mask"], variant, center_index)
    timesteps = torch.randint(
        0, diffusion.timesteps, (clean.shape[0],), device=clean.device, dtype=torch.long
    )
    noise = torch.randn_like(clean)
    noisy = diffusion.q_sample(clean, timesteps, noise)
    predicted_noise = model(noisy, masked, target, edit, timesteps)
    weights = edit + (1.0 - edit) * float(background_noise_weight)
    noise_loss = ((predicted_noise - noise).square() * weights).sum() / weights.sum().clamp_min(1.0)

    slice_loss = clean.new_zeros(())
    if variant.slice_difference_loss:
        predicted_clean = diffusion.predict_clean(noisy, timesteps, predicted_noise).clamp(-1.0, 1.0)
        slice_loss = adjacent_slice_difference_loss(predicted_clean, clean, edit)
    total = noise_loss + float(slice_difference_weight) * slice_loss
    return total, {
        "total_loss": total.detach(),
        "noise_loss": noise_loss.detach(),
        "slice_difference_loss": slice_loss.detach(),
    }
