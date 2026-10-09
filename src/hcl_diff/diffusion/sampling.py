from __future__ import annotations

from collections.abc import Mapping

import torch

from ..variants import VariantSpec, get_variant
from .losses import select_slices
from .process import GaussianDiffusion


@torch.inference_mode()
def ddim_sample(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    masked_ct: torch.Tensor,
    real_ct: torch.Tensor,
    target_mask: torch.Tensor,
    edit_mask: torch.Tensor,
    steps: int,
    seed: int,
) -> torch.Tensor:
    if steps < 1 or steps > diffusion.timesteps:
        raise ValueError("sampling steps must be between 1 and the training timestep count")
    generator = torch.Generator(device=real_ct.device).manual_seed(int(seed))
    current = torch.randn(real_ct.shape, generator=generator, device=real_ct.device, dtype=real_ct.dtype)
    background_noise = torch.randn(
        real_ct.shape, generator=generator, device=real_ct.device, dtype=real_ct.dtype
    )
    sequence = torch.linspace(
        diffusion.timesteps - 1, 0, steps, device=real_ct.device
    ).round().long().unique_consecutive()
    batch_size = real_ct.shape[0]
    for index, timestep in enumerate(sequence):
        t = torch.full((batch_size,), int(timestep.item()), device=real_ct.device, dtype=torch.long)
        predicted_noise = model(current, masked_ct, target_mask, edit_mask, t)
        alpha = diffusion.alpha_cumprod[timestep]
        clean = ((current - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)).clamp(-1.0, 1.0)
        if index == len(sequence) - 1:
            current = clean
        else:
            next_timestep = sequence[index + 1]
            next_alpha = diffusion.alpha_cumprod[next_timestep]
            current = torch.sqrt(next_alpha) * clean + torch.sqrt(1.0 - next_alpha) * predicted_noise
            background_t = (
                torch.sqrt(next_alpha) * real_ct
                + torch.sqrt(1.0 - next_alpha) * background_noise
            )
            current = current * edit_mask + background_t * (1.0 - edit_mask)
    return current * edit_mask + real_ct * (1.0 - edit_mask)


@torch.inference_mode()
def generate_triplet(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    variant_name: str,
    sampling_steps: int,
    seed: int,
) -> torch.Tensor:
    variant = get_variant(variant_name)
    if variant.slices == 1:
        outputs = []
        for slice_index in range(batch["ct"].shape[1]):
            real = select_slices(batch["ct"], variant, slice_index)
            masked = select_slices(batch["masked_ct"], variant, slice_index)
            target = select_slices(batch["target_mask"], variant, slice_index)
            edit = select_slices(batch["edit_mask"], variant, slice_index)
            outputs.append(
                ddim_sample(
                    model,
                    diffusion,
                    masked_ct=masked,
                    real_ct=real,
                    target_mask=target,
                    edit_mask=edit,
                    steps=int(sampling_steps),
                    seed=int(seed) + slice_index,
                )
            )
        return torch.cat(outputs, dim=1)
    return ddim_sample(
        model,
        diffusion,
        masked_ct=batch["masked_ct"],
        real_ct=batch["ct"],
        target_mask=batch["target_mask"],
        edit_mask=batch["edit_mask"],
        steps=int(sampling_steps),
        seed=int(seed),
    )
