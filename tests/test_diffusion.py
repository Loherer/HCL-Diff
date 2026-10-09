from __future__ import annotations

import torch

from hcl_diff.diffusion.losses import compute_generator_loss
from hcl_diff.diffusion.process import GaussianDiffusion, cosine_beta_schedule
from hcl_diff.diffusion.sampling import generate_triplet
from hcl_diff.models.diffusion_unet import HCLDiffUNet
from hcl_diff.variants import get_variant


def _model(variant_name: str) -> HCLDiffUNet:
    return HCLDiffUNet(
        variant=get_variant(variant_name),
        base_channels=8,
        channel_multipliers=[1, 2],
        attention_heads=1,
        time_embedding_multiplier=2,
        gradient_checkpointing=False,
    )


def test_all_generator_variants_preserve_shape() -> None:
    for name in ("C0", "C1", "C2"):
        variant = get_variant(name)
        model = _model(name)
        noisy = torch.randn(2, variant.slices, 32, 32)
        masked = torch.randn_like(noisy)
        target = torch.zeros_like(noisy)
        edit = torch.ones_like(noisy)
        output = model(noisy, masked, target, edit, torch.tensor([1, 2]))
        assert output.shape == noisy.shape


def test_c2_loss_propagates_gradient() -> None:
    model = _model("C2")
    diffusion = GaussianDiffusion(cosine_beta_schedule(16, 0.01))
    batch = {
        "ct": torch.randn(1, 3, 32, 32),
        "masked_ct": torch.randn(1, 3, 32, 32),
        "target_mask": torch.ones(1, 3, 32, 32),
        "edit_mask": torch.ones(1, 3, 32, 32),
    }
    loss, metrics = compute_generator_loss(
        model=model,
        diffusion=diffusion,
        batch=batch,
        variant=get_variant("C2"),
        background_noise_weight=0.0,
        slice_difference_weight=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(metrics["slice_difference_loss"]) >= 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_sampling_restores_background_exactly() -> None:
    model = _model("C2").eval()
    diffusion = GaussianDiffusion(cosine_beta_schedule(8, 0.01))
    ct = torch.randn(1, 3, 32, 32)
    edit = torch.zeros_like(ct)
    edit[:, :, 8:24, 8:24] = 1
    batch = {
        "ct": ct,
        "masked_ct": ct * (1 - edit),
        "target_mask": edit.clone(),
        "edit_mask": edit,
    }
    generated = generate_triplet(model, diffusion, batch, "C2", sampling_steps=4, seed=7)
    assert torch.equal(generated[edit == 0], ct[edit == 0])
