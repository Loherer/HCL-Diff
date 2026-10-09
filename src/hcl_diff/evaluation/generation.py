from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else float("nan")


def lesion_ssim(generated: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    generated = np.asarray(generated, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    values: list[float] = []
    for index in range(generated.shape[0]):
        if not mask[index].any():
            continue
        coordinates = np.argwhere(mask[index])
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0) + 1
        first = generated[index, y0:y1, x0:x1]
        second = reference[index, y0:y1, x0:x1]
        data_range = float(max(first.max(), second.max()) - min(first.min(), second.min()))
        if data_range <= 0 or min(first.shape) < 3:
            values.append(float(np.allclose(first, second)))
            continue
        window = min(7, min(first.shape))
        if window % 2 == 0:
            window -= 1
        values.append(float(structural_similarity(first, second, data_range=data_range, win_size=window)))
    return float(np.mean(values)) if values else float("nan")


def slice_difference_error(
    generated: np.ndarray,
    reference: np.ndarray,
    edit_mask: np.ndarray,
) -> float:
    generated_delta = np.diff(np.asarray(generated, dtype=np.float32), axis=0)
    reference_delta = np.diff(np.asarray(reference, dtype=np.float32), axis=0)
    pair_mask = np.maximum(edit_mask[1:], edit_mask[:-1]).astype(bool)
    return _masked_mean(np.abs(generated_delta - reference_delta), pair_mask)


def wall_gradient_error(
    generated: np.ndarray,
    reference: np.ndarray,
    wall_band: np.ndarray,
) -> float:
    generated = np.asarray(generated, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    wall_band = np.asarray(wall_band, dtype=bool)
    errors = []
    for index in range(generated.shape[0]):
        generated_gradient = np.hypot(
            ndimage.sobel(generated[index], axis=0), ndimage.sobel(generated[index], axis=1)
        )
        reference_gradient = np.hypot(
            ndimage.sobel(reference[index], axis=0), ndimage.sobel(reference[index], axis=1)
        )
        if wall_band[index].any():
            errors.append(np.abs(generated_gradient - reference_gradient)[wall_band[index]])
    return float(np.concatenate(errors).mean()) if errors else float("nan")


def seam_error(generated: np.ndarray, reference: np.ndarray, edit_mask: np.ndarray) -> float:
    edit_mask = np.asarray(edit_mask, dtype=bool)
    ring = np.logical_xor(
        ndimage.binary_dilation(edit_mask, iterations=1),
        ndimage.binary_erosion(edit_mask, iterations=1),
    )
    return _masked_mean(np.abs(np.asarray(generated) - np.asarray(reference)), ring)


def generation_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    lesion_mask: np.ndarray,
    edit_mask: np.ndarray,
    wall_band: np.ndarray | None = None,
) -> dict[str, float]:
    generated = np.asarray(generated, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    lesion_mask = np.asarray(lesion_mask, dtype=bool)
    edit_mask = np.asarray(edit_mask, dtype=bool)
    if generated.shape != reference.shape or generated.shape != lesion_mask.shape:
        raise ValueError("Generated image, reference image and lesion mask must share a shape")
    background = ~edit_mask
    absolute_error = np.abs(generated - reference)
    values = {
        "lesion_mae": _masked_mean(absolute_error, lesion_mask),
        "lesion_ssim": lesion_ssim(generated, reference, lesion_mask),
        "slice_difference_error": slice_difference_error(generated, reference, edit_mask),
        "background_mae": _masked_mean(absolute_error, background),
        "background_max_abs_error": (
            float(absolute_error[background].max()) if background.any() else 0.0
        ),
        "seam_error": seam_error(generated, reference, edit_mask),
    }
    values["wall_gradient_error"] = (
        wall_gradient_error(generated, reference, wall_band)
        if wall_band is not None
        else float("nan")
    )
    return values
