from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from .evaluation.generation import generation_metrics
from .evaluation.segmentation import dice_score, hd95


def boundary_dice(prediction: np.ndarray, target: np.ndarray, width: int) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    predicted_boundary = np.logical_xor(
        ndimage.binary_dilation(prediction, iterations=width),
        ndimage.binary_erosion(prediction, iterations=width),
    )
    target_boundary = np.logical_xor(
        ndimage.binary_dilation(target, iterations=width),
        ndimage.binary_erosion(target, iterations=width),
    )
    return dice_score(predicted_boundary, target_boundary)


def candidate_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    target_mask: np.ndarray,
    edit_mask: np.ndarray,
    wall_band: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    mask_probability: np.ndarray | None,
    probability_threshold: float,
    boundary_width: int,
) -> dict[str, float | bool]:
    values: dict[str, float | bool] = generation_metrics(
        generated=generated,
        reference=reference,
        lesion_mask=target_mask,
        edit_mask=edit_mask,
        wall_band=wall_band,
    )
    values["finite"] = bool(np.isfinite(generated).all())
    values["non_empty_edit"] = bool(np.asarray(edit_mask, dtype=bool).any())
    if mask_probability is not None:
        prediction = np.asarray(mask_probability) >= float(probability_threshold)
        target = np.asarray(target_mask, dtype=bool)
        values["teacher_mask_dice"] = dice_score(prediction, target)
        values["boundary_dice"] = boundary_dice(prediction, target, int(boundary_width))
        values["hd95"] = hd95(prediction, target, spacing_zyx)
        outside = np.logical_and(prediction, ~target)
        values["outside_mask_fraction"] = float(outside.sum() / max(prediction.sum(), 1))
    return values


def apply_thresholds(metrics: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not bool(metrics.get("finite", False)):
        failures.append("non_finite_image")
    if not bool(metrics.get("non_empty_edit", False)):
        failures.append("empty_edit_region")
    minimums = {
        "teacher_mask_dice": "mask_dice_min",
        "boundary_dice": "boundary_dice_min",
        "lesion_ssim": "lesion_ssim_min",
    }
    maximums = {
        "hd95": "hd95_max",
        "outside_mask_fraction": "outside_mask_fraction_max",
        "lesion_mae": "lesion_mae_max",
        "background_mae": "background_mae_max",
        "background_max_abs_error": "background_max_abs_error_max",
        "slice_difference_error": "slice_difference_error_max",
        "wall_gradient_error": "wall_gradient_error_max",
        "seam_error": "seam_error_max",
    }
    for metric, threshold_name in minimums.items():
        threshold = thresholds.get(threshold_name)
        if threshold is not None:
            if metric not in metrics or not np.isfinite(float(metrics[metric])):
                failures.append(f"missing_{metric}")
            elif float(metrics[metric]) < float(threshold):
                failures.append(metric)
    for metric, threshold_name in maximums.items():
        threshold = thresholds.get(threshold_name)
        if threshold is not None:
            if metric not in metrics or not np.isfinite(float(metrics[metric])):
                failures.append(f"missing_{metric}")
            elif float(metrics[metric]) > float(threshold):
                failures.append(metric)
    return not failures, failures
