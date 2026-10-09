from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    return np.logical_and(mask, ~ndimage.binary_erosion(mask, structure=structure, border_value=0))


def surface_distances(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() or not target.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    predicted_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = ndimage.distance_transform_edt(~predicted_surface, sampling=spacing)
    return distance_to_target[predicted_surface], distance_to_prediction[target_surface]


def normalized_surface_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    tolerance_mm: float,
) -> float:
    first, second = surface_distances(prediction, target, spacing)
    if first.size == 0 or second.size == 0:
        return 1.0 if not np.asarray(prediction).any() and not np.asarray(target).any() else 0.0
    return float((np.sum(first <= tolerance_mm) + np.sum(second <= tolerance_mm)) / (first.size + second.size))


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        physical_extent = (np.asarray(prediction.shape, dtype=float) - 1.0) * np.asarray(
            spacing, dtype=float
        )
        return float(np.linalg.norm(physical_extent))
    first, second = surface_distances(prediction, target, spacing)
    return float(np.percentile(np.concatenate([first, second]), 95))


def segmentation_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    probability_threshold: float,
    nsd_tolerance_mm: float,
) -> dict[str, float | int | bool]:
    probability = np.asarray(probability, dtype=np.float32)
    target = np.asarray(target, dtype=bool)
    if probability.shape != target.shape:
        raise ValueError("Probability and target volumes must share a shape")
    prediction = probability >= float(probability_threshold)
    voxel_volume_ml = float(np.prod(spacing_zyx)) / 1000.0
    target_volume = float(target.sum() * voxel_volume_ml)
    predicted_volume = float(prediction.sum() * voxel_volume_ml)
    absolute_volume_error = abs(predicted_volume - target_volume)
    relative_volume_error = (
        absolute_volume_error / target_volume if target_volume > 0 else float("nan")
    )
    overlap = bool(np.logical_and(prediction, target).any())
    return {
        "dice_3d": dice_score(prediction, target),
        "nsd": normalized_surface_dice(
            prediction, target, spacing_zyx, float(nsd_tolerance_mm)
        ),
        "hd95": hd95(prediction, target, spacing_zyx),
        "target_volume_ml": target_volume,
        "predicted_volume_ml": predicted_volume,
        "absolute_volume_error_ml": absolute_volume_error,
        "relative_volume_error": relative_volume_error,
        "prediction_empty": bool(not prediction.any()),
        "complete_miss": bool(target.any() and not overlap),
        "predicted_component_count": int(ndimage.label(prediction)[1]),
    }
