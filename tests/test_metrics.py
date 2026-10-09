from __future__ import annotations

import numpy as np

from hcl_diff.evaluation.generation import generation_metrics
from hcl_diff.evaluation.segmentation import segmentation_metrics


def test_background_locking_metric_is_zero_for_identical_background() -> None:
    reference = np.zeros((3, 16, 16), dtype=np.float32)
    generated = reference.copy()
    mask = np.zeros_like(reference, dtype=np.uint8)
    mask[:, 4:12, 4:12] = 1
    generated[mask.astype(bool)] = 0.5
    metrics = generation_metrics(generated, reference, mask, mask, mask)
    assert metrics["background_mae"] == 0.0
    assert metrics["background_max_abs_error"] == 0.0


def test_segmentation_metrics_for_exact_match() -> None:
    target = np.zeros((3, 16, 16), dtype=np.uint8)
    target[:, 4:12, 4:12] = 1
    probability = target.astype(np.float32)
    metrics = segmentation_metrics(probability, target, (5.0, 1.0, 1.0), 0.5, 1.0)
    assert metrics["dice_3d"] == 1.0
    assert metrics["nsd"] == 1.0
    assert metrics["hd95"] == 0.0
    assert metrics["prediction_empty"] is False
