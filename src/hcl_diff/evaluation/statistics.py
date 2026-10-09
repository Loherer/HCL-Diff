from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    value_column: str,
    method_column: str,
    first_method: str,
    second_method: str,
    patient_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    paired = frame.pivot_table(
        index=patient_column,
        columns=method_column,
        values=value_column,
        aggfunc="mean",
    ).dropna(subset=[first_method, second_method])
    differences = (paired[first_method] - paired[second_method]).to_numpy(float)
    if differences.size == 0:
        raise ValueError("No complete patient pairs are available")
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(differences, size=(int(iterations), differences.size), replace=True)
    bootstrap = samples.mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.percentile(bootstrap, 2.5)),
        "ci_upper": float(np.percentile(bootstrap, 97.5)),
        "patient_count": int(differences.size),
    }


def paired_wilcoxon(first: Sequence[float], second: Sequence[float]) -> float:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    difference = first_array[valid] - second_array[valid]
    if difference.size == 0:
        raise ValueError("No finite paired observations are available")
    if np.allclose(difference, 0):
        return 1.0
    return float(wilcoxon(difference, alternative="two-sided").pvalue)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()
