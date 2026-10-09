from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hcl_diff.config import load_config
from hcl_diff.qc import apply_thresholds, candidate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply locked quality-control rules to candidates.")
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--qc-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probability-threshold", required=True, type=float)
    parser.add_argument("--boundary-width", required=True, type=int)
    parser.add_argument("--spacing-zyx", required=True, nargs=3, type=float)
    arguments = parser.parse_args()
    thresholds = load_config(arguments.qc_config)["thresholds"]
    frame = pd.read_csv(arguments.candidate_index)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        with np.load(record["output_path"], allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        wall_band = arrays.get("wall_band", arrays["target_mask"])
        mask_probability = arrays.get("mask_probability")
        metrics = candidate_metrics(
            generated=arrays["generated"],
            reference=arrays["reference"],
            target_mask=arrays["target_mask"],
            edit_mask=arrays["edit_mask"],
            wall_band=wall_band,
            spacing_zyx=tuple(float(value) for value in arguments.spacing_zyx),
            mask_probability=mask_probability,
            probability_threshold=float(arguments.probability_threshold),
            boundary_width=int(arguments.boundary_width),
        )
        passed, failures = apply_thresholds(metrics, thresholds)
        rows.append({**record, **metrics, "qc_pass": passed, "failure_reasons": ";".join(failures)})
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


if __name__ == "__main__":
    main()
