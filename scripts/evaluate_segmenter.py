from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hcl_diff.config import load_config, require
from hcl_diff.data.arrays import SegmentationDataset
from hcl_diff.evaluation.segmentation import segmentation_metrics
from hcl_diff.models.segmentation import build_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed lesion-level 3D predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patient-output")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    require(
        config,
        [
            "evaluation.probability_threshold",
            "evaluation.nsd_tolerance_mm",
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(config).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = SegmentationDataset(arguments.manifest)
    grouped: dict[tuple[str, str], list[tuple[dict[str, object], np.ndarray, np.ndarray]]] = defaultdict(list)
    patient_slices: dict[str, list[tuple[dict[str, object], np.ndarray, np.ndarray]]] = defaultdict(list)
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            logits = model(item["ct"][None].to(device))
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
            record = (item["metadata"], probability, item["mask"][0].numpy())
            grouped[(str(item["metadata"]["case_id"]), str(item["metadata"]["lesion_id"]))].append(record)
            patient_slices[str(item["metadata"]["case_id"])].append(record)
    rows: list[dict[str, object]] = []
    for (case_id, lesion_id), slices in sorted(grouped.items()):
        slices.sort(key=lambda value: int(value[0]["z_index"]))
        probability = np.stack([value[1] for value in slices])
        target = np.stack([value[2] for value in slices])
        metadata = slices[0][0]
        spacing = (
            float(metadata["spacing_z"]),
            float(metadata["spacing_y"]),
            float(metadata["spacing_x"]),
        )
        metrics = segmentation_metrics(
            probability=probability,
            target=target,
            spacing_zyx=spacing,
            probability_threshold=float(config["evaluation"]["probability_threshold"]),
            nsd_tolerance_mm=float(config["evaluation"]["nsd_tolerance_mm"]),
        )
        rows.append(
            {
                "case_id": case_id,
                "lesion_id": lesion_id,
                "slice_count": len(slices),
                **metrics,
            }
        )
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    if arguments.patient_output:
        patient_rows: list[dict[str, object]] = []
        geometry = {"full_depth", "full_height", "full_width", "roi_y0", "roi_x0"}
        for case_id, slices in sorted(patient_slices.items()):
            metadata = slices[0][0]
            missing = geometry - set(metadata)
            if missing:
                raise ValueError(
                    "Patient-merged evaluation requires manifest columns: "
                    + ", ".join(sorted(geometry))
                )
            shape = (
                int(metadata["full_depth"]),
                int(metadata["full_height"]),
                int(metadata["full_width"]),
            )
            full_probability = np.zeros(shape, dtype=np.float32)
            full_target = np.zeros(shape, dtype=bool)
            for row, probability_slice, target_slice in slices:
                z = int(row["z_index"])
                y0 = int(row["roi_y0"])
                x0 = int(row["roi_x0"])
                y1 = y0 + probability_slice.shape[0]
                x1 = x0 + probability_slice.shape[1]
                destination_y0, destination_y1 = max(y0, 0), min(y1, shape[1])
                destination_x0, destination_x1 = max(x0, 0), min(x1, shape[2])
                source_y0, source_y1 = destination_y0 - y0, destination_y1 - y0
                source_x0, source_x1 = destination_x0 - x0, destination_x1 - x0
                if not (0 <= z < shape[0]) or destination_y0 >= destination_y1 or destination_x0 >= destination_x1:
                    raise ValueError(f"ROI geometry is outside the patient volume for {case_id}")
                area = np.s_[z, destination_y0:destination_y1, destination_x0:destination_x1]
                source = np.s_[source_y0:source_y1, source_x0:source_x1]
                full_probability[area] = np.maximum(full_probability[area], probability_slice[source])
                full_target[area] |= target_slice[source].astype(bool)
            spacing = (
                float(metadata["spacing_z"]),
                float(metadata["spacing_y"]),
                float(metadata["spacing_x"]),
            )
            metrics = segmentation_metrics(
                full_probability,
                full_target,
                spacing,
                float(config["evaluation"]["probability_threshold"]),
                float(config["evaluation"]["nsd_tolerance_mm"]),
            )
            patient_rows.append(
                {
                    "case_id": case_id,
                    "lesion_count": len({str(value[0]["lesion_id"]) for value in slices}),
                    "merged_dice": metrics.pop("dice_3d"),
                    **metrics,
                }
            )
        patient_destination = Path(arguments.patient_output)
        patient_destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(patient_rows).to_csv(patient_destination, index=False)


if __name__ == "__main__":
    main()
