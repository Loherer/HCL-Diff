from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hcl_diff.config import load_config, require
from hcl_diff.data.arrays import TripletDataset, collate_arrays
from hcl_diff.diffusion.process import GaussianDiffusion, cosine_beta_schedule
from hcl_diff.diffusion.sampling import generate_triplet
from hcl_diff.models.diffusion_unet import build_diffusion_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate traceable fixed-condition candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates-per-condition", required=True, type=int)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    require(
        config,
        [
            "project.seed",
            "model.variant",
            "diffusion.timesteps",
            "diffusion.cosine_offset",
            "diffusion.sampling_steps",
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_diffusion_model(config).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    diffusion = GaussianDiffusion(
        cosine_beta_schedule(
            int(config["diffusion"]["timesteps"]),
            float(config["diffusion"]["cosine_offset"]),
        )
    ).to(device)
    dataset = TripletDataset(arguments.manifest, seed=int(config["project"]["seed"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_arrays)
    output_dir = Path(arguments.output_dir)
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for condition_index, batch in enumerate(loader):
        metadata = batch["metadata"][0]
        tensors = {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        for candidate_index in range(int(arguments.candidates_per_condition)):
            seed = int(config["project"]["seed"]) + condition_index * int(
                arguments.candidates_per_condition
            ) + candidate_index
            generated = generate_triplet(
                model=model,
                diffusion=diffusion,
                batch=tensors,
                variant_name=str(config["model"]["variant"]),
                sampling_steps=int(config["diffusion"]["sampling_steps"]),
                seed=seed,
            )[0].float().cpu().numpy()
            candidate_id = f"{metadata['sample_id']}__{candidate_index:02d}"
            path = array_dir / f"{candidate_id}.npz"
            payload = {
                "generated": generated.astype(np.float32),
                "reference": batch["ct"][0].numpy().astype(np.float32),
                "target_mask": batch["target_mask"][0].numpy().astype(np.uint8),
                "edit_mask": batch["edit_mask"][0].numpy().astype(np.uint8),
            }
            if "wall_band" in batch:
                payload["wall_band"] = batch["wall_band"][0].numpy().astype(np.uint8)
            if "signed_distance" in batch:
                payload["signed_distance"] = batch["signed_distance"][0].numpy().astype(np.float32)
            np.savez_compressed(path, **payload)
            rows.append(
                {
                    **metadata,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "diffusion_seed": seed,
                    "variant": config["model"]["variant"],
                    "checkpoint": str(Path(arguments.checkpoint).resolve()),
                    "sampling_steps": int(config["diffusion"]["sampling_steps"]),
                    "output_path": str(path.resolve()),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "candidate_index.csv", index=False)


if __name__ == "__main__":
    main()
