# HCL-Diff

HCL-Diff is a local 2.5D diffusion augmentation pipeline for hepatic lesion CT segmentation. It edits a prescribed lesion region while preserving the acquired CT background. Three generator variants isolate the contribution of joint slice generation and cross-slice consistency:

| Variant | Input and output | Cross-slice module | Training objective |
|---|---|---|---|
| C0 | One axial slice per sampling run | None | Mask-conditioned diffusion noise prediction |
| C1 | Joint generation of `z-1`, `z`, and `z+1` | None | Mask-conditioned diffusion noise prediction |
| C2 | Joint generation of `z-1`, `z`, and `z+1` | Bottleneck cross-slice attention with relative slice encoding | Mask-conditioned diffusion noise prediction plus adjacent-slice difference consistency |

The repository also contains the downstream ROI segmentation pipeline used with PlainConvUNet, ResUNet, and SwinUNETR, as well as fixed-condition generation metrics, candidate quality control, volumetric segmentation metrics, patient-clustered bootstrap analysis, paired Wilcoxon tests, and Holm correction.

No clinical data, trained weights, experiment outputs, or study-specific hyperparameter values are distributed with the source tree. Configuration templates list the required fields and must be completed from an approved experiment protocol.

## Repository layout

```text
HCL-Diff/
├── configs/                  Configuration templates
├── scripts/                  Training, sampling, QC, evaluation, and statistics entry points
├── src/hcl_diff/
│   ├── data/                 Manifest readers, ROI datasets, and condition maps
│   ├── diffusion/            Noise process, objectives, and background-locked DDIM sampling
│   ├── evaluation/           Generation, segmentation, and paired statistical metrics
│   ├── models/               Generator and segmentation backbones
│   └── training/             Generator and segmentation training loops
└── tests/                    Unit and gradient-flow tests
```

## Installation

Create a Python environment with a CUDA-enabled PyTorch build when GPU training is required, then install the package:

```bash
python -m pip install -e .[medical,test]
```

Run the test suite before starting an experiment:

```bash
python -m pytest
```

## Data contract

The code reads explicit CSV manifests. Patient identifiers must be assigned to one split only. All slices, lesions, phases, and derived samples from the same patient must remain in that split.

### Generator manifest

Required columns:

```text
case_id,sample_id,array_path,roi_size
```

Each `array_path` points to a compressed NumPy archive with:

| Array | Shape | Description |
|---|---|---|
| `ct` | `[3,H,W]` | Normalized real CT triplet in `z-1,z,z+1` order |
| `target_mask` | `[3,H,W]` | Target lesion mask |
| `edit_mask` | `[3,H,W]` | Editable region containing the source and target masks |
| `masked_ct` | `[3,H,W]` | Optional precomputed masked input |
| `wall_band` | `[3,H,W]` | Optional lesion-wall band used by QC |
| `signed_distance` | `[3,H,W]` | Optional normalized signed-distance map used by QC and visualization |

The target mask, wall band, signed-distance map, and edit mask are created from a single 3D mask before extracting the triplet. The three slices must not be transformed independently. The denoiser receives the noisy CT, masked CT, target mask, and edit mask. Wall-band and signed-distance arrays are retained for quality control and visualization rather than added as hidden supervision.

### Segmentation manifest

Required columns:

```text
case_id,lesion_id,sample_id,array_path,roi_size,z_index,spacing_z,spacing_y,spacing_x
```

Each archive contains normalized `ct` with shape `[3,H,W]` and `center_mask` with shape `[H,W]`. Rows belonging to one lesion are reassembled in increasing `z_index` order for volumetric evaluation.

Patient-merged evaluation additionally requires `full_depth`, `full_height`, `full_width`, `roi_y0`, and `roi_x0`. These fields place every lesion ROI back into the common patient volume without using the label to modify a prediction.

## Condition construction

`hcl_diff.data.build_condition_maps` creates four maps from a full 3D target mask:

- target lesion mask;
- lesion-wall band;
- normalized signed-distance map;
- local edit mask covering both the original and target lesion masks.

Spacing and physical widths are supplied by the experiment configuration. Outside the edit mask, every reverse-diffusion step copies the acquired CT values back into the sample. The final output therefore preserves the background exactly, subject only to numerical storage precision.

## Generator training

Copy the generator template to a local configuration and complete every required value:

```bash
copy configs\generator.template.yaml configs\generator.local.yaml
python scripts\train_generator.py --config configs\generator.local.yaml
```

Set `model.variant` to `C0`, `C1`, or `C2`. C0 uses the same single-slice network for each member of a triplet. C1 generates all three slices in one pass. C2 adds cross-slice attention, relative slice-position encoding, and the adjacent-slice difference loss. Model checkpoints are written only to the configured output directory.

## Candidate generation and quality control

Generate the same number of stochastic candidates for every locked condition:

```bash
python scripts\generate_candidates.py \
  --config configs\generator.local.yaml \
  --checkpoint PATH_TO_CHECKPOINT \
  --manifest PATH_TO_CONDITION_MANIFEST \
  --output-dir PATH_TO_CANDIDATE_DIRECTORY \
  --candidates-per-condition N
```

Candidate records include the patient, sample, random seed, generator variant, checkpoint, sampling schedule, and array path. Quality-control thresholds are external to the implementation:

```bash
copy configs\qc.template.yaml configs\qc.local.yaml
python scripts\score_candidates.py \
  --candidate-index PATH_TO_CANDIDATE_INDEX \
  --qc-config configs\qc.local.yaml \
  --output PATH_TO_QC_TABLE \
  --probability-threshold VALUE \
  --boundary-width VALUE \
  --spacing-zyx Z Y X
```

The QC module checks finite values, non-empty edit regions, lesion MAE and SSIM, adjacent-slice difference error, wall-gradient error, seam error, and exact background preservation. If a separately trained segmentation teacher has produced `mask_probability` in a candidate archive, the same command also reports mask Dice, boundary Dice, HD95, and the fraction of prediction outside the target mask. Teacher scores are offline selection criteria; the teacher is not part of C2 training or inference.

Thresholds must be calibrated on the validation candidates and locked before scoring training candidates. Candidate selection must not use test-set segmentation performance or visual cherry-picking.

## Downstream segmentation

Complete the segmentation template for one backbone and training group:

```bash
copy configs\segmentation.template.yaml configs\segmentation.local.yaml
python scripts\train_segmenter.py --config configs\segmentation.local.yaml
```

The study groups map to data sources as follows:

| Group | Training data |
|---|---|
| G0 | Real positive lesion ROIs only |
| G2 | Real ROIs plus copy-paste augmentation |
| G3 | Real ROIs plus C0 independent 2D diffusion samples |
| G4 | Real ROIs plus C1 joint 2.5D diffusion samples |
| G5 | Real ROIs plus C2 HCL-Diff samples |

Use an identical real-data manifest, optimizer-step budget, augmentation policy, validation set, probability threshold, and post-processing rule within each backbone. Only the added synthetic-data manifest changes between G0, G2, G3, G4, and G5.

Volumetric lesion metrics are calculated from slice predictions reconstructed by `case_id`, `lesion_id`, and `z_index`:

```bash
python scripts\evaluate_segmenter.py \
  --config configs\segmentation.local.yaml \
  --checkpoint PATH_TO_CHECKPOINT \
  --manifest PATH_TO_EVALUATION_MANIFEST \
  --output PATH_TO_LESION_METRICS \
  --patient-output PATH_TO_PATIENT_METRICS
```

The evaluation output contains 3D Dice, normalized surface Dice, HD95, empty prediction and complete miss flags, predicted and target volume, and absolute and relative volume errors. Surfaces are extracted with 26-neighbour connectivity. If both masks are empty, NSD is 1 and HD95 is 0. If only one mask is empty, NSD is 0 and HD95 is set to the physical diagonal of the evaluated volume. Complete misses are therefore retained in every summary rather than dropped as missing observations.

## Paired statistical analysis

Method comparisons use the patient as the resampling and pairing unit:

```bash
python scripts\compare_methods.py \
  --metrics PATH_TO_PATIENT_METRICS \
  --method-column group \
  --patient-column case_id \
  --value-column merged_dice \
  --comparisons G4:G3 G5:G4 G5:G2 G5:G0 \
  --bootstrap-iterations N \
  --seed SEED \
  --output PATH_TO_COMPARISON_TABLE
```

The output reports the paired mean difference, patient-clustered bootstrap confidence interval, paired Wilcoxon p-value, Holm-adjusted p-value, and the number of complete patient pairs.

## Reproducibility rules

- Split patients before generating any triplets, target masks, or synthetic images.
- Keep validation and test patients out of generator training, QC calibration, and downstream model selection.
- Store the configuration, random seed, manifest hash, code revision, checkpoint hash, and command line for every run.
- Use identical target masks, real backgrounds, candidate budgets, and sampling schedules for fixed-pair generator comparisons.
- Select representative images with a prespecified rule rather than test performance.
- Keep clinical images, annotations, checkpoints, and generated arrays outside version control.

## Scope

This repository implements the methods required for the C0-to-C1-to-C2 comparison and the associated multi-backbone lesion segmentation study. Activity classification, global 3D shape generation, utility prediction, and unrelated exploratory branches are outside this codebase.
