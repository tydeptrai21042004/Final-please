# MDL-Evidence Adaptive Tangent-Core V6

This repository contains one active TRSO proposal and a capability-aware set of
compatible PEFT and full-tuning baselines.

The proposal is **MDL-Evidence Adaptive Tangent-Core V6**. It automatically
selects layers, ranks, diagonal/dense core structure, and task-head policy from
complete-loader gradient evidence. It exposes no proposal-specific rank,
budget, threshold, layer list, core switch, head switch, or calibration-length
hyperparameter.

## Core implementation

```text
models/tuning_modules/mdl_tangent_core.py
```

Mathematical specification:

```text
METHOD_MDL_EVIDENCE_TANGENT_CORE_V6.md
```

## Quick test

```bash
python -m pytest -q
```

## Offline proposal smoke run

```bash
bash scripts/run_smoke_trso_fake.sh
```

## Fair DTD comparison

```bash
python -m tools.run_fair_suite \
  --dataset dtd \
  --task auto \
  --data_path ./data \
  --download True \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods linear,trso,full,lora,bitfit,ssf,adaptformer,conv \
  --seeds 0,1,2 \
  --epochs 50 \
  --batch_size 64 \
  --peft_lr 1e-3 \
  --full_lr 1e-4 \
  --linear_lr 1e-3 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --augmentation strong \
  --output_root outputs_dtd \
  --manifest outputs_dtd/fair_manifest.json \
  --execute
```

The fair runner trains a shared task-aware linear head once per
backbone/seed. Compatible PEFT methods, including TRSO, start from that same
head checkpoint. Unsupported method/backbone combinations are written to the
compatibility report rather than silently omitted.

## Proposal outputs

Each TRSO run writes:

- `mdl_tangent_calibration.json` — MDL/BIC selection and storage diagnostics;
- `parameter_summary.json` — trainable and total parameter counts;
- `efficiency_profile.json` — optional latency/FPS/memory measurements;
- `history.json` and `convergence_summary.json` — optimization behavior;
- `test_summary.json` — final held-out metrics.

Aggregate all completed runs with:

```bash
python -m tools.aggregate_revision_results \
  --root outputs_dtd \
  --out_csv outputs_dtd/all_results.csv
```

## Kaggle

Use either:

```text
KAGGLE_MDL_TANGENT_V6_ONE_CELL.py
kaggle/TRSO_Universal_Fair_OneCell.py
```

Both runners verify that the uploaded repository contains the MDL tangent-core
module and no removed proposal implementation.

## Deployment

`--trso_fast_inference True` exactly merges every learned
\(U_\ell K_\ell V_\ell^\top\) update into the corresponding original weight
before final evaluation. The deployed network contains no runtime adapter.

## Current verification

- full repository tests: 83 passed;
- targeted MDL proposal/integration/release tests: 12 passed;
- one-epoch ResNet-18 fake-data run: calibration, training, checkpoint reload,
  final test, and exact merge completed successfully.
