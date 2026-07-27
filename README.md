# MDL-Evidence Adaptive Tangent-Core V6

This repository contains one active TRSO proposal and compatible PEFT baselines.

The corrected proposal addresses the confirmed causes of the first real DTD failure:

1. the task classifier was incorrectly frozen;
2. the former MDL score treated `Bmn` gradient entries as independent observations;
3. the small diagonal selector lacked a reliable rescue for difficult domain shifts.

The active method now provides:

- a fully trainable task classifier;
- cross-fitted predictive evidence for layer and coordinate selection;
- stable-rank automatic capacity;
- a parameter-free square-root-chance dense rescue;
- target-statistics BatchNorm calibration with exact state restoration;
- a zero-update validation safeguard;
- exact adapter merging before final inference;
- no proposal-specific rank, budget, threshold, layer, core, head, or calibration-length argument.

## Main files

- `models/tuning_modules/mdl_tangent_core.py` — active proposal;
- `main.py` — training, checkpoint safeguard, and exact merge;
- `tools/run_fair_suite.py` — controlled benchmark runner;
- `KAGGLE_CROSS_FITTED_V6_ONE_CELL.py` — canonical one-cell Kaggle runner;
- `METHOD_MDL_EVIDENCE_TANGENT_CORE_V6.md` — mathematical specification;
- `tools/validate_cross_fitted_v6.py` — offline comparison with historical fixed V6;
- `validation/CORRECTED_V6_VALIDATION.json` — executed evidence;
- `FINAL_TEST_REPORT.md` — verification and limitations.

## Tests

```bash
pytest -q
```

## Controlled validation

```bash
python -m tools.validate_cross_fitted_v6 --suite standard --seeds 0,1,2
python -m tools.validate_cross_fitted_v6 --suite hard --seeds 0,1,2,3,4,5
```

## Real DTD benchmark

Upload the repository as a Kaggle dataset, enable a GPU, and run
`KAGGLE_CROSS_FITTED_V6_ONE_CELL.py`. The runner loads the same shared linear head and uses the same fair training recipe for all compatible PEFT methods.
