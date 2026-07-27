# Release contents

## Active proposal

- `models/tuning_modules/mdl_tangent_core.py` — MDL-Evidence Adaptive
  Tangent-Core V6.
- `METHOD_MDL_EVIDENCE_TANGENT_CORE_V6.md` — full mathematical definition.
- `tests/test_mdl_tangent_core.py` — rank/core/head-free API, BatchNorm safety,
  determinism, parameter complexity, and exact-merge tests.
- `tests/test_trso_integration.py` — integration with `main.py` and report output.

## Runners

- `main.py` — training/evaluation entry point.
- `tools/run_fair_suite.py` — compatibility-aware controlled comparison runner.
- `tools/verify_fairness.py` — shared-protocol verifier.
- `tools/aggregate_revision_results.py` — raw, mean/std, and paper metric tables,
  including `mdl_*` diagnostics.
- `KAGGLE_MDL_TANGENT_V6_ONE_CELL.py` — self-contained Kaggle runner.
- `kaggle/TRSO_Universal_Fair_OneCell.py` — universal Kaggle copy.

## Removed proposal artifacts

The release does not contain:

- `models/tuning_modules/information_spectral.py`;
- `models/tuning_modules/tangent_core.py`;
- legacy unified/task-response adapter modules;
- fixed-rank/budget proposal runner arguments.

The only TRSO-specific CLI option is `--trso_fast_inference`, which controls
exact post-training merging and does not change training capacity.
