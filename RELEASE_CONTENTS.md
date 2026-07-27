# Release contents

## Active proposal

- `models/tuning_modules/mdl_tangent_core.py` — Cross-Fitted Evidence Adaptive Tangent-Core V6. The legacy filename is retained for result/checkpoint compatibility.
- `METHOD_MDL_EVIDENCE_TANGENT_CORE_V6.md` — mathematical definition.
- `tests/test_mdl_tangent_core.py` — automatic-capacity, head, BatchNorm, determinism, and exact-merge tests.
- `tests/test_trso_integration.py` — integration with `main.py` and report output.

## Runners

- `main.py` — training, validation-safe checkpoint selection, final testing, and exact merge.
- `tools/run_fair_suite.py` — compatibility-aware controlled comparison runner.
- `tools/verify_fairness.py` — shared-protocol verifier.
- `tools/aggregate_revision_results.py` — raw, mean/std, and paper metric tables with proposal diagnostics.
- `KAGGLE_CROSS_FITTED_V6_ONE_CELL.py` — canonical one-cell Kaggle runner.
- `KAGGLE_MDL_TANGENT_V6_ONE_CELL.py` — compatibility copy of the canonical runner.
- `kaggle/TRSO_Universal_Fair_OneCell.py` — universal Kaggle copy.

## Removed proposal artifacts

The release does not contain the old Information-Spectral module, historical fixed-rank production tangent-core module, or fixed proposal rank/budget/threshold controls. The only TRSO-specific CLI option is `--trso_fast_inference`, which controls exact merging and does not change training capacity.
