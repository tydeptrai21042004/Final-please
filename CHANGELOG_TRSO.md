# TRSO changelog

## MDL-Evidence Adaptive Tangent-Core V6

- Removed the former Information-Spectral proposal.
- Removed all fixed proposal rank, budget, threshold, layer, calibration-length,
  head-policy, and core-mode controls.
- Added complete-loader gradient sufficient statistics.
- Made calibration BatchNorm-safe by using evaluation mode.
- Added rank-zero layer skipping through MDL.
- Added automatic diagonal/dense core selection by cross-fold BIC.
- Added automatic frozen/tangent/full task-head selection.
- Replaced flattened full-size basis directions with compact two-sided bases.
- Added exact weight merging and restored-trainability support for best-checkpoint
  evaluation.
- Updated fair/Kaggle/shell runners and aggregate result tables.
- Added proposal, integration, determinism, BatchNorm, and release-contract tests.
