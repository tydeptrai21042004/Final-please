# TRSO changelog

## Cross-Fitted Evidence Adaptive Tangent-Core V6

- Removed the frozen/tangent/full classifier BIC policy; the complete task head is always trainable.
- Replaced the invalid `Bmn` MDL observation count with symmetric held-out cross-fold predictive gain.
- Added stable-rank intrinsic capacity with rank-zero layer skipping and no manual rank or threshold.
- Added an automatic full-core rescue at the square-root-chance boundary.
- Calibrated CNN response directions with target-batch BatchNorm statistics while restoring every running buffer exactly.
- Added the zero-update shared-head model as the initial best-validation checkpoint.
- Preserved compact two-sided bases and exact post-training weight merging.
- Updated fair/Kaggle runners, aggregation, documentation, and regression tests.
