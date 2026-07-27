# Correction report

## Confirmed failure in the previous release

The first DTD run selected `head_policy=frozen` for both ResNet-50 and ViT-Tiny. It obtained
67.45% and 50.53% accuracy, respectively. The classifier BIC penalty made the frozen policy
almost unavoidable, while the layer MDL score incorrectly treated matrix entries as
independent observations.

## Implemented corrections

- Removed frozen/tangent/full head selection; the complete task head is always trained.
- Replaced the `Bmn` information score with symmetric cross-fold predictive gain.
- Added stable-rank automatic capacity with no rank or energy threshold.
- Added chance-referenced dense rescue for an uninformative transferred head.
- Changed CNN calibration to use target-batch BatchNorm statistics while restoring all
  running buffers exactly.
- Registered the zero-update shared-head model as the initial best validation checkpoint.
- Updated result aggregation to report calibration loss, chance reference, rescue state,
  selected ranks, coordinate modes, adapter parameters, and basis storage.

## What is guaranteed

The selected TRSO checkpoint cannot be worse than its zero-update shared-head starting point
on the validation metric used by the runner. Exact test-set improvement cannot be promised
without executing the real DTD rerun; test labels are never used for model selection.
