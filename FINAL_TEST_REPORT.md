# Final test report

## Automated tests

```text
83 passed in 16.87 s
```

The suite covers baseline modules, datasets, compatibility contracts, fair
protocol construction, checkpoint behavior, CLI smoke behavior, and the active
MDL tangent-core proposal.

## Proposal-specific checks

```text
12 passed in 2.40 s
```

Verified properties:

- no manual proposal model-selection controls;
- complete-loader calibration;
- BatchNorm running-buffer preservation;
- zero-update function preservation;
- deterministic model selection;
- linear versus quadratic core parameter counts;
- trainability after calibration;
- exact merge with unchanged output;
- report creation through `main.py`;
- release contract and old-module absence.

## End-to-end smoke run

Configuration: fake dataset, torchvision ResNet-18, CPU, one epoch, 12 training
examples, four classes.

Observed proposal selection:

- eligible tensors: 20;
- selected tensors: 19;
- skipped tensors: 1;
- adapter parameters: 26;
- frozen basis values: 61,568;
- automatic head policy: frozen;
- exact merged updates before final test: 19.

The run completed optimizer construction, training, validation selection,
checkpoint reload, exact merging, and held-out final evaluation without error.

## Controlled fixed-V6 comparison

On the offline TinyCNN source-to-target texture transfer control, both methods
reached 100.0% accuracy. Historical fixed-rank V6 trained 306 total parameters;
MDL-Evidence V6 trained 22, a 92.8% reduction. The result is stored in
`validation/mdl_vs_fixed_v6_cnn.json`.

A full ViT performance rerun was not included because the CPU validation job
exceeded the execution limit. Transformer-path calibration is nevertheless
covered by the architecture-neutral TinyTokenNet automated test.
