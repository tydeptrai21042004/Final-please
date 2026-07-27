# Support matrix

| Method | Families | Tasks | Implementation contract |
|---|---|---|---|
| Full tuning | all supported | single-label, multilabel, regression | all parameters |
| Linear probe | all supported | single-label, multilabel, regression | task head only |
| **Cross-Fitted Tangent-Core V6 (`trso`)** | supported models with eligible floating matrix weights | single-label, multilabel, regression | automatic two-sided backbone updates; full task head; exact merge |
| Conv-Adapter | ResNet-50 | supported tasks | paper insertion schemes |
| LoRA | Transformer | supported tasks | attention Q/V projections |
| BitFit | Transformer | supported tasks | biases and task head |
| SSF | supported ConvNeXt/ViT/Swin | supported tasks | published affine insertion points |
| AdaptFormer | ViT | supported tasks | parallel FFN adapters |
| Piggyback | CNN/ResNet | supported tasks | binary frozen-weight masks |
| BAM | ResNet-50 | supported tasks | stage attention modules |
| Residual Adapter | dedicated ResNet-26 | supported tasks | paper architecture |
| Side-Tuning | ResNet | supported tasks | frozen base plus side network |

For `trso`, normalization and embedding parameters are excluded from tangent cores. Matrix-shaped Conv/Linear weights are handled architecture-neutrally. The full task head is trained under the same shared-head fair protocol as compatible PEFT baselines.
