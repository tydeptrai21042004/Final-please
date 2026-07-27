# Support matrix

| Method | Families | Tasks | Implementation contract |
|---|---|---|---|
| Full tuning | all supported | single-label, multilabel, regression | all parameters |
| Linear probe | all supported | single-label, multilabel, regression | task head only |
| **MDL Tangent-Core V6 (`trso`)** | all supported models with eligible floating matrix weights | single-label, multilabel, regression | MDL-selected two-sided weight updates; automatic head policy; exact merge |
| Conv-Adapter | ResNet-50 | supported tasks | paper insertion schemes |
| LoRA | Transformer | supported tasks | attention Q/V projections |
| BitFit | Transformer | supported tasks | biases and task head |
| SSF | supported ConvNeXt/ViT/Swin | supported tasks | published affine insertion points |
| AdaptFormer | ViT | supported tasks | parallel FFN adapters |
| Piggyback | CNN/ResNet | supported tasks | binary frozen-weight masks |
| BAM | ResNet-50 | supported tasks | stage attention modules |
| Residual Adapter | dedicated ResNet-26 | supported tasks | paper architecture |
| Side-Tuning | ResNet | supported tasks | frozen base plus side network |

For `trso`, normalization and embedding parameters are excluded from tangent
cores. Matrix-shaped Conv/Linear weights are handled architecture-neutrally.
The task head is selected automatically among frozen, tangent, and full modes.
