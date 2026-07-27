# Validation artifacts

## `mdl_vs_fixed_v6_cnn.json`

Offline controlled source-to-target texture transfer on the same TinyCNN,
training schedule, data split, and seed for both methods.

| Method | Accuracy | Adapter parameters | Head parameters | Total trainable | Frozen basis values |
|---|---:|---:|---:|---:|---:|
| Historical fixed-rank V6 | 100.0% | 12 | 294 | 306 | 1,110 |
| MDL-Evidence V6 | 100.0% | 16 | 6 | 22 | 2,159 |

The new method preserved accuracy while reducing total trainable parameters by
92.8%. It selected a tangent-compressed head automatically. This is a controlled
sanity check, not a replacement for the required multi-seed DTD benchmark.
