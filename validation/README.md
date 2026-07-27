# Executed validation artifacts

- `CORRECTED_V6_VALIDATION.json`: combined rows and summary statistics.
- `reproduced_standard_seeds_0_1_2.json`: three-seed standard CNN/ViT comparison.
- `reproduced_hard_seeds_0_1_2_3_4_5.json`: six-seed difficult CNN comparison.
- `final_smoke/`: end-to-end CPU runner output proving checkpoint fallback and exact merge.

The controlled validator compares the active proposal with a test-only historical fixed-rank V6 reference. It does not use real DTD test labels and does not replace the required GPU DTD rerun.
