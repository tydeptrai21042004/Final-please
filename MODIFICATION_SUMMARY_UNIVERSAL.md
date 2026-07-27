# Correction report

The repository now uses only MDL-Evidence Adaptive Tangent-Core V6.

## Corrected scientific issues

1. **Classifier over-compression:** replaced the chance-level readiness gate
   with automatic frozen/tangent/full head BIC selection.
2. **ResNet BatchNorm drift:** calibration now runs in evaluation mode and
   preserves running buffers exactly.
3. **Large flattened bases:** replaced with compact two-sided
   \(U_\ell K_\ell V_\ell^\top\) bases.
4. **Manual fixed rank:** replaced with MDL selection including rank zero.
5. **Manual dense/diagonal choice:** replaced with cross-fold BIC rescue.
6. **All-layer selection:** unsupported layers automatically receive rank zero.
7. **Incomplete efficiency reporting:** calibration output and aggregate CSV now
   report adapter coordinates, head parameters, and frozen basis values.

## Verification

- 82 repository tests pass.
- The fake-data ResNet-18 smoke run completes calibration, one training epoch,
  validation checkpoint selection, checkpoint reload, exact merge, and final
  testing.
