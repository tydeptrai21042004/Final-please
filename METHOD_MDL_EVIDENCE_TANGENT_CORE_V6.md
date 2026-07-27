# MDL-Evidence Adaptive Tangent-Core V6

The active implementation is in `models/tuning_modules/mdl_tangent_core.py`. The filename is retained for checkpoint compatibility; the corrected selector uses cross-fitted predictive evidence rather than the invalid former `Bmn` information count.

## 1. Structured update

For every eligible backbone tensor, reshaped as
\(W_\ell\in\mathbb R^{m_\ell\times n_\ell}\),

\[
W_\ell^{\mathrm{eff}}
=
W_\ell+U_\ell K_\ell V_\ell^{\top}.
\]

The bases are frozen after calibration. Only the selected entries of \(K_\ell\) and the complete task classifier are trained. The update is merged exactly into the backbone before final inference.

## 2. Cross-fitted evidence

The complete calibration loader is split deterministically into alternating folds. Let \(G_\ell^{(0)}\) and \(G_\ell^{(1)}\) denote the fold-mean gradients. The SVD of their pooled mean supplies \(U_\ell,V_\ell\). For coordinate \((i,j)\),

\[
c_{ij}^{(f)}=u_i^\top G_\ell^{(f)}v_j.
\]

Its exact symmetric held-out improvement over the zero-coordinate predictor is

\[
\Delta_{ij}
=
4c_{ij}^{(0)}c_{ij}^{(1)}
-
\bigl(c_{ij}^{(0)}\bigr)^2
-
\bigl(c_{ij}^{(1)}\bigr)^2.
\]

A coordinate is supported only when \(\Delta_{ij}>0\). Thus unstable directions are removed without a manually chosen gain threshold.

## 3. Automatic intrinsic rank

For an informative transferred classifier, the initial layer rank is the ceiling of the stable rank,

\[
r_\ell
=
\left\lceil
\frac{\sum_j\sigma_{\ell,j}^2}{\sigma_{\ell,1}^2}
\right\rceil,
\]

and is truncated at the first unsupported leading diagonal mode. Positive cross-mode evidence may add sparse off-diagonal entries. No rank, maximum rank, energy percentage, layer list, or parameter budget is exposed.

## 4. Parameter-free capacity rescue

Let \(\overline L_{\mathrm{cal}}\) be the mean calibration cross-entropy and let \(C\) be the number of classes. Then

\[
\exp(-\overline L_{\mathrm{cal}})
\]

is the geometric mean probability assigned to the true class. Dense rescue is activated exactly when

\[
\exp(-\overline L_{\mathrm{cal}})\leq C^{-1/2},
\]

or equivalently

\[
\overline L_{\mathrm{cal}}\geq \frac12\log C.
\]

At this square-root-chance boundary, all independently predictive leading modes receive a full interaction core. The rule has no dataset-specific threshold and recovered the difficult controlled cases in which the smaller diagonal selector collapsed.

## 5. Full task classifier

The classifier is always fully trainable:

\[
\boxed{\text{full task head}+\text{automatic tangent backbone}.}
\]

The former BIC policy froze the DTD task head and was the main cause of the observed underperformance.

## 6. BatchNorm-safe target calibration

Dropout is disabled during calibration, while BatchNorm uses target-batch statistics to form target-domain gradients. Every running mean, running variance, counter, and module training state is restored exactly afterward. Consequently, all zero-initialized tangent coordinates preserve the starting function.

## 7. Validation safeguard

The zero-update shared-head model is registered as the initial best checkpoint. A trained epoch replaces it only when the validation objective improves. The final selected model therefore cannot be worse than the starting shared-head model on the validation criterion.

## 8. Public controls

The proposal API exposes no rank, maximum rank, calibration length, layer set, parameter budget, energy threshold, core type, evidence threshold, or head policy. Optimizer, learning rate, epochs, augmentation, and batch size remain common benchmark settings shared with competing methods.
