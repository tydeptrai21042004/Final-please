# MDL-Evidence Adaptive Tangent-Core V6

## 1. Proposal

MDL-Evidence Adaptive Tangent-Core V6 is the only active TRSO proposal in this
repository. It replaces the former Information-Spectral implementation and all
legacy fixed-rank tangent-core variants.

For every eligible downstream weight tensor \(W_\ell\), reshape its first axis
against all remaining axes:

\[
W_\ell\in\mathbb R^{m_\ell\times n_\ell},
\qquad n_\ell=\prod_{j\ge 2}\operatorname{shape}(W_\ell)_j.
\]

The adapted weight is

\[
W_\ell^{\mathrm{eff}}
=
W_\ell+U_{\ell,r_\ell}K_\ell V_{\ell,r_\ell}^{\top}.
\]

The proposal automatically decides:

1. whether the layer is adapted (\(r_\ell=0\) skips it);
2. the layer rank \(r_\ell\);
3. whether \(K_\ell\) is diagonal or dense;
4. whether the task head is frozen, tangent-compressed, or fully trainable.

No proposal-specific rank, parameter budget, energy threshold, layer list,
core-mode switch, head-policy switch, or calibration-batch count is exposed.
The ordinary training protocol still specifies optimizer, learning rate,
weight decay, epochs, and augmentation, just as it does for every baseline.

## 2. BatchNorm-safe complete-loader calibration

Let \(G_{\ell,b}=\partial\mathcal L_b/\partial W_\ell\) be the gradient from
calibration batch \(b\), and let \(B\) be the number of batches in the complete
training loader. Calibration runs with `model.eval()` while gradients remain
enabled. Consequently:

- BatchNorm running means and variances are unchanged;
- dropout is disabled;
- the zero-initialized proposal exactly preserves the pre-calibration function.

Only sufficient statistics are retained:

\[
\bar G_\ell=\frac1B\sum_{b=1}^{B}G_{\ell,b},
\]

and

\[
E_{\ell,\mathrm{noise}}
=
\sum_{b=1}^{B}\|G_{\ell,b}\|_F^2
-B\|\bar G_\ell\|_F^2.
\]

Two deterministic odd/even fold sums are also retained for core-structure
selection. The implementation therefore needs two gradient-sized CPU buffers,
not one full gradient history per batch.

## 3. Data-identifiable candidate spectrum

Compute a truncated SVD of the mean response:

\[
\bar G_\ell=U_\ell\Sigma_\ell V_\ell^\top.
\]

The largest statistically identifiable candidate rank is derived from data:

\[
r_{\ell,\max}
=
\min\{m_\ell,n_\ell,\max(1,B-1)\}.
\]

Thus there is no manual `max_rank`. Small matrices use exact SVD. Large
matrices use a deterministic randomized SVD only as a numerical accelerator;
the selected statistical rank is still determined by MDL.

## 4. MDL layer and rank selection

For candidate rank \(r\), define the residual description error

\[
\operatorname{RSS}_\ell(r)
=
E_{\ell,\mathrm{noise}}
+B\left(
\|\bar G_\ell\|_F^2-
\sum_{j=1}^{r}\sigma_{\ell,j}^2
\right).
\]

Rank zero is included. The description complexity is

\[
k_\ell(r)
=
\begin{cases}
0,&r=0,\\
r(m_\ell+n_\ell-r)+r,&r>0,
\end{cases}
\]

where the first term describes the two-sided task-response subspace and the
last term describes a diagonal trainable core. With
\(N_\ell=B m_\ell n_\ell\), the layer score is

\[
\operatorname{MDL}_\ell(r)
=
N_\ell\log\!\left(
\frac{\operatorname{RSS}_\ell(r)}{N_\ell}
\right)
+k_\ell(r)\log N_\ell.
\]

The selected rank is

\[
r_\ell^*=\arg\min_{0\le r\le r_{\ell,\max}}
\operatorname{MDL}_\ell(r).
\]

A layer is omitted automatically when \(r_\ell^*=0\).

## 5. Diagonal core and automatic dense rescue

The parsimonious update is

\[
K_\ell=\operatorname{diag}(\alpha_{\ell,1},\ldots,
\alpha_{\ell,r_\ell}),
\]

which uses only \(r_\ell\) trainable coordinates. A dense core would use
\(r_\ell^2\) coordinates.

To determine whether cross-mode interactions are supported, calibration batches
are divided deterministically into odd and even folds. A core fitted from one
fold is evaluated against the other fold, in both directions. Let
\(R_{\ell,\mathrm{diag}}\) and \(R_{\ell,\mathrm{dense}}\) denote the resulting
cross-fold reconstruction errors. The two BIC scores are

\[
\operatorname{BIC}_{\ell,\mathrm{diag}}
=N_\ell\log(R_{\ell,\mathrm{diag}}/N_\ell)
+r_\ell\log N_\ell,
\]

\[
\operatorname{BIC}_{\ell,\mathrm{dense}}
=N_\ell\log(R_{\ell,\mathrm{dense}}/N_\ell)
+r_\ell^2\log N_\ell.
\]

The dense core is used only if its score is lower. Rank-one layers are always
diagonal because the two structures coincide.

## 6. Automatic task-head policy

Let \(H\) be the number of task-head parameters. The method compares three
models using the complete head-gradient evidence:

- **Frozen:** zero additional head parameters;
- **Tangent:** MDL-selected tangent weight updates plus ordinary head biases;
- **Full:** all head parameters trainable.

For each policy \(q\), its residual evidence error is
\(\operatorname{RSS}_{h,q}\), its trainable complexity is \(k_{h,q}\), and

\[
\operatorname{BIC}_{h,q}
=BH\log\!\left(
\frac{\operatorname{RSS}_{h,q}}{BH}
\right)
+k_{h,q}\log(BH).
\]

The minimum-score policy is selected. If every candidate selects zero
coordinates, the ordinary full head is enabled as the unique non-degenerate
training fallback; the reason is written to the calibration report.

## 7. Parameter and deployment complexity

For a selected layer:

- diagonal core parameters: \(r_\ell\);
- dense core parameters: \(r_\ell^2\);
- frozen basis values: \(r_\ell(m_\ell+n_\ell)\).

This is substantially smaller than storing full flattened directions of size
\(r_\ell m_\ell n_\ell\). After training, each update is merged exactly:

\[
W_\ell\leftarrow
W_\ell+U_{\ell,r_\ell}K_\ell V_{\ell,r_\ell}^\top.
\]

The PyTorch parametrization is then removed. The deployed model has the original
architecture, original number of parameters, and no adapter operation in the
forward graph.

## 8. Algorithm

1. Load the pretrained backbone and the shared task-aware linear head.
2. Freeze all parameters, then enable gradients only for eligible matrix-shaped
   backbone tensors and task-head tensors.
3. Run the complete training loader in evaluation mode and accumulate gradient
   sufficient statistics.
4. For each backbone tensor, select \(r_\ell^*\) by MDL.
5. For each selected tensor, choose diagonal or dense core by cross-fold BIC.
6. Select the head policy by BIC.
7. Attach zero-initialized tangent cores and enable only evidence-selected
   coordinates.
8. Train under the common fair protocol.
9. Restore the best validation checkpoint.
10. Merge all tangent updates exactly before final evaluation when
    `--trso_fast_inference True`.

## 9. Implementation outputs

Every run writes `mdl_tangent_calibration.json`, including:

- calibration batch and example counts;
- candidate, selected, and skipped tensor counts;
- selected rank and core type for every tensor;
- MDL and BIC scores;
- adapter parameter count;
- head policy and head parameter count;
- frozen basis value count;
- any non-degenerate fallback reason.

The aggregate runner exports these fields with the `mdl_` prefix in raw and
paper-facing CSV files.
