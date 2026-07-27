"""Cross-Fitted Evidence Adaptive Tangent-Core TRSO V6.

This module implements a proposal-capacity rule with no manual rank, layer,
parameter-budget, energy-threshold, core-type, head-policy, or calibration-
length controls.

For every eligible backbone tensor W_l, the complete calibration loader is
split deterministically into odd/even folds.  Let G_l^(0), G_l^(1) be their
mean task gradients and let U_l, V_l be the singular bases of the pooled mean.
For a tangent coordinate (i,j), its fold coefficients are

    c_ij^(f) = u_i^T G_l^(f) v_j.

Fitting the coordinate on one fold and predicting the other gives the exact
symmetric held-out reduction

    gain_ij = 4 c_ij^(0)c_ij^(1) - (c_ij^(0))^2 - (c_ij^(1))^2.

A coordinate is retained iff this gain is strictly positive.  Thus every
selected coordinate improves cross-fold prediction relative to the zero
update; unstable/noisy coordinates are discarded without a user threshold.
Leading stable diagonal coordinates define the layer rank, and stable
cross-mode coordinates provide a sparse automatic rescue when they also have
positive held-out evidence.

The task classifier is always fully trainable.  It is task-specific rather
than reusable pretrained structure, and compressing/freezing it was the main
failure mode of the previous MDL version.  Calibration disables dropout while using target-batch BatchNorm statistics; every BatchNorm buffer and module state is restored exactly afterward.  Learned updates merge exactly
into the original weights for adapter-free deployment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable, Iterator, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class MDLTangentRecord:
    name: str
    shape: tuple[int, ...]
    role: str
    rank: int
    core_mode: str
    core_parameters: int
    basis_values: int
    total_gradient_energy: float
    noise_energy: float
    captured_mean_energy: float
    mdl_zero: float
    mdl_selected: float
    diagonal_bic: float | None
    dense_bic: float | None
    stable_diagonal_coordinates: int = 0
    stable_cross_coordinates: int = 0
    heldout_predictive_gain: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MDLTangentReport:
    method: str
    calibration_batches: int
    calibration_examples: int
    candidate_tensors: int
    selected_tensors: int
    skipped_tensors: int
    adapter_parameters: int
    head_policy: str
    head_trainable_parameters: int
    frozen_basis_values: int
    fallback_reason: str | None
    records: tuple[MDLTangentRecord, ...]
    calibration_mean_loss: float | None = None
    chance_reference_loss: float | None = None
    dense_rescue_activated: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True)
class _Candidate:
    module_name: str
    parameter_name: str
    module: nn.Module
    parameter: Tensor
    is_head: bool

    @property
    def name(self) -> str:
        return f"{self.module_name}.{self.parameter_name}" if self.module_name else self.parameter_name

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.parameter.shape)


@dataclass
class _Statistic:
    count: int
    fold_counts: list[int]
    fold_sums: list[Tensor]
    squared_norm_sum: float

    @classmethod
    def empty(cls, parameter: Tensor) -> "_Statistic":
        shape = tuple(parameter.shape)
        return cls(
            count=0,
            fold_counts=[0, 0],
            fold_sums=[
                torch.zeros(shape, dtype=torch.float32, device="cpu"),
                torch.zeros(shape, dtype=torch.float32, device="cpu"),
            ],
            squared_norm_sum=0.0,
        )

    def update(self, gradient: Tensor, fold: int) -> None:
        value = gradient.detach().to(device="cpu", dtype=torch.float32)
        self.fold_sums[fold].add_(value)
        self.squared_norm_sum += float(value.square().sum().item())
        self.count += 1
        self.fold_counts[fold] += 1

    def mean(self) -> Tensor:
        if self.count <= 0:
            raise RuntimeError("gradient statistic has no observations")
        return (self.fold_sums[0] + self.fold_sums[1]) / float(self.count)

    def fold_mean(self, fold: int) -> Tensor | None:
        count = self.fold_counts[fold]
        if count <= 0:
            return None
        return self.fold_sums[fold] / float(count)

    def noise_sse(self) -> float:
        mean_energy = float(self.mean().double().square().sum().item())
        return max(0.0, self.squared_norm_sum - self.count * mean_energy)


class MDLTangentCoreParametrization(nn.Module):
    """Sparse two-sided task-response update selected by cross-fold evidence."""

    def __init__(
        self,
        left_basis: Tensor,
        right_basis: Tensor,
        original_shape: Iterable[int],
        *,
        coordinate_rows: Tensor | None = None,
        coordinate_columns: Tensor | None = None,
        dense_core: bool | None = None,
    ) -> None:
        super().__init__()
        if left_basis.ndim != 2 or right_basis.ndim != 2:
            raise ValueError("left_basis and right_basis must be matrices")
        if left_basis.shape[1] != right_basis.shape[1]:
            raise ValueError("left/right basis ranks must match")
        rank = int(left_basis.shape[1])
        if rank <= 0:
            raise ValueError("rank must be positive")
        shape = tuple(int(value) for value in original_shape)
        if len(shape) < 2:
            raise ValueError("tangent-core parameters require ndim >= 2")
        flat_input = math.prod(shape[1:])
        if left_basis.shape[0] != shape[0] or right_basis.shape[0] != flat_input:
            raise ValueError("basis dimensions do not match the original tensor")

        # Backward-compatible construction for older checkpoints/tests.
        if coordinate_rows is None or coordinate_columns is None:
            if dense_core:
                rows = torch.arange(rank).repeat_interleave(rank)
                columns = torch.arange(rank).repeat(rank)
            else:
                rows = torch.arange(rank)
                columns = torch.arange(rank)
        else:
            rows = coordinate_rows.detach().to(device="cpu", dtype=torch.long).flatten()
            columns = coordinate_columns.detach().to(device="cpu", dtype=torch.long).flatten()
        if rows.numel() == 0 or rows.numel() != columns.numel():
            raise ValueError("at least one matched tangent coordinate is required")
        if int(rows.min()) < 0 or int(columns.min()) < 0 or int(rows.max()) >= rank or int(columns.max()) >= rank:
            raise ValueError("coordinate index exceeds tangent rank")

        self.original_shape = shape
        self.rank = rank
        self.dense_core = bool(rows.numel() == rank * rank)
        self.register_buffer("left_basis", left_basis.detach().contiguous())
        self.register_buffer("right_basis", right_basis.detach().contiguous())
        self.register_buffer("coordinate_rows", rows.contiguous())
        self.register_buffer("coordinate_columns", columns.contiguous())
        self.core = nn.Parameter(
            torch.zeros(rows.numel(), dtype=left_basis.dtype, device=left_basis.device)
        )

    def delta_matrix(self) -> Tensor:
        left = self.left_basis.index_select(1, self.coordinate_rows)
        right = self.right_basis.index_select(1, self.coordinate_columns)
        return (left * self.core.unsqueeze(0)) @ right.transpose(0, 1)

    def forward(self, parameter: Tensor) -> Tensor:
        return parameter + self.delta_matrix().reshape(self.original_shape).to(dtype=parameter.dtype)

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.core.numel())

    @property
    def basis_value_count(self) -> int:
        return int(self.left_basis.numel() + self.right_basis.numel())


def _default_is_head(name: str) -> bool:
    parts = [part for part in name.lower().split(".") if part]
    modules = parts[:-1]
    if any(part in {"head", "heads", "classifier", "classifiers", "logits", "output"} for part in modules):
        return True
    return len(modules) == 1 and modules[0] in {"fc", "linear"}


def _is_normalization_or_embedding(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Embedding,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


def _iter_calibration_candidates(model: nn.Module, is_head: Callable[[str], bool]) -> Iterator[_Candidate]:
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in seen or not parameter.is_floating_point():
                continue
            seen.add(id(parameter))
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            head = bool(is_head(full_name))
            if head:
                # The full task head is trained directly and does not need a
                # gradient-history basis.
                continue
            if parameter.ndim < 2 or _is_normalization_or_embedding(module):
                continue
            yield _Candidate(module_name, parameter_name, module, parameter, False)


def _iter_head_parameters(model: nn.Module, is_head: Callable[[str], bool]) -> Iterator[tuple[str, Tensor]]:
    for name, parameter in model.named_parameters():
        if ".parametrizations." not in name and is_head(name):
            yield name, parameter


def _unpack_batch(batch, device: torch.device | str):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("calibration batches must provide at least (inputs, targets)")
    inputs, targets = batch[0], batch[1]
    if hasattr(inputs, "to"):
        inputs = inputs.to(device, non_blocking=True)
    if hasattr(targets, "to"):
        targets = targets.to(device, non_blocking=True)
    return inputs, targets


def _truncated_svd(matrix: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = matrix.shape
    resolved = max(1, min(int(rank), rows, columns))
    matrix = matrix.float()
    if min(rows, columns) <= 128 or matrix.numel() <= 262_144 or resolved == min(rows, columns):
        left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :resolved], singular[:resolved], right_h[:resolved]
    oversampling = max(4, int(math.ceil(math.log2(max(2, min(rows, columns))))))
    sketch = min(min(rows, columns), resolved + oversampling)
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(rows * 1009 + columns * 9176 + resolved * 53)
    omega = torch.randn(columns, sketch, generator=generator, device=matrix.device, dtype=matrix.dtype)
    projected = matrix @ omega
    for _ in range(2):
        projected = matrix @ (matrix.transpose(0, 1) @ projected)
    q, _ = torch.linalg.qr(projected, mode="reduced")
    compressed = q.transpose(0, 1) @ matrix
    small_left, singular, right_h = torch.linalg.svd(compressed, full_matrices=False)
    left = q @ small_left
    return left[:, :resolved], singular[:resolved], right_h[:resolved]


def _distributed_reduce_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return torch.device("cpu")
    backend = str(torch.distributed.get_backend()).lower()
    return resolved if "nccl" in backend else torch.device("cpu")


def _synchronize_statistics(
    statistics: dict[str, _Statistic], *, device, batches: int, examples: int,
    loss_sum: float, class_count: int | None,
) -> tuple[int, int, float, int | None]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return batches, examples, loss_sum, class_count
    reduce_device = _distributed_reduce_device(device)
    for statistic in statistics.values():
        for index in range(2):
            value = statistic.fold_sums[index].to(reduce_device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            statistic.fold_sums[index] = value.to(device="cpu", dtype=torch.float32)
        metadata = torch.tensor(
            [float(statistic.count), float(statistic.fold_counts[0]), float(statistic.fold_counts[1]), float(statistic.squared_norm_sum)],
            dtype=torch.float64,
            device=reduce_device,
        )
        torch.distributed.all_reduce(metadata, op=torch.distributed.ReduceOp.SUM)
        statistic.count = int(round(float(metadata[0].item())))
        statistic.fold_counts = [int(round(float(metadata[1].item()))), int(round(float(metadata[2].item())))]
        statistic.squared_norm_sum = float(metadata[3].item())
    totals = torch.tensor(
        [float(batches), float(examples), float(loss_sum)],
        dtype=torch.float64, device=reduce_device,
    )
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    classes = torch.tensor(float(class_count or 0), dtype=torch.float64, device=reduce_device)
    torch.distributed.all_reduce(classes, op=torch.distributed.ReduceOp.MAX)
    resolved_classes = int(round(float(classes.item()))) or None
    return (
        int(round(float(totals[0].item()))),
        int(round(float(totals[1].item()))),
        float(totals[2].item()),
        resolved_classes,
    )


def _collect_statistics(model, candidates, data_loader, loss_fn, *, device, logits_fn, batch_to_device):
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for candidate in candidates:
        candidate.parameter.requires_grad_(True)
    statistics = {candidate.name: _Statistic.empty(candidate.parameter) for candidate in candidates}
    was_training = model.training
    module_training = {id(module): module.training for module in model.modules()}
    batchnorm_state = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            batchnorm_state.append((
                module,
                module.running_mean.detach().clone() if module.running_mean is not None else None,
                module.running_var.detach().clone() if module.running_var is not None else None,
                module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else None,
            ))
    # Disable stochastic dropout, but let BatchNorm use target-batch statistics
    # while gradients are collected. Running buffers are restored exactly below.
    model.eval()
    for module, _, _, _ in batchnorm_state:
        module.train(True)
    batches = 0
    examples = 0
    loss_sum = 0.0
    class_count: int | None = None
    classification_compatible = True
    try:
        for batch in data_loader:
            inputs, targets = batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            logits = logits_fn(model(inputs))
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite tangent calibration loss")
            batch_examples = int(targets.shape[0]) if getattr(targets, "ndim", 0) else 1
            loss_sum += float(loss.detach().item()) * batch_examples
            if (
                isinstance(logits, Tensor) and logits.ndim == 2
                and getattr(targets, "ndim", 0) == 1
                and targets.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            ):
                observed_classes = int(logits.shape[-1])
                class_count = observed_classes if class_count in (None, observed_classes) else None
            else:
                classification_compatible = False
            loss.backward()
            fold = batches & 1
            for candidate in candidates:
                gradient = candidate.parameter.grad
                if gradient is not None:
                    statistics[candidate.name].update(gradient, fold)
            examples += batch_examples
            batches += 1
    finally:
        model.zero_grad(set_to_none=True)
        for module, running_mean, running_var, num_batches_tracked in batchnorm_state:
            if running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None:
                module.running_var.copy_(running_var)
            if num_batches_tracked is not None:
                module.num_batches_tracked.copy_(num_batches_tracked)
        for module in model.modules():
            module.train(module_training[id(module)])
        model.train(was_training)
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
    if batches == 0:
        raise RuntimeError("calibration loader produced no batches")
    if not classification_compatible:
        class_count = None
    batches, examples, loss_sum, class_count = _synchronize_statistics(
        statistics, device=device, batches=batches, examples=examples,
        loss_sum=loss_sum, class_count=class_count,
    )
    mean_loss = float(loss_sum / max(1, examples))
    return statistics, batches, examples, mean_loss, class_count


def _cross_fold_gain(first: Tensor, second: Tensor) -> Tensor:
    """Exact symmetric held-out improvement over the zero-coordinate predictor."""
    return 4.0 * first * second - first.square() - second.square()


def _analyze_weight(
    candidate: _Candidate, statistic: _Statistic, *, dense_rescue: bool,
):
    pooled = statistic.mean().float()
    matrix = pooled.reshape(pooled.shape[0], -1)
    first = statistic.fold_mean(0)
    second = statistic.fold_mean(1)
    total = float(pooled.double().square().sum().item())
    noise = statistic.noise_sse()
    if first is None or second is None:
        record = MDLTangentRecord(
            name=candidate.name, shape=candidate.shape, role="backbone", rank=0,
            core_mode="skipped", core_parameters=0, basis_values=0,
            total_gradient_energy=total, noise_energy=noise, captured_mean_energy=0.0,
            mdl_zero=0.0, mdl_selected=0.0, diagonal_bic=None, dense_bic=None,
        )
        return record, None, None, None, None

    identifiable_rank = min(matrix.shape[0], matrix.shape[1], max(1, min(statistic.fold_counts)))
    left, singular, right_h = _truncated_svd(matrix, identifiable_rank)
    right = right_h.transpose(0, 1)
    first_matrix = first.float().reshape(matrix.shape)
    second_matrix = second.float().reshape(matrix.shape)
    coordinates_first = left.transpose(0, 1) @ first_matrix @ right
    coordinates_second = left.transpose(0, 1) @ second_matrix @ right
    gains = _cross_fold_gain(coordinates_first, coordinates_second)

    diagonal_gains = torch.diagonal(gains)
    stable_diagonal = torch.nonzero(diagonal_gains > 0, as_tuple=False).flatten()
    if stable_diagonal.numel() == 0:
        record = MDLTangentRecord(
            name=candidate.name, shape=candidate.shape, role="backbone", rank=0,
            core_mode="skipped", core_parameters=0, basis_values=0,
            total_gradient_energy=total, noise_energy=noise, captured_mean_energy=0.0,
            mdl_zero=0.0, mdl_selected=0.0, diagonal_bic=None, dense_bic=None,
        )
        return record, None, None, None, None

    if dense_rescue:
        # When the geometric-mean probability assigned to the true class is
        # no better than the square-root chance boundary, preserve every
        # independently predictive leading mode and its full interaction core.
        # This is an automatic capacity rescue, not a user setting.
        predictive = diagonal_gains > 0
        unsupported = torch.nonzero(~predictive, as_tuple=False).flatten()
        intrinsic_rank = int(unsupported[0].item()) if unsupported.numel() else int(singular.numel())
    else:
        # Otherwise use the ceiling of the stable rank, a scale-free intrinsic
        # dimension estimator, and require positive cross-fold evidence for
        # every retained leading mode.
        stable_rank = float(
            singular.double().square().sum().item()
            / max(singular[0].double().square().item(), torch.finfo(torch.float64).tiny)
        )
        intrinsic_rank = max(1, min(int(singular.numel()), int(math.ceil(stable_rank))))
        predictive = diagonal_gains[:intrinsic_rank] > 0
        unsupported = torch.nonzero(~predictive, as_tuple=False).flatten()
        if unsupported.numel():
            intrinsic_rank = int(unsupported[0].item())
    if intrinsic_rank <= 0:
        record = MDLTangentRecord(
            name=candidate.name, shape=candidate.shape, role="backbone", rank=0,
            core_mode="skipped", core_parameters=0, basis_values=0,
            total_gradient_energy=total, noise_energy=noise, captured_mean_energy=0.0,
            mdl_zero=0.0, mdl_selected=0.0, diagonal_bic=None, dense_bic=None,
        )
        return record, None, None, None, None
    selected_modes = torch.arange(intrinsic_rank)
    left_selected = left.index_select(1, selected_modes).contiguous()
    right_selected = right.index_select(1, selected_modes).contiguous()
    selected_gain_matrix = gains.index_select(0, selected_modes).index_select(1, selected_modes)
    if dense_rescue:
        mask = torch.ones_like(selected_gain_matrix, dtype=torch.bool)
        predictive_gain = float(
            torch.diagonal(selected_gain_matrix).clamp_min(0).double().sum().item()
        )
    else:
        mask = selected_gain_matrix > 0
        predictive_gain = float(selected_gain_matrix[mask].double().sum().item())
    rows, columns = torch.nonzero(mask, as_tuple=True)
    diagonal_count = int(torch.diagonal(mask).sum().item())
    cross_count = int(mask.sum().item()) - diagonal_count
    captured = float(singular.index_select(0, selected_modes).double().square().sum().item())
    core_mode = (
        "dense_chance_rescue" if dense_rescue
        else ("diagonal" if cross_count == 0 else "sparse_cross_evidence")
    )
    basis_values = int(left_selected.numel() + right_selected.numel())
    # Retain legacy fields for result-table compatibility. They now hold
    # directly interpretable held-out risk values rather than invalid Bmn-BIC.
    zero_risk = float(first_matrix.double().square().sum().item() + second_matrix.double().square().sum().item())
    selected_risk = max(0.0, zero_risk - predictive_gain)
    record = MDLTangentRecord(
        name=candidate.name,
        shape=candidate.shape,
        role="backbone",
        rank=int(selected_modes.numel()),
        core_mode=core_mode,
        core_parameters=int(rows.numel()),
        basis_values=basis_values,
        total_gradient_energy=total,
        noise_energy=noise,
        captured_mean_energy=captured,
        mdl_zero=zero_risk,
        mdl_selected=selected_risk,
        diagonal_bic=None,
        dense_bic=None,
        stable_diagonal_coordinates=diagonal_count,
        stable_cross_coordinates=cross_count,
        heldout_predictive_gain=predictive_gain,
    )
    return record, left_selected, right_selected, rows, columns


def _attach(candidate, record, left, right, rows, columns):
    transformation = MDLTangentCoreParametrization(
        left.to(device=candidate.parameter.device, dtype=candidate.parameter.dtype),
        right.to(device=candidate.parameter.device, dtype=candidate.parameter.dtype),
        candidate.shape,
        coordinate_rows=rows,
        coordinate_columns=columns,
    )
    parametrize.register_parametrization(candidate.module, candidate.parameter_name, transformation)


def _needs_dense_rescue(calibration_mean_loss: float | None, class_count: int | None) -> bool:
    """Return the parameter-free square-root-chance capacity decision.

    For mean cross-entropy L and C classes, exp(-L) is the geometric mean
    probability assigned to the true class.  Dense rescue is activated when
    exp(-L) <= C^{-1/2}, equivalently L >= (1/2) log C.
    """
    return bool(
        calibration_mean_loss is not None
        and class_count is not None
        and class_count > 1
        and calibration_mean_loss >= 0.5 * math.log(class_count)
    )


def calibrate_mdl_tangent_core(
    model: nn.Module,
    data_loader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    device: torch.device | str,
    logits_fn: Optional[Callable[[object], Tensor]] = None,
    is_head: Optional[Callable[[str], bool]] = None,
    batch_to_device: Optional[Callable[[object, torch.device | str], tuple[Tensor, Tensor]]] = None,
) -> MDLTangentReport:
    """Calibrate and attach the corrected automatic tangent-core proposal."""
    logits_fn = logits_fn or (lambda output: output)
    is_head = is_head or _default_is_head
    batch_to_device = batch_to_device or _unpack_batch
    candidates = list(_iter_calibration_candidates(model, is_head))
    head_parameters = list(_iter_head_parameters(model, is_head))
    if not candidates and not head_parameters:
        raise RuntimeError("no eligible backbone or task-head parameters were found")

    if candidates:
        statistics, batches, examples, calibration_mean_loss, class_count = _collect_statistics(
            model, candidates, data_loader, loss_fn, device=device,
            logits_fn=logits_fn, batch_to_device=batch_to_device,
        )
    else:
        batches = len(data_loader) if hasattr(data_loader, "__len__") else 0
        examples = 0
        calibration_mean_loss = None
        class_count = None
        statistics = {}

    chance_reference_loss = math.log(class_count) if class_count and class_count > 1 else None
    dense_rescue = _needs_dense_rescue(calibration_mean_loss, class_count)

    records: list[MDLTangentRecord] = []
    for candidate in candidates:
        record, left, right, rows, columns = _analyze_weight(
            candidate, statistics[candidate.name], dense_rescue=dense_rescue,
        )
        records.append(record)
        if record.rank > 0 and left is not None and right is not None and rows is not None and columns is not None:
            _attach(candidate, record, left, right, rows, columns)

    # The task head is always fully trainable. This is part of the method, not
    # a dataset/backbone-specific option.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MDLTangentCoreParametrization):
            module.core.requires_grad_(True)
    head_trainable = 0
    for _, parameter in _iter_head_parameters(model, is_head):
        parameter.requires_grad_(True)
        head_trainable += int(parameter.numel())

    adapter_parameters = sum(
        module.trainable_parameter_count for module in model.modules()
        if isinstance(module, MDLTangentCoreParametrization)
    )
    basis_values = sum(
        module.basis_value_count for module in model.modules()
        if isinstance(module, MDLTangentCoreParametrization)
    )
    report = MDLTangentReport(
        method="cross_fitted_evidence_adaptive_tangent_core_v6",
        calibration_batches=batches,
        calibration_examples=examples,
        candidate_tensors=len(records),
        selected_tensors=sum(record.rank > 0 for record in records),
        skipped_tensors=sum(record.rank == 0 for record in records),
        adapter_parameters=adapter_parameters,
        head_policy="full",
        head_trainable_parameters=head_trainable,
        frozen_basis_values=basis_values,
        fallback_reason=None,
        records=tuple(records),
        calibration_mean_loss=calibration_mean_loss,
        chance_reference_loss=chance_reference_loss,
        dense_rescue_activated=dense_rescue,
    )
    payload = report.to_dict()
    payload["selection_rule"] = (
        "full_core_square_root_chance_rescue" if dense_rescue
        else "stable_rank_positive_cross_fold_predictive_gain"
    )
    payload["head_policy_scores"] = {}
    model._mdl_tangent_report = payload  # type: ignore[attr-defined]
    return report


def set_mdl_tangent_trainability(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MDLTangentCoreParametrization):
            module.core.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if ".parametrizations." not in name and _default_is_head(name):
            parameter.requires_grad_(True)


def iter_mdl_tangent_parametrizations(model: nn.Module):
    for module_name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name, sequence in list(parametrizations.items()):
            for transformation in sequence:
                if isinstance(transformation, MDLTangentCoreParametrization):
                    yield module_name, module, parameter_name, transformation


def mdl_tangent_parameter_count(model: nn.Module) -> int:
    return sum(transformation.trainable_parameter_count for _, _, _, transformation in iter_mdl_tangent_parametrizations(model))


def mdl_tangent_basis_value_count(model: nn.Module) -> int:
    return sum(transformation.basis_value_count for _, _, _, transformation in iter_mdl_tangent_parametrizations(model))


@torch.no_grad()
def merge_mdl_tangent_cores_(model: nn.Module) -> int:
    targets: list[tuple[nn.Module, str]] = []
    seen: set[tuple[int, str]] = set()
    for _, module, parameter_name, _ in iter_mdl_tangent_parametrizations(model):
        key = (id(module), parameter_name)
        if key not in seen:
            seen.add(key)
            targets.append((module, parameter_name))
    for module, parameter_name in targets:
        parametrize.remove_parametrizations(module, parameter_name, leave_parametrized=True)
    return len(targets)


__all__ = [
    "MDLTangentCoreParametrization",
    "MDLTangentRecord",
    "MDLTangentReport",
    "calibrate_mdl_tangent_core",
    "iter_mdl_tangent_parametrizations",
    "mdl_tangent_basis_value_count",
    "mdl_tangent_parameter_count",
    "merge_mdl_tangent_cores_",
    "set_mdl_tangent_trainability",
]
