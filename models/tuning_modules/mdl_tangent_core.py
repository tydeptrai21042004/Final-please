"""MDL-Evidence Adaptive Tangent-Core TRSO.

This module replaces every previous TRSO proposal implementation.  It exposes
no proposal-specific rank, layer list, parameter budget, energy threshold,
core type, head policy, or calibration-length hyperparameter.

For every eligible weight tensor W_l, the complete calibration loader provides
batch gradients.  Their sufficient statistics define a mean task response
G_l and its across-batch noise.  The update is

    W_l^eff = W_l + U_l K_l V_l^T,

where the rank is selected by a minimum-description-length (MDL) criterion that
admits rank zero.  A diagonal core is the parsimonious default; a deterministic
cross-fold BIC comparison promotes it to a dense core only when off-diagonal
interactions improve out-of-fold gradient reconstruction enough to pay for the
extra coordinates.  The task head is selected automatically among frozen,
tangent-compressed, and fully trainable policies by the same evidence/complexity
principle.

Calibration is performed in evaluation mode, so BatchNorm running statistics
and dropout state are not mutated.  All learned updates can be merged exactly
into the original weights for an adapter-free inference graph.
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
    """Two-sided task-response update with an automatically selected core."""

    def __init__(
        self,
        left_basis: Tensor,
        right_basis: Tensor,
        original_shape: Iterable[int],
        *,
        dense_core: bool,
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

        self.original_shape = shape
        self.rank = rank
        self.dense_core = bool(dense_core)
        self.register_buffer("left_basis", left_basis.detach().contiguous())
        self.register_buffer("right_basis", right_basis.detach().contiguous())
        if self.dense_core:
            self.core = nn.Parameter(
                torch.zeros(rank, rank, dtype=left_basis.dtype, device=left_basis.device)
            )
        else:
            self.core = nn.Parameter(
                torch.zeros(rank, dtype=left_basis.dtype, device=left_basis.device)
            )

    def delta_matrix(self) -> Tensor:
        if self.dense_core:
            return self.left_basis @ self.core @ self.right_basis.transpose(0, 1)
        return (self.left_basis * self.core.unsqueeze(0)) @ self.right_basis.transpose(0, 1)

    def forward(self, parameter: Tensor) -> Tensor:
        delta = self.delta_matrix().reshape(self.original_shape)
        return parameter + delta.to(dtype=parameter.dtype)

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


def _iter_calibration_candidates(
    model: nn.Module,
    is_head: Callable[[str], bool],
) -> Iterator[_Candidate]:
    """Yield unique floating tensors needed by backbone and head selection."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in seen or not parameter.is_floating_point():
                continue
            seen.add(id(parameter))
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            head = bool(is_head(full_name))
            if head:
                yield _Candidate(module_name, parameter_name, module, parameter, True)
                continue
            if parameter.ndim < 2 or _is_normalization_or_embedding(module):
                continue
            yield _Candidate(module_name, parameter_name, module, parameter, False)


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
    """Deterministic exact-small/randomized-large SVD up to a data-derived rank."""
    rows, columns = matrix.shape
    resolved = max(1, min(int(rank), rows, columns))
    matrix = matrix.float()
    if min(rows, columns) <= 128 or matrix.numel() <= 262_144 or resolved == min(rows, columns):
        left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :resolved], singular[:resolved], right_h[:resolved]

    # Oversampling and power iterations are numerical approximation constants,
    # not proposal controls.  The statistical rank itself is set only by MDL.
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


def _safe_information_score(rss: float, observations: int, complexity: int) -> float:
    observations = max(2, int(observations))
    scale = max(float(rss) / observations, torch.finfo(torch.float64).tiny)
    return observations * math.log(scale) + int(complexity) * math.log(observations)


def _rank_mdl(
    statistic: _Statistic,
    matrix: Tensor,
    singular: Tensor,
) -> tuple[int, list[float], list[float], float]:
    """Select rank, including zero, from gradient fit versus description length."""
    batches = statistic.count
    rows, columns = matrix.shape
    values = rows * columns
    observations = batches * values
    mean_energy = float(matrix.double().square().sum().item())
    noise = statistic.noise_sse()
    cumulative = 0.0
    scores: list[float] = []
    residuals: list[float] = []
    for rank in range(0, int(singular.numel()) + 1):
        if rank:
            cumulative += float(singular[rank - 1].double().square().item())
        mean_residual = max(0.0, mean_energy - cumulative)
        rss = noise + batches * mean_residual
        # The code length includes the data-derived two-sided subspace and the
        # diagonal trainable coordinates.  This prevents large layers from
        # receiving unsupported modes merely because they contain many entries.
        complexity = 0 if rank == 0 else rank * (rows + columns - rank) + rank
        scores.append(_safe_information_score(rss, observations, complexity))
        residuals.append(rss)
    selected = min(range(len(scores)), key=lambda index: (scores[index], index))
    return selected, scores, residuals, noise


def _core_bic(
    statistic: _Statistic,
    left: Tensor,
    right: Tensor,
) -> tuple[bool, float | None, float | None]:
    """Cross-fold evidence test for diagonal versus dense tangent interactions."""
    first = statistic.fold_mean(0)
    second = statistic.fold_mean(1)
    rank = int(left.shape[1])
    if first is None or second is None or rank <= 1:
        return False, None, None
    first_matrix = first.float().reshape(left.shape[0], -1)
    second_matrix = second.float().reshape(left.shape[0], -1)
    right_t = right.transpose(0, 1)

    def prediction_error(fit: Tensor, validation: Tensor, dense: bool) -> float:
        coordinates = left.transpose(0, 1) @ fit @ right
        if not dense:
            coordinates = torch.diag(torch.diagonal(coordinates))
        prediction = left @ coordinates @ right_t
        return float((validation - prediction).double().square().sum().item())

    diagonal_rss = (
        statistic.fold_counts[1] * prediction_error(first_matrix, second_matrix, False)
        + statistic.fold_counts[0] * prediction_error(second_matrix, first_matrix, False)
    )
    dense_rss = (
        statistic.fold_counts[1] * prediction_error(first_matrix, second_matrix, True)
        + statistic.fold_counts[0] * prediction_error(second_matrix, first_matrix, True)
    )
    observations = statistic.count * first_matrix.numel()
    diagonal_bic = _safe_information_score(diagonal_rss, observations, rank)
    dense_bic = _safe_information_score(dense_rss, observations, rank * rank)
    return dense_bic < diagonal_bic, diagonal_bic, dense_bic



def _distributed_reduce_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return torch.device("cpu")
    backend = str(torch.distributed.get_backend()).lower()
    return resolved if "nccl" in backend else torch.device("cpu")


def _synchronize_statistics(
    statistics: dict[str, _Statistic],
    *,
    device: torch.device | str,
    batches: int,
    examples: int,
) -> tuple[int, int]:
    """All-reduce calibration evidence before DDP wrapping.

    The fair runner normally schedules one process per GPU.  This synchronization
    also makes native distributed runs safe: every rank derives identical MDL
    ranks, core structures, and parametrization shapes.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return batches, examples
    reduce_device = _distributed_reduce_device(device)
    for statistic in statistics.values():
        for index in range(2):
            value = statistic.fold_sums[index].to(reduce_device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            statistic.fold_sums[index] = value.to(device="cpu", dtype=torch.float32)
        metadata = torch.tensor(
            [
                float(statistic.count),
                float(statistic.fold_counts[0]),
                float(statistic.fold_counts[1]),
                float(statistic.squared_norm_sum),
            ],
            dtype=torch.float64,
            device=reduce_device,
        )
        torch.distributed.all_reduce(metadata, op=torch.distributed.ReduceOp.SUM)
        statistic.count = int(round(float(metadata[0].item())))
        statistic.fold_counts = [
            int(round(float(metadata[1].item()))),
            int(round(float(metadata[2].item()))),
        ]
        statistic.squared_norm_sum = float(metadata[3].item())
    totals = torch.tensor([float(batches), float(examples)], dtype=torch.float64, device=reduce_device)
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return int(round(float(totals[0].item()))), int(round(float(totals[1].item())))

def _collect_statistics(
    model: nn.Module,
    candidates: list[_Candidate],
    data_loader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    device: torch.device | str,
    logits_fn: Callable[[object], Tensor],
    batch_to_device: Callable[[object, torch.device | str], tuple[Tensor, Tensor]],
) -> tuple[dict[str, _Statistic], int, int]:
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for candidate in candidates:
        candidate.parameter.requires_grad_(True)

    statistics = {candidate.name: _Statistic.empty(candidate.parameter) for candidate in candidates}
    was_training = model.training
    model.eval()  # Critical: preserve BatchNorm buffers and disable dropout.
    batches = 0
    examples = 0
    try:
        for batch in data_loader:
            inputs, targets = batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            logits = logits_fn(model(inputs))
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MDL tangent calibration loss")
            loss.backward()
            fold = batches & 1
            for candidate in candidates:
                gradient = candidate.parameter.grad
                if gradient is not None:
                    statistics[candidate.name].update(gradient, fold)
            examples += int(targets.shape[0]) if getattr(targets, "ndim", 0) else 1
            batches += 1
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])

    if batches == 0:
        raise RuntimeError("calibration loader produced no batches")
    batches, examples = _synchronize_statistics(
        statistics, device=device, batches=batches, examples=examples
    )
    return statistics, batches, examples


def _analyze_weight(
    candidate: _Candidate,
    statistic: _Statistic,
) -> tuple[MDLTangentRecord, Tensor | None, Tensor | None, float]:
    mean = statistic.mean().float()
    matrix = mean.reshape(mean.shape[0], -1)
    identifiable_rank = min(matrix.shape[0], matrix.shape[1], max(1, statistic.count - 1))
    left, singular, right_h = _truncated_svd(matrix, identifiable_rank)
    rank, scores, residuals, noise = _rank_mdl(statistic, matrix, singular)
    total = float(mean.double().square().sum().item())
    if rank <= 0:
        record = MDLTangentRecord(
            name=candidate.name,
            shape=candidate.shape,
            role="head" if candidate.is_head else "backbone",
            rank=0,
            core_mode="skipped",
            core_parameters=0,
            basis_values=0,
            total_gradient_energy=total,
            noise_energy=noise,
            captured_mean_energy=0.0,
            mdl_zero=scores[0],
            mdl_selected=scores[0],
            diagonal_bic=None,
            dense_bic=None,
        )
        return record, None, None, residuals[0]

    left = left[:, :rank].contiguous()
    right = right_h[:rank].transpose(0, 1).contiguous()
    dense, diagonal_bic, dense_bic = _core_bic(statistic, left, right)
    core_parameters = rank * rank if dense else rank
    captured = float(singular[:rank].double().square().sum().item())
    record = MDLTangentRecord(
        name=candidate.name,
        shape=candidate.shape,
        role="head" if candidate.is_head else "backbone",
        rank=rank,
        core_mode="dense" if dense else "diagonal",
        core_parameters=core_parameters,
        basis_values=int(left.numel() + right.numel()),
        total_gradient_energy=total,
        noise_energy=noise,
        captured_mean_energy=captured,
        mdl_zero=scores[0],
        mdl_selected=scores[rank],
        diagonal_bic=diagonal_bic,
        dense_bic=dense_bic,
    )
    return record, left, right, residuals[rank]


def _head_policy(
    head_candidates: list[_Candidate],
    statistics: dict[str, _Statistic],
    analyses: dict[str, tuple[MDLTangentRecord, Tensor | None, Tensor | None, float]],
) -> tuple[str, dict[str, float]]:
    if not head_candidates:
        return "absent", {}
    total_values = sum(candidate.parameter.numel() for candidate in head_candidates)
    batches = max(statistics[candidate.name].count for candidate in head_candidates)
    observations = batches * total_values
    frozen_rss = 0.0
    full_rss = 0.0
    tangent_rss = 0.0
    tangent_parameters = 0
    for candidate in head_candidates:
        statistic = statistics[candidate.name]
        mean_energy = float(statistic.mean().double().square().sum().item())
        noise = statistic.noise_sse()
        frozen_rss += noise + statistic.count * mean_energy
        full_rss += noise
        if candidate.parameter.ndim >= 2:
            record, _, _, selected_rss = analyses[candidate.name]
            tangent_rss += selected_rss
            tangent_parameters += record.core_parameters
        else:
            # Bias vectors are inexpensive and are trained in the tangent-head
            # candidate; their evidence fit is exact up to observed noise.
            tangent_rss += noise
            tangent_parameters += candidate.parameter.numel()

    scores = {
        "frozen": _safe_information_score(frozen_rss, observations, 0),
        "tangent": _safe_information_score(tangent_rss, observations, tangent_parameters),
        "full": _safe_information_score(full_rss, observations, total_values),
    }
    policy = min(scores, key=lambda key: (scores[key], {"frozen": 0, "tangent": 1, "full": 2}[key]))
    return policy, scores


def _attach(
    candidate: _Candidate,
    record: MDLTangentRecord,
    left: Tensor,
    right: Tensor,
) -> None:
    transformation = MDLTangentCoreParametrization(
        left.to(device=candidate.parameter.device, dtype=candidate.parameter.dtype),
        right.to(device=candidate.parameter.device, dtype=candidate.parameter.dtype),
        candidate.shape,
        dense_core=record.core_mode == "dense",
    )
    parametrize.register_parametrization(candidate.module, candidate.parameter_name, transformation)


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
    """Calibrate and attach the fully automatic MDL-Evidence proposal.

    The complete loader is consumed.  The API intentionally accepts no method-
    specific rank, budget, layer, threshold, head-policy, core-mode, or batch-
    count controls.
    """
    logits_fn = logits_fn or (lambda output: output)
    is_head = is_head or _default_is_head
    batch_to_device = batch_to_device or _unpack_batch
    candidates = list(_iter_calibration_candidates(model, is_head))
    if not candidates:
        raise RuntimeError("no eligible floating model parameters were found")

    statistics, batches, examples = _collect_statistics(
        model,
        candidates,
        data_loader,
        loss_fn,
        device=device,
        logits_fn=logits_fn,
        batch_to_device=batch_to_device,
    )

    analyses: dict[str, tuple[MDLTangentRecord, Tensor | None, Tensor | None, float]] = {}
    for candidate in candidates:
        if candidate.parameter.ndim >= 2:
            analyses[candidate.name] = _analyze_weight(candidate, statistics[candidate.name])

    head_candidates = [candidate for candidate in candidates if candidate.is_head]
    head_policy, head_scores = _head_policy(head_candidates, statistics, analyses)

    records: list[MDLTangentRecord] = []
    for candidate in candidates:
        if candidate.parameter.ndim < 2:
            continue
        record, left, right, _ = analyses[candidate.name]
        should_attach = record.rank > 0 and (not candidate.is_head or head_policy == "tangent")
        if should_attach and left is not None and right is not None:
            _attach(candidate, record, left, right)
            records.append(record)
        elif not candidate.is_head:
            records.append(record)
        elif head_policy == "tangent":
            records.append(record)

    # Activate exactly the model selected by evidence.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MDLTangentCoreParametrization):
            module.core.requires_grad_(True)

    head_trainable = 0
    if head_policy == "full":
        for name, parameter in model.named_parameters():
            if ".parametrizations." not in name and is_head(name):
                parameter.requires_grad_(True)
                head_trainable += parameter.numel()
    elif head_policy == "tangent":
        for name, parameter in model.named_parameters():
            if ".parametrizations." not in name and is_head(name) and parameter.ndim < 2:
                parameter.requires_grad_(True)
                head_trainable += parameter.numel()

    adapter_parameters = sum(
        module.trainable_parameter_count
        for module in model.modules()
        if isinstance(module, MDLTangentCoreParametrization)
    )
    basis_values = sum(
        module.basis_value_count
        for module in model.modules()
        if isinstance(module, MDLTangentCoreParametrization)
    )

    fallback_reason: str | None = None
    if adapter_parameters + head_trainable == 0:
        # A completely frozen model cannot execute a training run.  The least
        # assumption-heavy non-degenerate fallback is the ordinary task head.
        for name, parameter in model.named_parameters():
            if ".parametrizations." not in name and is_head(name):
                parameter.requires_grad_(True)
                head_trainable += parameter.numel()
        head_policy = "full"
        fallback_reason = "all MDL candidates selected zero trainable coordinates"

    report = MDLTangentReport(
        method="mdl_evidence_adaptive_tangent_core_v6",
        calibration_batches=batches,
        calibration_examples=examples,
        candidate_tensors=len(records),
        selected_tensors=sum(record.rank > 0 for record in records),
        skipped_tensors=sum(record.rank == 0 for record in records),
        adapter_parameters=adapter_parameters,
        head_policy=head_policy,
        head_trainable_parameters=head_trainable,
        frozen_basis_values=basis_values,
        fallback_reason=fallback_reason,
        records=tuple(records),
    )
    payload = report.to_dict()
    payload["head_policy_scores"] = head_scores
    model._mdl_tangent_report = payload  # type: ignore[attr-defined]
    return report


def set_mdl_tangent_trainability(model: nn.Module) -> None:
    report = getattr(model, "_mdl_tangent_report", {})
    head_policy = report.get("head_policy", "full") if isinstance(report, dict) else "full"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MDLTangentCoreParametrization):
            module.core.requires_grad_(True)
    if head_policy == "full":
        for name, parameter in model.named_parameters():
            if ".parametrizations." not in name and _default_is_head(name):
                parameter.requires_grad_(True)
    elif head_policy == "tangent":
        for name, parameter in model.named_parameters():
            if ".parametrizations." not in name and _default_is_head(name) and parameter.ndim < 2:
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
    return sum(
        transformation.trainable_parameter_count
        for _, _, _, transformation in iter_mdl_tangent_parametrizations(model)
    )


def mdl_tangent_basis_value_count(model: nn.Module) -> int:
    return sum(
        transformation.basis_value_count
        for _, _, _, transformation in iter_mdl_tangent_parametrizations(model)
    )


@torch.no_grad()
def merge_mdl_tangent_cores_(model: nn.Module) -> int:
    """Exactly merge all selected updates and remove runtime parametrizations."""
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
