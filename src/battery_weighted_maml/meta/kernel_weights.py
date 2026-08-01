"""RBF mean embeddings and simplex-constrained MMD QP source weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np
import torch
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class KernelWeightResult:
    alpha: torch.Tensor
    kernel_ss: np.ndarray
    kernel_st: np.ndarray
    kernel_tt: float
    sigma: float
    objective: float
    solver_status: str


def _validate_sets(sets: Sequence[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    dimension: int | None = None
    for index, values in enumerate(sets):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"empirical set {index} must be a nonempty 2-D array")
        if dimension is None:
            dimension = array.shape[1]
        if array.shape[1] != dimension:
            raise ValueError("all empirical sets must use the same feature dimension")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"empirical set {index} contains NaN or Inf")
        result.append(array)
    return result


def median_heuristic(sets: Sequence[np.ndarray], epsilon: float = 1e-8) -> float:
    joined = np.concatenate(_validate_sets(sets), axis=0)
    distances = pdist(joined, metric="euclidean")
    valid = distances[np.isfinite(distances) & (distances > epsilon)]
    if valid.size == 0:
        return float(epsilon)
    return max(float(np.median(valid)), epsilon)


def rbf_mean_kernel(left: np.ndarray, right: np.ndarray, sigma: float) -> float:
    distances_sq = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
    values = np.exp(-distances_sq / (2.0 * sigma * sigma))
    return float(values.mean())


def build_task_kernels(
    source_sets: Sequence[np.ndarray], target_set: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, float]:
    sources = _validate_sets(source_sets)
    target = _validate_sets([target_set])[0]
    count = len(sources)
    kernel_ss = np.empty((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(row, count):
            value = rbf_mean_kernel(sources[row], sources[column], sigma)
            kernel_ss[row, column] = value
            kernel_ss[column, row] = value
    kernel_st = np.asarray(
        [rbf_mean_kernel(source, target, sigma) for source in sources], dtype=np.float64
    )
    kernel_tt = rbf_mean_kernel(target, target, sigma)
    return kernel_ss, kernel_st, kernel_tt


def solve_mmd_qp(
    kernel_ss: np.ndarray,
    kernel_st: np.ndarray,
    kernel_tt: float,
    diagonal_jitter: float = 1e-8,
    primary_solver: str = "OSQP",
    fallback_solver: str = "SCS",
    tolerance: float = 1e-5,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, float, str]:
    """Solve the MMD QP. Solver failures raise; weights never silently become uniform."""
    log = logger or logging.getLogger("battery_weighted_maml")
    matrix = np.asarray(kernel_ss, dtype=np.float64)
    vector = np.asarray(kernel_st, dtype=np.float64).reshape(-1)
    if matrix.shape != (len(vector), len(vector)) or len(vector) == 0:
        raise ValueError("invalid source kernel dimensions")
    matrix = 0.5 * (matrix + matrix.T) + diagonal_jitter * np.eye(len(vector))
    alpha_variable = cp.Variable(len(vector))
    objective = cp.Minimize(
        cp.quad_form(alpha_variable, cp.psd_wrap(matrix))
        - 2.0 * vector @ alpha_variable
        + float(kernel_tt)
    )
    problem = cp.Problem(objective, [alpha_variable >= 0, cp.sum(alpha_variable) == 1])
    failures: list[str] = []
    status = "not_solved"
    solved_weights: np.ndarray | None = None
    for solver in dict.fromkeys([primary_solver, fallback_solver]):
        try:
            solver_options: dict[str, float | int | bool] = {}
            if solver.upper() == "OSQP":
                # OSQP's loose default tolerances can report ``optimal`` while
                # violating alpha >= 0 by around 1e-5. Full MAML recomputes this
                # QP thousands of times, so request a high-accuracy solution.
                solver_options = {
                    "eps_abs": 1e-8,
                    "eps_rel": 1e-8,
                    "max_iter": 100_000,
                    "polishing": True,
                }
            elif solver.upper() == "SCS":
                solver_options = {"eps": 1e-7, "max_iters": 100_000}
            problem.solve(
                solver=solver,
                verbose=False,
                warm_start=False,
                **solver_options,
            )
            status = f"{solver}:{problem.status}"
            log.debug("MMD QP solver status: %s", status)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and alpha_variable.value is not None:
                candidate = np.asarray(alpha_variable.value, dtype=np.float64).reshape(-1)
                if not np.all(np.isfinite(candidate)):
                    message = f"{status}:non-finite alpha={candidate}"
                    failures.append(message)
                    log.error("MMD QP solution rejected: %s", message)
                    continue
                if np.any(candidate < -tolerance):
                    message = f"{status}:constraint violation alpha={candidate}"
                    failures.append(message)
                    log.error("MMD QP solution rejected; trying fallback: %s", message)
                    continue
                if not np.isclose(candidate.sum(), 1.0, atol=tolerance):
                    message = f"{status}:invalid alpha sum={candidate.sum()}"
                    failures.append(message)
                    log.error("MMD QP solution rejected; trying fallback: %s", message)
                    continue
                solved_weights = candidate
                break
            failures.append(status)
        except Exception as exc:
            message = f"{solver}:{type(exc).__name__}:{exc}"
            failures.append(message)
            log.error("MMD QP solver failed: %s", message)
    if solved_weights is None:
        raise RuntimeError("all MMD QP solvers failed: " + " | ".join(failures))
    weights = solved_weights
    if np.any(weights < -tolerance):
        raise RuntimeError(f"MMD QP returned materially negative alpha ({status}): {weights}")
    weights[(weights < 0) & (weights >= -tolerance)] = 0.0
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"MMD QP returned invalid alpha sum ({status}): {total}")
    weights /= total
    if not np.isclose(weights.sum(), 1.0, atol=tolerance):
        raise RuntimeError(f"MMD QP alpha does not sum to one ({status}): {weights.sum()}")
    value = float(weights @ matrix @ weights - 2.0 * vector @ weights + kernel_tt)
    return weights, value, status


def compute_target_aware_weights(
    source_points: Sequence[torch.Tensor],
    target_points: torch.Tensor,
    sigma: str | float = "median",
    diagonal_jitter: float = 1e-8,
    primary_solver: str = "OSQP",
    fallback_solver: str = "SCS",
    device: torch.device | str = "cpu",
    logger: logging.Logger | None = None,
) -> KernelWeightResult:
    """Detach empirical points, build kernels, and solve target-aware source weights."""
    source_arrays = [point.detach().to("cpu", torch.float64).numpy() for point in source_points]
    target_array = target_points.detach().to("cpu", torch.float64).numpy()
    selected_sigma = (
        median_heuristic([*source_arrays, target_array]) if sigma == "median" else float(sigma)
    )
    if not np.isfinite(selected_sigma) or selected_sigma <= 0:
        raise ValueError(f"invalid RBF sigma: {selected_sigma}")
    kernel_ss, kernel_st, kernel_tt = build_task_kernels(
        source_arrays, target_array, selected_sigma
    )
    weights, objective, status = solve_mmd_qp(
        kernel_ss,
        kernel_st,
        kernel_tt,
        diagonal_jitter=diagonal_jitter,
        primary_solver=primary_solver,
        fallback_solver=fallback_solver,
        logger=logger,
    )
    alpha = torch.as_tensor(weights, dtype=torch.float32, device=device).detach()
    return KernelWeightResult(
        alpha=alpha,
        kernel_ss=kernel_ss,
        kernel_st=kernel_st,
        kernel_tt=kernel_tt,
        sigma=selected_sigma,
        objective=objective,
        solver_status=status,
    )
