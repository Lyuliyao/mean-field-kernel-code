"""Smoothness-penalized ridge regression with relative group thresholding.

Adapted from the validated r1c1 group regression semantics: columns are RMS
normalized, the second-difference roughness penalizes physical coefficients,
and whole coefficient groups are dropped when their dimensionless relative
effect (RMS group contribution / RMS target) falls below the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Group:
    name: str
    indices: tuple[int, ...]
    roughness: np.ndarray | None = None


@dataclass
class FitResult:
    coefficients: np.ndarray
    active_groups: tuple[str, ...]
    group_effects: dict[str, float] = field(default_factory=dict)
    residual_rmse: float = 0.0
    relative_residual: float = 0.0
    iterations: int = 0


def _solve(
    gram: np.ndarray,
    right: np.ndarray,
    groups: list[Group],
    active: set[str],
    column_scale: np.ndarray,
    smoothness: float,
    ridge: float,
) -> np.ndarray:
    size = gram.shape[0]
    active_idx = sorted(
        i for g in groups if g.name in active for i in g.indices
    )
    if not active_idx:
        return np.zeros(size, dtype=np.float64)
    system = gram[np.ix_(active_idx, active_idx)].copy()
    system += ridge * np.eye(len(active_idx))
    if smoothness > 0.0:
        penalty = np.zeros_like(system)
        position = {j: p for p, j in enumerate(active_idx)}
        for g in groups:
            if g.name not in active or g.roughness is None:
                continue
            scaled = (g.roughness / np.sqrt(g.roughness.shape[0])) @ np.diag(
                1.0 / column_scale[list(g.indices)]
            )
            block = scaled.T @ scaled
            pos = [position[j] for j in g.indices]
            penalty[np.ix_(pos, pos)] += block
        system += smoothness * penalty
    solution = np.zeros(size, dtype=np.float64)
    solution[active_idx] = np.linalg.solve(system, right[active_idx])
    return solution


def fit_smooth_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: list[Group],
    *,
    smoothness: float = 0.0,
    ridge: float = 1.0e-8,
    threshold: float = 0.0,
    max_iterations: int = 20,
) -> FitResult:
    A = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.size:
        raise ValueError("matrix and target shapes are inconsistent")
    n_rows, n_cols = A.shape
    covered = sorted(i for g in groups for i in g.indices)
    if covered != list(range(n_cols)):
        raise ValueError("groups must exactly partition the columns")

    column_scale = np.sqrt(np.mean(A**2, axis=0))
    column_scale = np.where(column_scale > 1e-300, column_scale, 1.0)
    An = A / column_scale[None, :]
    gram = An.T @ An / n_rows
    right = An.T @ y / n_rows
    response_rms = float(np.sqrt(np.mean(y**2)))

    active = {g.name for g in groups}
    iterations = 0
    while True:
        iterations += 1
        coeffs_n = _solve(gram, right, groups, active, column_scale, smoothness, ridge)
        effects: dict[str, float] = {}
        for g in groups:
            if g.name not in active:
                effects[g.name] = 0.0
                continue
            contribution = An[:, list(g.indices)] @ coeffs_n[list(g.indices)]
            effects[g.name] = float(
                np.sqrt(np.mean(contribution**2)) / max(response_rms, 1e-300)
            )
        dropped = {
            g.name
            for g in groups
            if g.name in active and effects[g.name] < threshold
        }
        if not dropped or iterations >= max_iterations or len(active - dropped) == 0:
            if dropped and len(active - dropped) > 0:
                active -= dropped
                coeffs_n = _solve(
                    gram, right, groups, active, column_scale, smoothness, ridge
                )
            break
        active -= dropped

    coefficients = coeffs_n / column_scale
    residual = y - A @ coefficients
    residual_rmse = float(np.sqrt(np.mean(residual**2)))
    return FitResult(
        coefficients=coefficients,
        active_groups=tuple(sorted(active)),
        group_effects=effects,
        residual_rmse=residual_rmse,
        relative_residual=residual_rmse / max(response_rms, 1e-300),
        iterations=iterations,
    )
