"""Evaluation metrics: quantile W1, bump tracking, moments, paired bootstrap."""

from __future__ import annotations

import numpy as np

from .observation import Grid


def quantiles_from_density(
    density: np.ndarray, grid: Grid, quantiles: np.ndarray
) -> np.ndarray:
    """Exact piecewise-constant CDF inversion; negatives are clipped."""

    weights = np.maximum(np.asarray(density, dtype=np.float64), 0.0) * grid.dx
    weights /= max(weights.sum(), np.finfo(float).tiny)
    cdf_edges = np.concatenate(([0.0], np.cumsum(weights)))
    q = np.clip(np.asarray(quantiles, dtype=np.float64), 0.0, 1.0)
    cell = np.searchsorted(cdf_edges, q, side="right") - 1
    cell = np.clip(cell, 0, weights.size - 1)
    local_mass = q - cdf_edges[cell]
    fraction = np.divide(
        local_mass,
        weights[cell],
        out=np.zeros_like(local_mass),
        where=weights[cell] > 0.0,
    )
    return grid.lower + (cell + fraction) * grid.dx


def midpoint_probabilities(quantile_count: int = 4096) -> np.ndarray:
    return (np.arange(quantile_count) + 0.5) / quantile_count


def w1_densities(
    density_a: np.ndarray,
    density_b: np.ndarray,
    grid: Grid,
    quantile_count: int = 4096,
) -> float:
    q = midpoint_probabilities(quantile_count)
    qa = quantiles_from_density(density_a, grid, q)
    qb = quantiles_from_density(density_b, grid, q)
    return float(np.mean(np.abs(qa - qb)))


def w1_samples(a: np.ndarray, b: np.ndarray, quantile_count: int = 4096) -> float:
    q = midpoint_probabilities(quantile_count)
    qa = np.quantile(np.asarray(a).reshape(-1), q, method="linear")
    qb = np.quantile(np.asarray(b).reshape(-1), q, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def w1_trajectory(
    predicted: np.ndarray, reference: np.ndarray, grid: Grid, quantile_count: int = 4096
) -> np.ndarray:
    """Per-frame W1 between two density trajectories (T, cells)."""

    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.shape != reference.shape:
        raise ValueError("density trajectories must have matching shapes")
    return np.asarray(
        [
            w1_densities(predicted[t], reference[t], grid, quantile_count)
            for t in range(predicted.shape[0])
        ]
    )


def density_mean(density: np.ndarray, grid: Grid) -> np.ndarray:
    """First moment M(t) of one density or a trajectory of densities."""

    rho = np.asarray(density, dtype=np.float64)
    mass = rho.sum(axis=-1) * grid.dx
    return (rho @ grid.centers) * grid.dx / np.maximum(mass, 1e-300)


def bump_centroid(
    density: np.ndarray, grid: Grid, center: float, half_width: float
) -> np.ndarray:
    """Centroid of the density restricted to a fixed window around ``center``."""

    rho = np.asarray(density, dtype=np.float64)
    window = (grid.centers >= center - half_width) & (grid.centers <= center + half_width)
    if not np.any(window):
        raise ValueError("bump window contains no grid cells")
    local = np.maximum(rho[..., window], 0.0)
    mass = local.sum(axis=-1)
    return (local @ grid.centers[window]) / np.maximum(mass, 1e-300)


def short_time_velocity(
    centroids: np.ndarray, times: np.ndarray, horizon: float = 0.2
) -> float:
    """Mean slope of the centroid over [0, horizon] via least squares."""

    times = np.asarray(times, dtype=np.float64)
    mask = times <= horizon + 1e-12
    if mask.sum() < 2:
        raise ValueError("need at least two frames within the short-time horizon")
    t = times[mask]
    y = np.asarray(centroids, dtype=np.float64)[mask]
    slope = np.polyfit(t, y, 1)[0]
    return float(slope)


def direction_match(predicted_velocity: float, reference_velocity: float) -> bool:
    return bool(np.sign(predicted_velocity) == np.sign(reference_velocity))


def drift_rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((predicted - truth) ** 2)))


def paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bootstrap_samples: int = 2000,
    seed: int = 20260711,
) -> dict[str, float]:
    """Paired bootstrap over triplets for the difference a - b."""

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("paired bootstrap needs matching 1D value vectors")
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = np.empty(bootstrap_samples, dtype=np.float64)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(a[idx] - b[idx]))
    return {
        "mean_difference": float(np.mean(a - b)),
        "bootstrap_std": float(diffs.std(ddof=1)),
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
    }


def bootstrap_mean(
    values: np.ndarray, bootstrap_samples: int = 2000, seed: int = 20260711
) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("bootstrap needs a 1D vector with at least two entries")
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, v.size, size=v.size)
        means[i] = float(v[idx].mean())
    return {
        "mean": float(v.mean()),
        "bootstrap_std": float(means.std(ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }
