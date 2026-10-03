"""Cubic B-spline bases (open uniform knots) and roughness matrices."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

SPLINE_DEGREE = 3


def open_uniform_knots(lower: float, upper: float, basis_count: int) -> np.ndarray:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("spline domain must be finite and increasing")
    if basis_count < SPLINE_DEGREE + 1:
        raise ValueError("basis_count is too small for cubic splines")
    interior_count = basis_count - SPLINE_DEGREE - 1
    interior = (
        np.linspace(lower, upper, interior_count + 2, dtype=np.float64)[1:-1]
        if interior_count
        else np.asarray([], dtype=np.float64)
    )
    return np.concatenate(
        (
            np.repeat(float(lower), SPLINE_DEGREE + 1),
            interior,
            np.repeat(float(upper), SPLINE_DEGREE + 1),
        )
    )


def evaluate_bspline_basis(points: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """B_j(points) with exactly-zero rows outside the closed knot domain."""

    values = np.asarray(points, dtype=np.float64)
    knot_vector = np.asarray(knots, dtype=np.float64)
    basis_count = knot_vector.size - SPLINE_DEGREE - 1
    if basis_count < SPLINE_DEGREE + 1:
        raise ValueError("invalid knot vector")
    lower = float(knot_vector[SPLINE_DEGREE])
    upper = float(knot_vector[-SPLINE_DEGREE - 1])
    flat = values.reshape(-1)
    result = np.zeros((flat.size, basis_count), dtype=np.float64)
    inside = (flat >= lower) & (flat <= upper)
    if np.any(inside):
        result[inside] = BSpline.design_matrix(
            flat[inside], knot_vector, SPLINE_DEGREE, extrapolate=False
        ).toarray()
    return result.reshape(values.shape + (basis_count,))


def second_difference_matrix(size: int) -> np.ndarray:
    if size < 4:
        raise ValueError("roughness penalty requires at least four coefficients")
    matrix = np.zeros((size - 2, size), dtype=np.float64)
    for row in range(size - 2):
        matrix[row, row : row + 3] = (1.0, -2.0, 1.0)
    return matrix
