"""Local PDE-WSINDy baseline with a rich mass-conservative weak-form library.

Model: d_t rho = -d_x[ sum_k c_k F_k(rho, d_x rho, x) ] + nu d_xx rho with the
KNOWN diffusion nu = sigma^2/2 (identical to the kernel baseline and to the
noise level used by MVNN rollouts). Flux library:
  - spatial B-spline fluxes rho^q B_j(x), q in {1, 2, 3}
  - nonlinear fluxes rho^2, rho^3, rho*d_x rho, rho^2*d_x rho
No global moments, no M(t), no convolution features, no future truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bsplines import (
    evaluate_bspline_basis,
    open_uniform_knots,
    second_difference_matrix,
)
from .observation import Grid
from .regression import FitResult, Group, fit_smooth_ridge
from .weakform import WeakConfig, assemble_design, weak_windows

NONLINEAR_FEATURES = ("rho2", "rho3", "rho_dxrho", "rho2_dxrho")


def build_local_flux_fields(
    density: np.ndarray, grid: Grid, knots: np.ndarray
) -> dict[str, np.ndarray]:
    rho = np.asarray(density, dtype=np.float64)
    gradient = np.gradient(rho, grid.dx, axis=-1, edge_order=2)
    basis = evaluate_bspline_basis(grid.centers, knots)  # (X, M)
    fields: dict[str, np.ndarray] = {}
    for q in (1, 2, 3):
        power = rho**q
        for j in range(basis.shape[1]):
            fields[f"q{q}_b{j:03d}"] = power * basis[None, :, j]
    fields["rho2"] = rho**2
    fields["rho3"] = rho**3
    fields["rho_dxrho"] = rho * gradient
    fields["rho2_dxrho"] = rho**2 * gradient
    return fields


@dataclass
class LocalModel:
    grid: Grid
    knots: np.ndarray
    coefficients: dict[str, np.ndarray | float]
    nu: float

    def __post_init__(self) -> None:
        self._basis = evaluate_bspline_basis(self.grid.centers, self.knots)
        self._spline_fields = {
            q: self._basis @ np.asarray(self.coefficients[f"q{q}"], dtype=np.float64)
            for q in (1, 2, 3)
        }

    @property
    def diffusion(self) -> float:
        """Known diffusion nu = sigma^2/2 (not estimated)."""

        return float(self.nu)

    def velocity(self, density: np.ndarray) -> np.ndarray:
        """Advective velocity v with flux = rho * v (diffusion handled separately)."""

        rho = np.asarray(density, dtype=np.float64)
        gradient = np.gradient(rho, self.grid.dx, axis=-1, edge_order=2)
        velocity = (
            self._spline_fields[1]
            + rho * self._spline_fields[2]
            + rho**2 * self._spline_fields[3]
        )
        velocity = velocity + float(self.coefficients["rho2"]) * rho
        velocity = velocity + float(self.coefficients["rho3"]) * rho**2
        velocity = velocity + float(self.coefficients["rho_dxrho"]) * gradient
        velocity = velocity + float(self.coefficients["rho2_dxrho"]) * rho * gradient
        return velocity

    def drift_field(self, density: np.ndarray) -> np.ndarray:
        """Effective drift (advective velocity); diffusion excluded, as in the FP form."""

        return self.velocity(density)


def save_local_model(path, model: LocalModel) -> None:
    np.savez_compressed(
        path,
        grid_lower=model.grid.lower,
        grid_upper=model.grid.upper,
        grid_cells=model.grid.cells,
        knots=model.knots,
        q1=np.asarray(model.coefficients["q1"], dtype=np.float64),
        q2=np.asarray(model.coefficients["q2"], dtype=np.float64),
        q3=np.asarray(model.coefficients["q3"], dtype=np.float64),
        rho2=float(model.coefficients["rho2"]),
        rho3=float(model.coefficients["rho3"]),
        rho_dxrho=float(model.coefficients["rho_dxrho"]),
        rho2_dxrho=float(model.coefficients["rho2_dxrho"]),
        nu=float(model.nu),
    )


def load_local_model(path) -> LocalModel:
    payload = np.load(path)
    grid = Grid(
        float(payload["grid_lower"]),
        float(payload["grid_upper"]),
        int(payload["grid_cells"]),
    )
    coefficients: dict[str, np.ndarray | float] = {
        "q1": payload["q1"],
        "q2": payload["q2"],
        "q3": payload["q3"],
    }
    for name in NONLINEAR_FEATURES:
        coefficients[name] = float(payload[name])
    return LocalModel(
        grid=grid,
        knots=payload["knots"],
        coefficients=coefficients,
        nu=float(payload["nu"]),
    )


def fit_local_wsindy(
    densities: list[np.ndarray],
    times: np.ndarray,
    grid: Grid,
    basis_count: int,
    nu: float,
    weak_config: WeakConfig,
    smoothness: float,
    threshold: float,
    ridge: float = 1.0e-8,
) -> tuple[LocalModel, FitResult]:
    """One shared local model fitted across all training density trajectories.

    The known diffusion nu enters the weak-form target exactly as in the
    kernel baseline; it is never estimated or thresholded.
    """

    knots = open_uniform_knots(grid.lower, grid.upper, basis_count)
    rows = []
    targets = []
    names: list[str] | None = None
    for index, density in enumerate(densities):
        windows = weak_windows(times, grid.centers, weak_config, index)
        fields = build_local_flux_fields(density, grid, knots)
        if names is None:
            names = list(fields.keys())
        matrix, target = assemble_design(
            density, times, grid.centers, fields, windows, nu=nu
        )
        rows.append(matrix)
        targets.append(target)
    assert names is not None
    matrix = np.vstack(rows)
    target = np.concatenate(targets)

    roughness = second_difference_matrix(basis_count)
    groups: list[Group] = []
    for q in (1, 2, 3):
        indices = tuple(
            names.index(f"q{q}_b{j:03d}") for j in range(basis_count)
        )
        groups.append(Group(f"q{q}", indices, roughness))
    for feature in NONLINEAR_FEATURES:
        groups.append(Group(feature, (names.index(feature),), None))

    fit = fit_smooth_ridge(
        matrix,
        target,
        groups,
        smoothness=smoothness,
        ridge=ridge,
        threshold=threshold,
    )
    coefficients: dict[str, np.ndarray | float] = {}
    for q in (1, 2, 3):
        idx = [names.index(f"q{q}_b{j:03d}") for j in range(basis_count)]
        coefficients[f"q{q}"] = fit.coefficients[idx]
    for feature in NONLINEAR_FEATURES:
        coefficients[feature] = float(fit.coefficients[names.index(feature)])
    model = LocalModel(grid=grid, knots=knots, coefficients=coefficients, nu=nu)
    return model, fit
