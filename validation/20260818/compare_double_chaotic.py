#!/usr/bin/env python3
"""Compare the 2026-08-18 two-field quadratic run with analytic theory.

The script performs four complementary checks:

* analytic two-field slow-roll trajectory and background estimates;
* an independent adaptive Dormand--Prince background ODE calculation, via
  ``validate_background_ode.py`` shipped beside this file;
* Friedmann/stress-energy identities and energy-budget diagnostics;
* Bunch--Davies initial spectrum and Parseval/FFT-normalisation checks.

The analytic curves are slow-roll approximations, not closed-form exact
solutions of the full inhomogeneous lattice system.  The generated metrics
therefore distinguish hard internal checks from approximation diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_ID = "two_field_quadratic_20260818T205254Z_6c5330a9ab54"


def find_project_root(start: Path) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = (start, *start.parents, script_dir, *script_dir.parents)
    for candidate in candidates:
        if (candidate / "Project.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise FileNotFoundError("Project.toml と runs を含むプロジェクトが見つかりません")


def find_run(project_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    fixed = project_root / "runs" / DEFAULT_RUN_ID
    if (fixed / "background.csv").is_file():
        return fixed
    candidates = [path.parent for path in (project_root / "runs").glob("*/background.csv")]
    if not candidates:
        raise FileNotFoundError("解析対象のrunが見つかりません")
    return max(candidates, key=lambda path: (path / "background.csv").stat().st_mtime)


def load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def normalized_rms(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denominator = math.sqrt(float(np.mean(reference**2)))
    denominator = max(denominator, np.finfo(float).tiny)
    return math.sqrt(float(np.mean((actual - reference) ** 2))) / denominator


def max_relative(actual: np.ndarray, reference: np.ndarray, floor_scale: float = 1.0e-14) -> float:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    floor = max(float(np.max(np.abs(reference))) * floor_scale, np.finfo(float).tiny)
    return float(np.max(np.abs(actual - reference) / np.maximum(np.abs(reference), floor)))


def epsilon_v(phi: np.ndarray, masses: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    numerator = 2.0 * np.sum((masses**4) * phi**2, axis=-1)
    denominator = np.sum((masses**2) * phi**2, axis=-1) ** 2
    return numerator / np.maximum(denominator, np.finfo(float).tiny)


def slow_roll_end_u(phi0: np.ndarray, masses: np.ndarray) -> float:
    ratio = (masses[0] / masses[1]) ** 2

    def fields(u: float) -> np.ndarray:
        return np.array([phi0[0] * u**ratio, phi0[1] * u])

    low, high = 1.0e-12, 1.0
    if float(epsilon_v(fields(high), masses)) >= 1.0:
        raise ValueError("初期点がslow-roll終了条件の外側です")
    for _ in range(200):
        middle = math.sqrt(low * high)
        if float(epsilon_v(fields(middle), masses)) > 1.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def slow_roll_solution(phi0: np.ndarray, masses: np.ndarray, samples: int = 30000) -> pd.DataFrame:
    """Parametric analytic slow-roll trajectory for two separable quadratics."""

    mass_ratio_squared = (masses[0] / masses[1]) ** 2
    u_end = slow_roll_end_u(phi0, masses)
    u = np.geomspace(1.0, u_end, samples)
    phi = phi0[0] * u**mass_ratio_squared
    psi = phi0[1] * u
    efolds = (
        phi0[0] ** 2 * (1.0 - u ** (2.0 * mass_ratio_squared))
        + phi0[1] ** 2 * (1.0 - u**2)
    ) / 4.0
    fields = np.column_stack((phi, psi))
    potential = 0.5 * np.sum(masses**2 * fields**2, axis=1)
    eps_v = epsilon_v(fields, masses)
    return pd.DataFrame(
        {
            "u": u,
            "efolds": efolds,
            "phi": phi,
            "psi": psi,
            "H": np.sqrt(potential / 3.0),
            "epsilon_V": eps_v,
            "w_slowroll": -1.0 + 2.0 * eps_v / 3.0,
        }
    )


def crossing_efolds(efolds: np.ndarray, difference: np.ndarray) -> float:
    changes = np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:]))
    if len(changes) == 0:
        return math.nan
    index = int(changes[0])
    x0, x1 = float(efolds[index]), float(efolds[index + 1])
    y0, y1 = float(difference[index]), float(difference[index + 1])
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def run_independent_ode(run_dir: Path, bundle_dir: Path) -> tuple[pd.DataFrame | None, dict]:
    candidates = [
        Path(__file__).resolve().with_name("validate_background_ode.py"),
        run_dir.parents[1] / "validate_background_ode.py",
    ]
    validator_path = next((path for path in candidates if path.is_file()), None)
    if validator_path is None:
        return None, {"available": False, "reason": "validate_background_ode.py not found"}

    specification = importlib.util.spec_from_file_location("mfil_ode_validator", validator_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"検証モジュールをロードできません: {validator_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    ode_dir = bundle_dir / "ode_validation"
    timeseries, summary, metadata = module.validate_run(
        run_dir, ode_dir, rtol=2.0e-10, field_atol=1.0e-12
    )
    return summary, {
        "available": True,
        "validator": str(validator_path.resolve()),
        "overall_hard_checks_passed": bool(metadata["overall_hard_checks_passed"]),
        "summary_csv": str((ode_dir / "ode_consistency_summary.csv").resolve()),
        "figure": str((ode_dir / "ode_consistency.png").resolve()),
        "rows": len(timeseries),
    }


def spectrum_checks(
    run_dir: Path,
    background: pd.DataFrame,
    metadata: dict,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    spectrum_path = run_dir / "field_spectra.csv"
    if not spectrum_path.is_file():
        raise FileNotFoundError(spectrum_path)
    spectra = pd.read_csv(spectrum_path)
    auto = spectra[spectra["field_i"] == spectra["field_j"]].copy()
    initial_id = int(auto["output_id"].min())
    initial = auto[auto["output_id"] == initial_id].copy()

    initial_section = config["initial"]
    masses = dict(zip(config["model"]["field_names"], config["model"]["masses"]))
    include_mass = bool(initial_section.get("include_effective_mass", False))
    a0 = float(initial["a"].iloc[0])
    expected = []
    for row in initial.itertuples(index=False):
        effective_mass = float(masses[row.field_i]) if include_mass else 0.0
        omega_comoving = math.sqrt(row.k_comoving**2 + (a0 * effective_mass) ** 2)
        expected.append(row.k_comoving**3 / (4.0 * math.pi**2 * a0**2 * omega_comoving))
    initial["bd_expected"] = expected
    initial["power_ratio_to_bd"] = initial["power_dimensionless"] / initial["bd_expected"]
    initial["sigma_fraction"] = np.sqrt(2.0 / initial["n_modes"])
    initial["z_from_bd"] = (initial["power_ratio_to_bd"] - 1.0) / initial["sigma_fraction"]

    # After horizon exit, a sufficiently light field obeys the de Sitter
    # estimate Delta^2_delta_psi(k) ~= [H(k=aH)/(2 pi)]^2.  All sampled modes
    # cross early in this run, so N~=10 is a clean frozen-spectrum comparison.
    light_auto = auto[(auto["field_i"] == "psi") & (auto["field_j"] == "psi")]
    available_outputs = light_auto[["output_id", "efolds"]].drop_duplicates()
    evaluation_row = available_outputs.iloc[
        np.argmin(np.abs(available_outputs["efolds"].to_numpy(float) - 10.0))
    ]
    horizon = light_auto[light_auto["output_id"] == int(evaluation_row["output_id"])].copy()
    n_background = background["efolds"].to_numpy(float)
    a_background = background["a"].to_numpy(float)
    h_background = background["H"].to_numpy(float)
    crossing_n: list[float] = []
    crossing_h: list[float] = []
    for k_value in horizon["k_comoving"].to_numpy(float):
        log_ratio = np.log(k_value) - np.log(a_background) - np.log(h_background)
        indices = np.flatnonzero((log_ratio[:-1] >= 0.0) & (log_ratio[1:] <= 0.0))
        if len(indices) == 0:
            crossing_n.append(math.nan)
            crossing_h.append(math.nan)
            continue
        index = int(indices[0])
        fraction = log_ratio[index] / (log_ratio[index] - log_ratio[index + 1])
        n_star = n_background[index] + fraction * (n_background[index + 1] - n_background[index])
        crossing_n.append(float(n_star))
        crossing_h.append(float(np.interp(n_star, n_background, h_background)))
    horizon["horizon_crossing_efolds"] = crossing_n
    horizon["H_at_horizon_crossing"] = crossing_h
    horizon["massless_de_sitter_expected"] = (
        horizon["H_at_horizon_crossing"] / (2.0 * math.pi)
    ) ** 2
    horizon["power_ratio_to_massless_de_sitter"] = (
        horizon["power_dimensionless"] / horizon["massless_de_sitter_expected"]
    )
    horizon["well_populated"] = horizon["n_modes"] >= 50

    b_scale = float(metadata["initial_diagnostics"]["B"])
    box_length = float(metadata["lattice"]["L"])
    k_code = auto["k_comoving"].to_numpy(float) / b_scale
    auto["variance_contribution"] = (
        auto["power_dimensionless"].to_numpy(float)
        * auto["n_modes"].to_numpy(float)
        * 2.0
        * math.pi**2
        / (box_length**3 * k_code**3)
    )
    reconstructed = (
        auto.groupby(["output_id", "field_i"], as_index=False)["variance_contribution"]
        .sum()
        .rename(columns={"variance_contribution": "variance_from_spectrum"})
    )
    variances = background[["output_id", "efolds", "phi_variance", "psi_variance"]]
    reconstructed = reconstructed.merge(variances, on="output_id", how="left", validate="many_to_one")
    reconstructed["variance_background"] = np.where(
        reconstructed["field_i"] == "phi",
        reconstructed["phi_variance"],
        reconstructed["psi_variance"],
    )
    reconstructed["absolute_difference"] = np.abs(
        reconstructed["variance_from_spectrum"] - reconstructed["variance_background"]
    )

    per_field = {}
    for field_name, group in reconstructed.groupby("field_i"):
        peak = max(float(group["variance_background"].max()), np.finfo(float).tiny)
        per_field[field_name] = {
            "peak_normalized_max_abs_difference": float(group["absolute_difference"].max() / peak),
            "peak_variance": peak,
        }
    bd = {}
    for field_name, group in initial.groupby("field_i"):
        bd[field_name] = {
            "mode_weighted_power_ratio": float(
                np.average(group["power_ratio_to_bd"], weights=group["n_modes"])
            ),
            "median_power_ratio": float(group["power_ratio_to_bd"].median()),
            "fraction_within_3sigma": float(np.mean(np.abs(group["z_from_bd"]) <= 3.0)),
            "maximum_abs_z": float(np.max(np.abs(group["z_from_bd"]))),
        }
    populated = horizon[
        horizon["well_populated"]
        & np.isfinite(horizon["power_ratio_to_massless_de_sitter"])
    ]
    horizon_summary = {
        "field": "psi",
        "evaluation_efolds": float(evaluation_row["efolds"]),
        "well_populated_minimum_modes": 50,
        "well_populated_bins": int(len(populated)),
        "mean_power_ratio": float(populated["power_ratio_to_massless_de_sitter"].mean()),
        "median_power_ratio": float(populated["power_ratio_to_massless_de_sitter"].median()),
        "percentile_16": float(populated["power_ratio_to_massless_de_sitter"].quantile(0.16)),
        "percentile_84": float(populated["power_ratio_to_massless_de_sitter"].quantile(0.84)),
        "minimum_crossing_efolds": float(horizon["horizon_crossing_efolds"].min()),
        "maximum_crossing_efolds": float(horizon["horizon_crossing_efolds"].max()),
    }
    return initial, horizon, reconstructed, {
        "bunch_davies": bd,
        "horizon_exit_light_field": horizon_summary,
        "parseval": per_field,
    }


def metric(
    name: str,
    value: float,
    tolerance: float | None,
    category: str,
    interpretation: str,
) -> dict:
    passed = None if tolerance is None else bool(np.isfinite(value) and value <= tolerance)
    return {
        "metric": name,
        "category": category,
        "value": value,
        "tolerance": tolerance,
        "passed": passed,
        "interpretation": interpretation,
    }


def save_figures(
    figure_dir: Path,
    background: pd.DataFrame,
    energies: pd.DataFrame,
    slowroll: pd.DataFrame,
    analytic: pd.DataFrame,
    timeseries: pd.DataFrame,
    initial_spectrum: pd.DataFrame,
    horizon_spectrum: pd.DataFrame,
    reconstructed: pd.DataFrame,
    masses: np.ndarray,
    transition_numeric: float,
    transition_analytic: float,
    friedmann_tolerance: float,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    n = background["efolds"].to_numpy(float)
    colors = {"phi": "#1769aa", "psi": "#00a6a6"}

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8), constrained_layout=True)
    axes[0, 0].plot(n, background["phi_mean"], color=colors["phi"], lw=2, label=r"lattice $\bar\phi$")
    axes[0, 0].plot(n, background["psi_mean"], color=colors["psi"], lw=2, label=r"lattice $\bar\psi$")
    axes[0, 0].plot(analytic["efolds"], analytic["phi"], color=colors["phi"], ls="--", label=r"slow-roll $\phi$")
    axes[0, 0].plot(analytic["efolds"], analytic["psi"], color=colors["psi"], ls="--", label=r"slow-roll $\psi$")
    axes[0, 0].set(title="Background fields", xlabel="e-folds N", ylabel="field value")
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].plot(background["psi_mean"], background["phi_mean"], lw=2, label="lattice")
    axes[0, 1].plot(analytic["psi"], analytic["phi"], ls="--", lw=2, label="analytic slow-roll")
    axes[0, 1].scatter([background["psi_mean"].iloc[0]], [background["phi_mean"].iloc[0]], s=45, label="initial")
    axes[0, 1].set(title="Field-space trajectory", xlabel=r"$\bar\psi$", ylabel=r"$\bar\phi$")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(n, background["H"], lw=2, label="lattice H")
    axes[1, 0].plot(analytic["efolds"], analytic["H"], ls="--", label="analytic slow-roll H")
    axes[1, 0].plot(n, timeseries["H_slowroll_from_lattice_means"], ls=":", label="sqrt(V(means)/3)")
    axes[1, 0].set(title="Hubble rate", xlabel="e-folds N", ylabel="H", yscale="log")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(n, slowroll["epsilon_H"], label=r"$\epsilon_H$")
    axes[1, 1].plot(n, slowroll["epsilon_V"], ls="--", label=r"$\epsilon_V$")
    axes[1, 1].plot(n, 1.5 * (1.0 + background["w"]), ls=":", label=r"$3(1+w)/2$")
    axes[1, 1].axhline(1.0, color="black", alpha=0.6, lw=1, label="inflation end")
    axes[1, 1].set(title="Slow-roll and equation of state", xlabel="e-folds N", ylabel="dimensionless", yscale="log", ylim=(1.0e-4, 2.0))
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    path_background = figure_dir / "background_vs_analytic.png"
    fig.savefig(path_background, dpi=180)
    plt.close(fig)

    merged = background[["output_id", "efolds", "rho", "p", "H", "Hdot"]].merge(
        energies, on="output_id", suffixes=("", "_energy"), validate="one_to_one"
    )
    v_phi = 0.5 * masses[0] ** 2 * background["phi_mean"].to_numpy(float) ** 2
    v_psi = 0.5 * masses[1] ** 2 * background["psi_mean"].to_numpy(float) ** 2
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8), constrained_layout=True)
    axes[0, 0].plot(n, merged["potential_total"] / merged["rho_total"], label=r"$\Omega_V$")
    axes[0, 0].plot(n, merged["kinetic_total"] / merged["rho_total"], label=r"$\Omega_K$")
    axes[0, 0].plot(n, merged["gradient_total"] / merged["rho_total"], label=r"$\Omega_G$")
    axes[0, 0].set(title="Energy fractions", xlabel="e-folds N", ylabel="fraction", yscale="log")
    axes[0, 0].set_ylim(1.0e-16, 2.0)
    axes[0, 0].legend()

    axes[0, 1].plot(n, v_phi, label=r"$V_\phi(\bar\phi)$", color=colors["phi"])
    axes[0, 1].plot(n, v_psi, label=r"$V_\psi(\bar\psi)$", color=colors["psi"])
    if np.isfinite(transition_numeric):
        axes[0, 1].axvline(transition_numeric, color="black", lw=1, label=f"lattice crossing N={transition_numeric:.2f}")
    if np.isfinite(transition_analytic):
        axes[0, 1].axvline(transition_analytic, color="gray", ls="--", lw=1, label=f"slow-roll N={transition_analytic:.2f}")
    axes[0, 1].set(title="Exchange of field dominance", xlabel="e-folds N", ylabel="mean-field potential", yscale="log")
    axes[0, 1].set_ylim(1.0e-20, 1.0e-7)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(n, timeseries["friedmann_residual_recomputed"], label="recomputed")
    axes[1, 0].axhline(friedmann_tolerance, color="black", ls="--", label="configured tolerance")
    axes[1, 0].set(title="Friedmann constraint", xlabel="e-folds N", ylabel="relative residual", yscale="log")
    axes[1, 0].legend()

    axes[1, 1].plot(n, timeseries["H_parametric_relative_error"], label="H vs parametric slow-roll")
    axes[1, 1].plot(n, timeseries["H_mean_potential_relative_error"], label="H vs sqrt(V(means)/3)")
    axes[1, 1].set(title="Analytic comparison errors", xlabel="e-folds N", ylabel="relative error", yscale="log")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    path_energy = figure_dir / "energy_and_constraints.png"
    fig.savefig(path_energy, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 13.6), constrained_layout=True)
    for field_name, group in initial_spectrum.groupby("field_i"):
        color = colors.get(field_name)
        axes[0, 0].plot(group["k_comoving"], group["power_dimensionless"], marker="o", ms=3, lw=1, color=color, label=f"{field_name}: realization")
        axes[0, 0].plot(group["k_comoving"], group["bd_expected"], ls="--", color=color, label=f"{field_name}: BD expectation")
        axes[0, 1].errorbar(group["k_comoving"], group["power_ratio_to_bd"], yerr=group["sigma_fraction"], fmt="o", ms=3, capsize=2, color=color, label=field_name)
    axes[0, 0].set(title="Initial field spectra", xlabel="comoving k", ylabel=r"$\Delta^2_{\delta\phi}$", xscale="log", yscale="log")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(1.0, color="black", lw=1)
    axes[0, 1].set(title="Single-realization / BD expectation", xlabel="comoving k", ylabel="power ratio", xscale="log")
    axes[0, 1].legend()

    axes[1, 0].plot(
        horizon_spectrum["k_comoving"],
        horizon_spectrum["power_dimensionless"],
        marker="o",
        ms=3,
        lw=1,
        label=rf"lattice $\delta\psi$ at N={horizon_spectrum['efolds'].iloc[0]:.2f}",
    )
    axes[1, 0].plot(
        horizon_spectrum["k_comoving"],
        horizon_spectrum["massless_de_sitter_expected"],
        ls="--",
        lw=2,
        label=r"$(H_*/2\pi)^2$ at $k=aH$",
    )
    axes[1, 0].set(
        title="Frozen light-field spectrum",
        xlabel="comoving k",
        ylabel=r"$\Delta^2_{\delta\psi}$",
        xscale="log",
        yscale="log",
    )
    axes[1, 0].legend(fontsize=8)
    populated = horizon_spectrum["well_populated"]
    sparse = ~horizon_spectrum["well_populated"]
    axes[1, 1].scatter(
        horizon_spectrum.loc[populated, "k_comoving"],
        horizon_spectrum.loc[populated, "power_ratio_to_massless_de_sitter"],
        s=25,
        label=r"$n_{modes}\geq50$",
    )
    axes[1, 1].scatter(
        horizon_spectrum.loc[sparse, "k_comoving"],
        horizon_spectrum.loc[sparse, "power_ratio_to_massless_de_sitter"],
        s=25,
        marker="x",
        label="sparse edge bins",
    )
    axes[1, 1].axhline(1.0, color="black", lw=1)
    axes[1, 1].set(
        title="Light-field power / horizon-exit theory",
        xlabel="comoving k",
        ylabel="power ratio",
        xscale="log",
    )
    axes[1, 1].legend(fontsize=8)

    for field_name, group in reconstructed.groupby("field_i"):
        color = colors.get(field_name)
        axes[2, 0].plot(group["efolds"], group["variance_background"], color=color, lw=2, label=f"{field_name}: background.csv")
        axes[2, 0].plot(group["efolds"], group["variance_from_spectrum"], color=color, ls="--", label=f"{field_name}: spectrum sum")
        peak = max(float(group["variance_background"].max()), np.finfo(float).tiny)
        axes[2, 1].plot(group["efolds"], group["absolute_difference"] / peak, color=color, label=field_name)
    axes[2, 0].set(title="Variance and Parseval reconstruction", xlabel="e-folds N", ylabel="variance", yscale="log")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].set(title="Parseval difference / peak variance", xlabel="e-folds N", ylabel="normalised absolute difference", yscale="log")
    axes[2, 1].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    path_spectrum = figure_dir / "fluctuation_spectrum_checks.png"
    fig.savefig(path_spectrum, dpi=180)
    plt.close(fig)
    return [path_background, path_energy, path_spectrum]


def write_tex_macros(path: Path, values: dict) -> None:
    macros = {
        "RunEfolds": values["run"]["total_efolds"],
        "SlowRollEfolds": values["analytic"]["total_efolds"],
        "EfoldRelativeDifference": values["analytic"]["total_efolds_relative_difference"],
        "EfoldRelativePercent": 100.0 * values["analytic"]["total_efolds_relative_difference"],
        "InitialHLattice": values["run"]["initial_H"],
        "InitialHAnalytic": values["analytic"]["initial_H"],
        "InitialHRelativeDifference": values["analytic"]["initial_H_relative_difference"],
        "InitialHRelativePercent": 100.0 * values["analytic"]["initial_H_relative_difference"],
        "InitialW": values["run"]["initial_w"],
        "FinalW": values["run"]["final_w"],
        "FinalEpsilonH": values["run"]["final_epsilon_H"],
        "TransitionNLattice": values["run"]["dominance_transition_efolds"],
        "TransitionNAnalytic": values["analytic"]["dominance_transition_efolds"],
        "TransitionRelativePercent": 100.0 * values["analytic"]["dominance_transition_relative_difference"],
        "FriedmannMaximum": values["constraints"]["friedmann_maximum"],
        "HSRRms": values["analytic"]["H_parametric_normalized_rms"],
        "HSRRmsPercent": 100.0 * values["analytic"]["H_parametric_normalized_rms"],
        "PhiSRRms": values["analytic"]["phi_normalized_rms"],
        "PhiSRRmsPercent": 100.0 * values["analytic"]["phi_normalized_rms"],
        "PsiSRRms": values["analytic"]["psi_normalized_rms"],
        "PsiSRRmsPercent": 100.0 * values["analytic"]["psi_normalized_rms"],
        "ComparisonNMin": values["analytic"]["comparison_efolds_min"],
        "ComparisonNMax": values["analytic"]["comparison_efolds_max"],
        "GradientFractionMaximum": values["energy"]["gradient_fraction_maximum_slowroll"],
        "InitialPotentialFraction": values["energy"]["initial_potential_fraction"],
        "InitialKineticFraction": values["energy"]["initial_kinetic_fraction"],
        "InitialGradientFraction": values["energy"]["initial_gradient_fraction"],
        "InhomogeneousFractionMaximum": values["energy"]["inhomogeneous_energy_fraction_maximum_slowroll"],
        "BDPhiWeightedRatio": values["spectra"]["bunch_davies"]["phi"]["mode_weighted_power_ratio"],
        "BDPsiWeightedRatio": values["spectra"]["bunch_davies"]["psi"]["mode_weighted_power_ratio"],
        "HorizonPsiMeanRatio": values["spectra"]["horizon_exit_light_field"]["mean_power_ratio"],
        "HorizonPsiMedianRatio": values["spectra"]["horizon_exit_light_field"]["median_power_ratio"],
        "HorizonCrossingNMin": values["spectra"]["horizon_exit_light_field"]["minimum_crossing_efolds"],
        "HorizonCrossingNMax": values["spectra"]["horizon_exit_light_field"]["maximum_crossing_efolds"],
        "ParsevalPhi": values["spectra"]["parseval"]["phi"]["peak_normalized_max_abs_difference"],
        "ParsevalPsi": values["spectra"]["parseval"]["psi"]["peak_normalized_max_abs_difference"],
        "ODEFieldRms": values["ode_metrics"]["csv_driven_field_mean_ode"],
        "ODEVelocityRms": values["ode_metrics"]["csv_driven_field_velocity_ode"],
        "ODEHomogeneousHRms": values["ode_metrics"]["homogeneous_H_comparison"],
    }
    lines = ["% Auto-generated by compare_double_chaotic.py. Do not edit by hand."]
    for name, value in macros.items():
        if isinstance(value, (float, int)) and np.isfinite(value):
            formatted = f"{value:.8g}"
        else:
            formatted = "n/a"
        lines.append(f"\\newcommand{{\\{name}}}{{{formatted}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir: Path, bundle_dir: Path) -> dict:
    data_dir = bundle_dir / "data"
    figure_dir = bundle_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    config = load_toml(run_dir / "config.effective.toml")
    metadata = load_toml(run_dir / "metadata.toml")
    background = pd.read_csv(run_dir / "background.csv").sort_values("output_id").reset_index(drop=True)
    energies = pd.read_csv(run_dir / "energies.csv").sort_values("output_id").reset_index(drop=True)
    slowroll = pd.read_csv(run_dir / "slowroll.csv").sort_values("output_id").reset_index(drop=True)
    diagnostics = pd.read_csv(run_dir / "diagnostics.csv").sort_values("output_id").reset_index(drop=True)
    if not (len(background) == len(energies) == len(slowroll) == len(diagnostics)):
        raise ValueError("主要CSVの行数が一致しません")

    names = list(config["model"]["field_names"])
    masses = np.asarray(config["model"]["masses"], dtype=float)
    if names != ["phi", "psi"] or len(masses) != 2:
        raise ValueError("本コードは phi, psi の二場二次ポテンシャルrunを対象とします")
    phi0 = background.loc[0, ["phi_mean", "psi_mean"]].to_numpy(float)
    analytic = slow_roll_solution(phi0, masses)
    n = background["efolds"].to_numpy(float)
    fields = background[["phi_mean", "psi_mean"]].to_numpy(float)
    velocities = background[["phi_velocity", "psi_velocity"]].to_numpy(float)
    v_mean = 0.5 * np.sum(masses**2 * fields**2, axis=1)
    h_mean_sr = np.sqrt(v_mean / 3.0)

    n_analytic = analytic["efolds"].to_numpy(float)
    phi_param = np.interp(n, n_analytic, analytic["phi"], left=np.nan, right=np.nan)
    psi_param = np.interp(n, n_analytic, analytic["psi"], left=np.nan, right=np.nan)
    h_param = np.interp(n, n_analytic, analytic["H"], left=np.nan, right=np.nan)
    epsv_param = np.interp(n, n_analytic, analytic["epsilon_V"], left=np.nan, right=np.nan)

    merged_energy = background[["output_id", "rho", "p", "H", "Hdot"]].merge(
        energies, on="output_id", validate="one_to_one"
    )
    rho = merged_energy["rho"].to_numpy(float)
    kinetic = merged_energy["kinetic_total"].to_numpy(float)
    gradient = merged_energy["gradient_total"].to_numpy(float)
    potential = merged_energy["potential_total"].to_numpy(float)
    friedmann = np.abs(background["H"].to_numpy(float) ** 2 - rho / 3.0) / np.maximum(
        np.maximum(background["H"].to_numpy(float) ** 2, rho / 3.0), np.finfo(float).tiny
    )
    rho_identity = np.abs(rho - (kinetic + gradient + potential)) / np.maximum(rho, np.finfo(float).tiny)
    pressure_expected = kinetic - gradient / 3.0 - potential
    pressure_identity = np.abs(background["p"].to_numpy(float) - pressure_expected) / np.maximum(
        np.abs(background["p"].to_numpy(float)), np.max(np.abs(background["p"])) * 1.0e-14
    )
    epsilon_from_hdot = -background["Hdot"].to_numpy(float) / background["H"].to_numpy(float) ** 2
    epsilon_stress = 1.5 * (1.0 + background["w"].to_numpy(float))

    gradient_fraction = gradient / rho
    slow_mask = (
        (n >= 1.0)
        & (slowroll["epsilon_H"].to_numpy(float) < 0.1)
        & (gradient_fraction < 1.0e-3)
        & np.isfinite(phi_param)
    )
    if slow_mask.sum() < 10:
        raise RuntimeError("slow-roll比較区間を確保できません")

    v_phi = 0.5 * masses[0] ** 2 * fields[:, 0] ** 2
    v_psi = 0.5 * masses[1] ** 2 * fields[:, 1] ** 2
    transition_numeric = crossing_efolds(n, v_phi - v_psi)
    v_phi_analytic = 0.5 * masses[0] ** 2 * analytic["phi"].to_numpy(float) ** 2
    v_psi_analytic = 0.5 * masses[1] ** 2 * analytic["psi"].to_numpy(float) ** 2
    transition_analytic = crossing_efolds(n_analytic, v_phi_analytic - v_psi_analytic)

    timeseries = pd.DataFrame(
        {
            "output_id": background["output_id"],
            "efolds": n,
            "phi_lattice": fields[:, 0],
            "psi_lattice": fields[:, 1],
            "phi_slowroll_parametric": phi_param,
            "psi_slowroll_parametric": psi_param,
            "H_lattice": background["H"],
            "H_slowroll_parametric": h_param,
            "H_slowroll_from_lattice_means": h_mean_sr,
            "H_parametric_relative_error": np.abs(h_param - background["H"]) / np.maximum(np.abs(background["H"]), np.finfo(float).tiny),
            "H_mean_potential_relative_error": np.abs(h_mean_sr - background["H"]) / np.maximum(np.abs(background["H"]), np.finfo(float).tiny),
            "epsilon_H": slowroll["epsilon_H"],
            "epsilon_V_lattice_means": slowroll["epsilon_V"],
            "epsilon_V_parametric": epsv_param,
            "epsilon_from_Hdot": epsilon_from_hdot,
            "epsilon_from_w_and_friedmann": epsilon_stress,
            "w": background["w"],
            "omega_kinetic": kinetic / rho,
            "omega_gradient": gradient / rho,
            "omega_potential": potential / rho,
            "friedmann_residual_recomputed": friedmann,
            "rho_identity_relative_error": rho_identity,
            "pressure_identity_relative_error": pressure_identity,
            "slowroll_comparison_mask": slow_mask,
        }
    )

    initial_spectrum, horizon_spectrum, reconstructed, spectrum_summary = spectrum_checks(
        run_dir, background, metadata, config
    )
    ode_summary, ode_info = run_independent_ode(run_dir, bundle_dir)
    ode_metrics = {
        "csv_driven_field_mean_ode": math.nan,
        "csv_driven_field_velocity_ode": math.nan,
        "homogeneous_H_comparison": math.nan,
    }
    if ode_summary is not None:
        for key, check_name in [
            ("csv_driven_field_mean_ode", "csv-driven field mean ODE"),
            ("csv_driven_field_velocity_ode", "csv-driven field velocity ODE"),
            ("homogeneous_H_comparison", "homogeneous H comparison"),
        ]:
            row = ode_summary[ode_summary["check"] == check_name]
            if len(row):
                ode_metrics[key] = float(row["value"].iloc[0])

    initial_h_analytic = float(analytic["H"].iloc[0])
    initial_h_lattice = float(background["H"].iloc[0])
    total_n_analytic = float(analytic["efolds"].iloc[-1])
    total_n_lattice = float(n[-1])
    inhomogeneous_energy = rho - (0.5 * np.sum(velocities**2, axis=1) + v_mean)
    friedmann_tolerance = float(config["evolution"]["friedmann_error_tolerance"])

    values = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "id": run_dir.name,
            "directory": str(run_dir.resolve()),
            "total_efolds": total_n_lattice,
            "initial_H": initial_h_lattice,
            "initial_w": float(background["w"].iloc[0]),
            "final_w": float(background["w"].iloc[-1]),
            "final_epsilon_H": float(slowroll["epsilon_H"].iloc[-1]),
            "dominance_transition_efolds": transition_numeric,
            "rows": len(background),
        },
        "model": {
            "potential": config["model"]["potential"],
            "field_names": names,
            "masses": masses.tolist(),
            "initial_means": phi0.tolist(),
            "mass_squared_ratio": float((masses[0] / masses[1]) ** 2),
            "lattice_size": int(config["lattice"]["size"]),
            "box_size": float(config["lattice"]["box_size"]),
            "seed": int(config["initial"]["seed"]),
            "timestep_factor": float(config["evolution"]["timestep_factor"]),
        },
        "analytic": {
            "description": "two-field quadratic slow-roll parametric approximation",
            "total_efolds": total_n_analytic,
            "total_efolds_relative_difference": abs(total_n_lattice - total_n_analytic) / total_n_analytic,
            "initial_H": initial_h_analytic,
            "initial_H_relative_difference": abs(initial_h_lattice - initial_h_analytic) / initial_h_analytic,
            "dominance_transition_efolds": transition_analytic,
            "dominance_transition_relative_difference": abs(transition_numeric - transition_analytic) / transition_analytic,
            "phi_normalized_rms": normalized_rms(fields[slow_mask, 0], phi_param[slow_mask]),
            "psi_normalized_rms": normalized_rms(fields[slow_mask, 1], psi_param[slow_mask]),
            "H_parametric_normalized_rms": normalized_rms(background["H"].to_numpy(float)[slow_mask], h_param[slow_mask]),
            "H_from_means_normalized_rms": normalized_rms(background["H"].to_numpy(float)[slow_mask], h_mean_sr[slow_mask]),
            "comparison_efolds_min": float(n[slow_mask].min()),
            "comparison_efolds_max": float(n[slow_mask].max()),
        },
        "constraints": {
            "friedmann_maximum": float(np.max(friedmann)),
            "friedmann_configured_tolerance": friedmann_tolerance,
            "rho_identity_maximum": float(np.max(rho_identity)),
            "pressure_identity_maximum": float(np.max(pressure_identity)),
            "epsilon_H_vs_Hdot_maximum": max_relative(slowroll["epsilon_H"], epsilon_from_hdot),
            "epsilon_H_vs_w_maximum_absolute": float(np.max(np.abs(slowroll["epsilon_H"] - epsilon_stress))),
        },
        "energy": {
            "initial_potential_fraction": float(potential[0] / rho[0]),
            "initial_kinetic_fraction": float(kinetic[0] / rho[0]),
            "initial_gradient_fraction": float(gradient[0] / rho[0]),
            "gradient_fraction_maximum_slowroll": float(np.max(gradient_fraction[slow_mask])),
            "inhomogeneous_energy_fraction_maximum_slowroll": float(np.max(np.abs(inhomogeneous_energy[slow_mask] / rho[slow_mask]))),
        },
        "spectra": spectrum_summary,
        "ode": ode_info,
        "ode_metrics": ode_metrics,
        "scope": {
            "validated": [
                "background two-field dynamics",
                "Friedmann and stress-energy consistency",
                "slow-roll qualitative and quantitative behaviour",
                "vacuum spectrum initialization",
                "field-spectrum FFT normalization",
            ],
            "not_established": [
                "continuum or infinite-volume limit",
                "time-step and spatial-resolution convergence",
                "ensemble uncertainty over random seeds",
                "independent validation of linear-curvature and delta-N spectra",
            ],
        },
    }

    criteria = [
        metric("Initial H vs slow-roll", values["analytic"]["initial_H_relative_difference"], 0.01, "analytic", "初期Hが平均場ポテンシャル予測と1%以内"),
        metric("Total e-folds vs slow-roll", values["analytic"]["total_efolds_relative_difference"], 0.05, "analytic", "slow-roll終了予測と5%以内"),
        metric("Dominance transition", values["analytic"]["dominance_transition_relative_difference"], 0.10, "analytic", "重い場から軽い場への交代時刻が10%以内"),
        metric("Parametric H normalized RMS", values["analytic"]["H_parametric_normalized_rms"], 0.05, "analytic", "slow-roll比較区間のH"),
        metric("phi normalized RMS", values["analytic"]["phi_normalized_rms"], 0.05, "analytic", "slow-roll比較区間の重い場"),
        metric("psi normalized RMS", values["analytic"]["psi_normalized_rms"], 0.05, "analytic", "slow-roll比較区間の軽い場"),
        metric("Friedmann constraint", values["constraints"]["friedmann_maximum"], friedmann_tolerance, "internal", "run設定の許容値"),
        metric("Energy decomposition", values["constraints"]["rho_identity_maximum"], 5.0e-12, "internal", "rho=K+G+V"),
        metric("Pressure decomposition", values["constraints"]["pressure_identity_maximum"], 5.0e-12, "internal", "p=K-G/3-V"),
        metric("Gradient fraction in slow-roll", values["energy"]["gradient_fraction_maximum_slowroll"], 0.01, "lattice", "背景比較が不均一勾配に支配されていない"),
        metric("BD phi weighted power", abs(values["spectra"]["bunch_davies"]["phi"]["mode_weighted_power_ratio"] - 1.0), 0.10, "spectrum", "単一実現のmode-weighted期待値"),
        metric("BD psi weighted power", abs(values["spectra"]["bunch_davies"]["psi"]["mode_weighted_power_ratio"] - 1.0), 0.10, "spectrum", "単一実現のmode-weighted期待値"),
        metric("Light-field horizon-exit spectrum", abs(values["spectra"]["horizon_exit_light_field"]["mean_power_ratio"] - 1.0), 0.10, "spectrum", "n_modes>=50でP_delta_psi=(H*/2pi)^2"),
        metric("Parseval phi", values["spectra"]["parseval"]["phi"]["peak_normalized_max_abs_difference"], 1.0e-3, "spectrum", "スペクトル和と実空間分散"),
        metric("Parseval psi", values["spectra"]["parseval"]["psi"]["peak_normalized_max_abs_difference"], 1.0e-3, "spectrum", "スペクトル和と実空間分散"),
    ]
    if ode_info.get("available"):
        criteria.extend(
            [
                metric("Independent ODE field means", ode_metrics["csv_driven_field_mean_ode"], 5.0e-4, "ode", "CSV Hで駆動した独立DOPRI5(4)"),
                metric("Independent ODE velocities", ode_metrics["csv_driven_field_velocity_ode"], 5.0e-3, "ode", "CSV Hで駆動した独立DOPRI5(4)"),
            ]
        )
    metrics_frame = pd.DataFrame(criteria)
    values["overall_selected_checks_passed"] = bool(metrics_frame["passed"].fillna(False).all())

    figures = save_figures(
        figure_dir,
        background,
        energies,
        slowroll,
        analytic,
        timeseries,
        initial_spectrum,
        horizon_spectrum,
        reconstructed,
        masses,
        transition_numeric,
        transition_analytic,
        friedmann_tolerance,
    )
    values["figures"] = [str(path.resolve()) for path in figures]

    timeseries.to_csv(data_dir / "analytic_comparison_timeseries.csv", index=False)
    analytic.to_csv(data_dir / "analytic_slowroll_curve.csv", index=False)
    initial_spectrum.to_csv(data_dir / "initial_spectrum_vs_bunch_davies.csv", index=False)
    horizon_spectrum.to_csv(data_dir / "light_field_horizon_exit_spectrum.csv", index=False)
    reconstructed.to_csv(data_dir / "parseval_variance_check.csv", index=False)
    metrics_frame.to_csv(data_dir / "validation_metrics.csv", index=False)
    (data_dir / "validation_summary.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tex_macros(data_dir / "analysis_results.tex", values)
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve() if args.project_root else find_project_root(Path.cwd().resolve())
    run_dir = find_run(project_root, args.run)
    bundle_dir = args.output_dir.resolve()
    values = analyze(run_dir, bundle_dir)
    print(f"Run: {run_dir}")
    print(f"Output: {bundle_dir}")
    print(f"Selected validation checks: {'PASS' if values['overall_selected_checks_passed'] else 'FAIL'}")
    print(f"Lattice N = {values['run']['total_efolds']:.8f}")
    print(f"Slow-roll N = {values['analytic']['total_efolds']:.8f}")
    print(f"max Friedmann residual = {values['constraints']['friedmann_maximum']:.6e}")
    return 0 if values["overall_selected_checks_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
