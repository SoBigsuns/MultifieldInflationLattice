#!/usr/bin/env python3
"""Validate a lattice run with independently integrated background ODEs.

This script intentionally does not import the Julia implementation and does not
require SciPy.  It integrates the physical-time Klein--Gordon/Friedmann system
with an adaptive Dormand--Prince 5(4) method implemented below, compares the
solution with the run CSV files, and writes CSV/JSON/PNG validation artifacts.

Two reference solutions are produced:

1. ``homogeneous``: a self-consistent homogeneous two-field solution.  It omits
   fluctuation and gradient energy, so it is a useful approximation rather than
   an exact reproduction of a lattice run with vacuum fluctuations.
2. ``csv_driven``: the mean-field Klein--Gordon equation driven by the Hubble
   rate recorded in ``background.csv``.  For a quadratic potential and periodic
   boundaries this is the sharper independent check of the lattice mean fields.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass
class IntegrationStats:
    accepted_steps: int = 0
    rejected_steps: int = 0
    function_evaluations: int = 0


def _dopri54_step(
    fun: Callable[[float, Array], Array], t: float, y: Array, h: float
) -> tuple[Array, Array, int]:
    """One Dormand--Prince 5(4) step; returns fifth-order state and error."""

    k1 = fun(t, y)
    k2 = fun(t + h * (1 / 5), y + h * ((1 / 5) * k1))
    k3 = fun(t + h * (3 / 10), y + h * ((3 / 40) * k1 + (9 / 40) * k2))
    k4 = fun(
        t + h * (4 / 5),
        y + h * ((44 / 45) * k1 - (56 / 15) * k2 + (32 / 9) * k3),
    )
    k5 = fun(
        t + h * (8 / 9),
        y
        + h
        * (
            (19372 / 6561) * k1
            - (25360 / 2187) * k2
            + (64448 / 6561) * k3
            - (212 / 729) * k4
        ),
    )
    k6 = fun(
        t + h,
        y
        + h
        * (
            (9017 / 3168) * k1
            - (355 / 33) * k2
            + (46732 / 5247) * k3
            + (49 / 176) * k4
            - (5103 / 18656) * k5
        ),
    )
    y5 = y + h * (
        (35 / 384) * k1
        + (500 / 1113) * k3
        + (125 / 192) * k4
        - (2187 / 6784) * k5
        + (11 / 84) * k6
    )
    k7 = fun(t + h, y5)
    y4 = y + h * (
        (5179 / 57600) * k1
        + (7571 / 16695) * k3
        + (393 / 640) * k4
        - (92097 / 339200) * k5
        + (187 / 2100) * k6
        + (1 / 40) * k7
    )
    return y5, y5 - y4, 7


def integrate_at_targets(
    fun: Callable[[float, Array], Array],
    y0: Sequence[float],
    targets: Sequence[float],
    rtol: float,
    atol: Sequence[float] | float,
    max_steps: int = 2_000_000,
) -> tuple[Array, IntegrationStats]:
    """Integrate an autonomous/non-autonomous ODE exactly to each target time."""

    times = np.asarray(targets, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("target times must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("target times must be finite and non-decreasing")

    y = np.asarray(y0, dtype=float).copy()
    atol_array = np.broadcast_to(np.asarray(atol, dtype=float), y.shape)
    if np.any(atol_array <= 0) or rtol <= 0:
        raise ValueError("rtol and atol must be positive")

    result = np.empty((len(times), len(y)), dtype=float)
    result[0] = y
    t = float(times[0])
    span = max(float(times[-1] - times[0]), np.finfo(float).eps)
    positive_intervals = np.diff(times)
    positive_intervals = positive_intervals[positive_intervals > 0]
    h = min(span / 1000, float(np.median(positive_intervals))) if len(positive_intervals) else span
    h = max(h, 100 * np.finfo(float).eps * max(1.0, abs(t)))
    stats = IntegrationStats()

    for target_index in range(1, len(times)):
        target = float(times[target_index])
        while t < target:
            if stats.accepted_steps + stats.rejected_steps >= max_steps:
                raise RuntimeError(f"adaptive integrator exceeded {max_steps:,} steps")
            h = min(h, target - t)
            if t + h == t:
                raise RuntimeError(f"step size underflow at t={t:.17g}")

            candidate, error, evaluations = _dopri54_step(fun, t, y, h)
            stats.function_evaluations += evaluations
            scale = atol_array + rtol * np.maximum(np.abs(y), np.abs(candidate))
            error_norm = float(np.max(np.abs(error) / scale))
            if not np.isfinite(error_norm) or not np.all(np.isfinite(candidate)):
                error_norm = math.inf

            if error_norm <= 1.0:
                t += h
                if abs(target - t) <= 8 * np.finfo(float).eps * max(1.0, abs(target)):
                    t = target
                y = candidate
                stats.accepted_steps += 1
                factor = 5.0 if error_norm == 0 else min(5.0, 0.9 * error_norm ** (-0.2))
                h *= max(0.2, factor)
            else:
                stats.rejected_steps += 1
                h *= max(0.1, 0.9 * error_norm ** (-0.2))

        result[target_index] = y

    return result, stats


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Project.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    script_parent = Path(__file__).resolve().parent
    if (script_parent / "Project.toml").is_file() and (script_parent / "runs").is_dir():
        return script_parent
    raise FileNotFoundError("Project.toml と runs ディレクトリを含むプロジェクトが見つかりません")


def find_latest_run(project_root: Path) -> Path:
    candidates = [p.parent for p in (project_root / "runs").glob("*/background.csv")]
    if not candidates:
        raise FileNotFoundError(f"実行結果が見つかりません: {project_root / 'runs'}")
    return max(candidates, key=lambda p: (p / "background.csv").stat().st_mtime)


def _relative_error(actual: Array, expected: Array, floor: float | None = None) -> Array:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if floor is None:
        finite_scale = np.max(np.abs(expected[np.isfinite(expected)])) if np.any(np.isfinite(expected)) else 1.0
        floor = max(np.finfo(float).tiny, finite_scale * 1.0e-14)
    return np.abs(actual - expected) / np.maximum(np.abs(expected), floor)


def _normalized_rms(error: Array, reference: Array) -> float:
    error = np.asarray(error, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = max(float(np.max(np.abs(reference))), float(np.ptp(reference)), np.finfo(float).tiny)
    return float(np.sqrt(np.mean(error**2)) / scale)


def _max_finite(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else math.nan


def _metric_row(
    check: str,
    value: float,
    tolerance: float | None,
    kind: str,
    note: str,
) -> dict[str, object]:
    passed: bool | None = None if tolerance is None else bool(np.isfinite(value) and value <= tolerance)
    return {
        "check": check,
        "kind": kind,
        "value": value,
        "tolerance": tolerance,
        "passed": passed,
        "note": note,
    }


def validate_run(
    run_dir: Path,
    output_dir: Path,
    rtol: float,
    field_atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = ["background.csv", "energies.csv", "config.effective.toml"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run に必要なファイルがありません: {', '.join(missing)}")

    with (run_dir / "config.effective.toml").open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    if model.get("potential") != "quadratic":
        raise ValueError("この検証コードは potential = 'quadratic' の実行結果を対象にしています")

    names = list(model["field_names"])
    masses = np.asarray(model["masses"], dtype=float)
    if len(names) != len(masses) or len(names) == 0:
        raise ValueError("field_names と masses の要素数が一致していません")

    background = pd.read_csv(run_dir / "background.csv").sort_values("output_id").reset_index(drop=True)
    energies = pd.read_csv(run_dir / "energies.csv").sort_values("output_id").reset_index(drop=True)
    diagnostics_path = run_dir / "diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.is_file() else None

    field_columns = [f"{name}_mean" for name in names]
    velocity_columns = [f"{name}_velocity" for name in names]
    needed_background = {
        "output_id", "t", "efolds", "H", "Hdot", "rho", "p", "w", *field_columns, *velocity_columns
    }
    absent = sorted(needed_background.difference(background.columns))
    if absent:
        raise ValueError(f"background.csv に必要な列がありません: {', '.join(absent)}")

    merged = background.merge(
        energies[
            [
                "output_id",
                "kinetic_total",
                "gradient_total",
                "potential_total",
                "rho_total",
            ]
        ],
        on="output_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(background):
        raise ValueError("background.csv と energies.csv の output_id が完全には一致しません")

    times = merged["t"].to_numpy(float)
    phi0 = merged.loc[0, field_columns].to_numpy(float)
    velocity0 = merged.loc[0, velocity_columns].to_numpy(float)
    efold0 = float(merged.loc[0, "efolds"])
    mass2 = masses**2
    n_fields = len(names)

    def homogeneous_rhs(_t: float, state: Array) -> Array:
        phi = state[:n_fields]
        velocity = state[n_fields : 2 * n_fields]
        rho_homogeneous = 0.5 * float(np.dot(velocity, velocity)) + 0.5 * float(np.dot(mass2, phi**2))
        hubble = math.sqrt(max(rho_homogeneous, 0.0) / 3.0)
        return np.concatenate((velocity, -3.0 * hubble * velocity - mass2 * phi, [hubble]))

    lattice_hubble = merged["H"].to_numpy(float)

    def csv_driven_rhs(t: float, state: Array) -> Array:
        phi = state[:n_fields]
        velocity = state[n_fields:]
        hubble = float(np.interp(t, times, lattice_hubble))
        return np.concatenate((velocity, -3.0 * hubble * velocity - mass2 * phi))

    velocity_atol = max(field_atol * float(np.max(masses)), np.finfo(float).tiny)
    homogeneous_atol = np.concatenate(
        (np.full(n_fields, field_atol), np.full(n_fields, velocity_atol), [field_atol])
    )
    driven_atol = np.concatenate((np.full(n_fields, field_atol), np.full(n_fields, velocity_atol)))

    homogeneous, homogeneous_stats = integrate_at_targets(
        homogeneous_rhs,
        np.concatenate((phi0, velocity0, [efold0])),
        times,
        rtol,
        homogeneous_atol,
    )
    csv_driven, driven_stats = integrate_at_targets(
        csv_driven_rhs,
        np.concatenate((phi0, velocity0)),
        times,
        rtol,
        driven_atol,
    )

    hom_phi = homogeneous[:, :n_fields]
    hom_velocity = homogeneous[:, n_fields : 2 * n_fields]
    hom_efolds = homogeneous[:, -1]
    hom_rho = 0.5 * np.sum(hom_velocity**2, axis=1) + 0.5 * np.sum(mass2 * hom_phi**2, axis=1)
    hom_hubble = np.sqrt(np.maximum(hom_rho, 0.0) / 3.0)
    driven_phi = csv_driven[:, :n_fields]
    driven_velocity = csv_driven[:, n_fields:]

    result = pd.DataFrame(
        {
            "output_id": merged["output_id"],
            "t": times,
            "efolds_lattice": merged["efolds"],
            "efolds_homogeneous": hom_efolds,
            "efolds_homogeneous_abs_error": np.abs(hom_efolds - merged["efolds"].to_numpy(float)),
            "H_lattice": lattice_hubble,
            "H_homogeneous": hom_hubble,
            "H_homogeneous_relative_error": _relative_error(hom_hubble, lattice_hubble),
            "rho_lattice": merged["rho"],
            "rho_from_energies": merged["rho_total"],
        }
    )
    for index, name in enumerate(names):
        lattice_phi = merged[field_columns[index]].to_numpy(float)
        lattice_velocity = merged[velocity_columns[index]].to_numpy(float)
        result[f"{name}_mean_lattice"] = lattice_phi
        result[f"{name}_mean_homogeneous"] = hom_phi[:, index]
        result[f"{name}_mean_csv_driven"] = driven_phi[:, index]
        result[f"{name}_mean_homogeneous_abs_error"] = np.abs(hom_phi[:, index] - lattice_phi)
        result[f"{name}_mean_csv_driven_abs_error"] = np.abs(driven_phi[:, index] - lattice_phi)
        result[f"{name}_velocity_lattice"] = lattice_velocity
        result[f"{name}_velocity_homogeneous"] = hom_velocity[:, index]
        result[f"{name}_velocity_csv_driven"] = driven_velocity[:, index]
        result[f"{name}_velocity_homogeneous_abs_error"] = np.abs(hom_velocity[:, index] - lattice_velocity)
        result[f"{name}_velocity_csv_driven_abs_error"] = np.abs(driven_velocity[:, index] - lattice_velocity)

    rho_expected = merged["kinetic_total"] + merged["gradient_total"] + merged["potential_total"]
    pressure_expected = merged["kinetic_total"] - merged["gradient_total"] / 3.0 - merged["potential_total"]
    hdot_expected = -merged["kinetic_total"] - merged["gradient_total"] / 3.0
    w_expected = pressure_expected / rho_expected
    friedmann_recomputed = np.abs(lattice_hubble**2 - merged["rho"].to_numpy(float) / 3.0) / np.maximum(
        np.abs(merged["rho"].to_numpy(float) / 3.0), np.finfo(float).tiny
    )
    result["rho_energy_identity_relative_error"] = _relative_error(
        merged["rho_total"].to_numpy(float), rho_expected.to_numpy(float)
    )
    result["rho_background_relative_error"] = _relative_error(
        merged["rho"].to_numpy(float), merged["rho_total"].to_numpy(float)
    )
    result["pressure_identity_relative_error"] = _relative_error(
        merged["p"].to_numpy(float), pressure_expected.to_numpy(float)
    )
    result["Hdot_identity_relative_error"] = _relative_error(
        merged["Hdot"].to_numpy(float), hdot_expected.to_numpy(float)
    )
    result["w_identity_relative_error"] = _relative_error(
        merged["w"].to_numpy(float), w_expected.to_numpy(float)
    )
    result["friedmann_residual_recomputed"] = friedmann_recomputed

    diagnostic_comparison = math.nan
    if diagnostics is not None and "friedmann_residual" in diagnostics.columns:
        diagnostic_values = diagnostics[["output_id", "friedmann_residual"]]
        aligned = merged[["output_id"]].merge(diagnostic_values, on="output_id", how="left")
        if aligned["friedmann_residual"].notna().all():
            diagnostic_comparison = _max_finite(
                np.abs(aligned["friedmann_residual"].to_numpy(float) - friedmann_recomputed)
            )
            result["friedmann_residual_recorded"] = aligned["friedmann_residual"].to_numpy(float)

    evolution = config.get("evolution", {})
    friedmann_tolerance = float(evolution.get("friedmann_error_tolerance", 1.0e-3))
    identity_tolerance = 5.0e-12
    field_errors = []
    velocity_errors = []
    hom_field_errors = []
    hom_velocity_errors = []
    for index, name in enumerate(names):
        lattice_phi = merged[field_columns[index]].to_numpy(float)
        lattice_velocity = merged[velocity_columns[index]].to_numpy(float)
        field_errors.append(_normalized_rms(driven_phi[:, index] - lattice_phi, lattice_phi))
        velocity_errors.append(_normalized_rms(driven_velocity[:, index] - lattice_velocity, lattice_velocity))
        hom_field_errors.append(_normalized_rms(hom_phi[:, index] - lattice_phi, lattice_phi))
        hom_velocity_errors.append(_normalized_rms(hom_velocity[:, index] - lattice_velocity, lattice_velocity))

    metrics = [
        _metric_row(
            "rho = kinetic + gradient + potential",
            _max_finite(result["rho_energy_identity_relative_error"]),
            identity_tolerance,
            "hard",
            "energies.csv 内のエネルギー恒等式",
        ),
        _metric_row(
            "background rho = energies rho_total",
            _max_finite(result["rho_background_relative_error"]),
            identity_tolerance,
            "hard",
            "background.csv と energies.csv の整合性",
        ),
        _metric_row(
            "p = K - G/3 - V",
            _max_finite(result["pressure_identity_relative_error"]),
            identity_tolerance,
            "hard",
            "圧力の定義",
        ),
        _metric_row(
            "Hdot = -K - G/3",
            _max_finite(result["Hdot_identity_relative_error"]),
            identity_tolerance,
            "hard",
            "Friedmann 第2式（換算プランク質量=1）",
        ),
        _metric_row(
            "w = p/rho",
            _max_finite(result["w_identity_relative_error"]),
            identity_tolerance,
            "hard",
            "状態方程式パラメータの定義",
        ),
        _metric_row(
            "H^2 = rho/3",
            _max_finite(friedmann_recomputed),
            friedmann_tolerance,
            "hard",
            "config.effective.toml の friedmann_error_tolerance を使用",
        ),
        _metric_row(
            "csv-driven field mean ODE",
            max(field_errors),
            5.0e-4,
            "hard",
            "各平均場の正規化RMS誤差の最大値。二次ポテンシャル＋周期境界で成立",
        ),
        _metric_row(
            "csv-driven field velocity ODE",
            max(velocity_errors),
            5.0e-3,
            "hard",
            "各平均速度の正規化RMS誤差の最大値",
        ),
        _metric_row(
            "homogeneous field mean comparison",
            max(hom_field_errors),
            None,
            "diagnostic",
            "ゆらぎ・勾配エネルギーを除いた近似なので合否判定には使わない",
        ),
        _metric_row(
            "homogeneous field velocity comparison",
            max(hom_velocity_errors),
            None,
            "diagnostic",
            "ゆらぎ・勾配エネルギーを除いた近似なので合否判定には使わない",
        ),
        _metric_row(
            "homogeneous H comparison",
            _normalized_rms(hom_hubble - lattice_hubble, lattice_hubble),
            None,
            "diagnostic",
            "格子の全エネルギーと平均場だけのエネルギーの差を含む",
        ),
        _metric_row(
            "recomputed vs recorded Friedmann residual",
            diagnostic_comparison,
            None,
            "diagnostic",
            "diagnostics.csv がある場合の最大絶対差",
        ),
    ]
    summary = pd.DataFrame(metrics)
    hard = summary[summary["kind"] == "hard"]
    overall_passed = bool(hard["passed"].fillna(False).all())

    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "ode_consistency_timeseries.csv"
    summary_path = output_dir / "ode_consistency_summary.csv"
    figure_path = output_dir / "ode_consistency.png"
    metadata_path = output_dir / "ode_consistency_metadata.json"
    result.to_csv(timeseries_path, index=False)
    summary.to_csv(summary_path, index=False)

    x = merged["efolds"].to_numpy(float)
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_fields, 2)))
    for index, name in enumerate(names):
        color = colors[index]
        axes[0, 0].plot(x, merged[field_columns[index]], color=color, lw=2, label=f"{name}: lattice")
        axes[0, 0].plot(x, hom_phi[:, index], color=color, ls="--", alpha=0.85, label=f"{name}: homogeneous")
        axes[0, 0].plot(x, driven_phi[:, index], color=color, ls=":", lw=2, label=f"{name}: CSV-driven")
        axes[0, 1].plot(x, np.abs(driven_phi[:, index] - merged[field_columns[index]].to_numpy(float)), color=color, label=name)
        axes[1, 0].plot(x, np.abs(driven_velocity[:, index] - merged[velocity_columns[index]].to_numpy(float)), color=color, label=name)

    axes[0, 0].set(title="Mean fields: lattice and independent ODEs", xlabel="e-folds N", ylabel="field")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].set(title="CSV-driven mean-field absolute error", xlabel="e-folds N", ylabel="absolute error", yscale="log")
    axes[0, 1].legend()
    axes[1, 0].set(title="CSV-driven velocity absolute error", xlabel="e-folds N", ylabel="absolute error", yscale="log")
    axes[1, 0].legend()
    axes[1, 1].plot(x, lattice_hubble, label="lattice", lw=2)
    axes[1, 1].plot(x, hom_hubble, label="homogeneous", ls="--")
    axes[1, 1].set(title="Hubble rate", xlabel="e-folds N", ylabel="H", yscale="log")
    axes[1, 1].legend()
    axes[2, 0].plot(x, friedmann_recomputed, label="Friedmann constraint")
    axes[2, 0].plot(x, result["rho_background_relative_error"], label="rho CSV agreement")
    axes[2, 0].plot(x, result["Hdot_identity_relative_error"], label="Hdot identity")
    axes[2, 0].axhline(friedmann_tolerance, color="black", ls="--", alpha=0.6, label="configured tolerance")
    axes[2, 0].set(title="Internal consistency", xlabel="e-folds N", ylabel="relative error", yscale="log")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(x, hom_efolds - x)
    axes[2, 1].set(title="Homogeneous approximation: e-fold difference", xlabel="lattice e-folds N", ylabel="N_homogeneous - N_lattice")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"ODE consistency validation: {run_dir.name}\nOverall hard checks: {'PASS' if overall_passed else 'FAIL'}", fontsize=14)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir.resolve()),
        "overall_hard_checks_passed": overall_passed,
        "equations": {
            "homogeneous": "dphi/dt=v; dv/dt=-3Hv-m^2 phi; dN/dt=H; H^2=(sum(v^2)+sum(m^2 phi^2))/6",
            "csv_driven": "dphi_bar/dt=v_bar; dv_bar/dt=-3 H_csv(t) v_bar-m^2 phi_bar",
        },
        "assumptions": [
            "reduced Planck mass is 1",
            "quadratic potential",
            "csv-driven mean equation uses periodic-boundary cancellation of the mean Laplacian",
            "homogeneous comparison intentionally omits fluctuation and gradient energy",
        ],
        "integrator": {
            "name": "independent adaptive Dormand-Prince 5(4)",
            "rtol": rtol,
            "field_atol": field_atol,
            "velocity_atol": velocity_atol,
            "homogeneous": vars(homogeneous_stats),
            "csv_driven": vars(driven_stats),
        },
        "outputs": {
            "timeseries_csv": str(timeseries_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "figure_png": str(figure_path.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, summary, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, help="Project.toml と runs を含むディレクトリ")
    parser.add_argument("--run", type=Path, help="検証する run ディレクトリ（省略時は最新）")
    parser.add_argument("--output-dir", type=Path, help="出力先（省略時は RUN/ode_validation）")
    parser.add_argument("--rtol", type=float, default=2.0e-10, help="ODE積分の相対許容誤差")
    parser.add_argument("--atol", type=float, default=1.0e-12, help="場のODE積分の絶対許容誤差")
    parser.add_argument("--strict", action="store_true", help="hard check が失敗したとき終了コード2を返す")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve() if args.project_root else find_project_root(Path.cwd().resolve())
    run_dir = args.run.resolve() if args.run else find_latest_run(project_root)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "ode_validation"
    _, summary, metadata = validate_run(run_dir, output_dir, args.rtol, args.atol)

    print(f"Run: {run_dir}")
    print(f"Output: {output_dir}")
    print(summary[["check", "kind", "value", "tolerance", "passed"]].to_string(index=False))
    passed = bool(metadata["overall_hard_checks_passed"])
    print(f"\nOverall hard checks: {'PASS' if passed else 'FAIL'}")
    if args.strict and not passed:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
