#!/usr/bin/env python3
"""Regenerate committed physics fixtures without importing the Julia package."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RTOL = 2.0e-13
ATOL = 2.0e-15


def add_scaled(y, terms):
    return [value + sum(scale * vector[i] for scale, vector in terms)
            for i, value in enumerate(y)]


def dopri54_step(rhs, t, y, h):
    """One Dormand--Prince 5(4) step; coefficients are not used by Julia RK4."""
    k1 = rhs(t, y)
    k2 = rhs(t + h / 5, add_scaled(y, [(h / 5, k1)]))
    k3 = rhs(t + 3 * h / 10, add_scaled(y, [
        (3 * h / 40, k1), (9 * h / 40, k2)]))
    k4 = rhs(t + 4 * h / 5, add_scaled(y, [
        (44 * h / 45, k1), (-56 * h / 15, k2), (32 * h / 9, k3)]))
    k5 = rhs(t + 8 * h / 9, add_scaled(y, [
        (19372 * h / 6561, k1), (-25360 * h / 2187, k2),
        (64448 * h / 6561, k3), (-212 * h / 729, k4)]))
    k6 = rhs(t + h, add_scaled(y, [
        (9017 * h / 3168, k1), (-355 * h / 33, k2),
        (46732 * h / 5247, k3), (49 * h / 176, k4),
        (-5103 * h / 18656, k5)]))
    y5 = add_scaled(y, [
        (35 * h / 384, k1), (500 * h / 1113, k3),
        (125 * h / 192, k4), (-2187 * h / 6784, k5),
        (11 * h / 84, k6)])
    k7 = rhs(t + h, y5)
    y4 = add_scaled(y, [
        (5179 * h / 57600, k1), (7571 * h / 16695, k3),
        (393 * h / 640, k4), (-92097 * h / 339200, k5),
        (187 * h / 2100, k6), (h / 40, k7)])
    return y5, [a - b for a, b in zip(y5, y4)]


def integrate(rhs, y0, targets):
    t = 0.0
    y = list(y0)
    h = 1.0e-3
    snapshots = []
    for target in targets:
        while t < target:
            h = min(h, target - t)
            candidate, error = dopri54_step(rhs, t, y, h)
            scaled_error = max(
                abs(e) / (ATOL + RTOL * max(abs(old), abs(new)))
                for e, old, new in zip(error, y, candidate)
            )
            if scaled_error <= 1.0:
                t += h
                y = candidate
            factor = 5.0 if scaled_error == 0.0 else 0.9 * scaled_error ** (-0.2)
            h *= min(5.0, max(0.2, factor))
            if h < 1.0e-15:
                raise RuntimeError("adaptive reference integrator underflow")
        snapshots.append((target, list(y)))
    return snapshots


def homogeneous_rhs(masses, scale_equation):
    nfields = len(masses)

    def rhs(_tau, y):
        phi = y[:nfields]
        prime = y[nfields:2 * nfields]
        a = y[2 * nfields]
        a_prime = y[2 * nfields + 1]
        potential = 0.5 * sum((mass * value) ** 2
                              for mass, value in zip(masses, phi))
        kinetic = 0.5 * sum(value * value for value in prime) / (a * a)
        rho = kinetic + potential
        pressure = kinetic - potential
        dphi = list(prime)
        dprime = [-2.0 * a_prime * velocity / a - a * a * mass * mass * value
                  for mass, value, velocity in zip(masses, phi, prime)]
        if scale_equation == "inflationeasy":
            # evolution.cpp:509-516 with rescale_s=0 and zero gradients.
            da_prime = a ** 3 * potential - a_prime * a_prime / a
        elif scale_equation == "einstein":
            da_prime = a ** 3 * (rho - 3.0 * pressure) / 6.0
        else:
            raise ValueError(scale_equation)
        return dphi + dprime + [a_prime, da_prime, a]

    return rhs


def initial_state(masses, phi, cosmic_velocity, a0):
    prime = [a0 * value for value in cosmic_velocity]
    potential = 0.5 * sum((mass * value) ** 2
                          for mass, value in zip(masses, phi))
    kinetic = 0.5 * sum(value * value for value in cosmic_velocity)
    h_code = math.sqrt((kinetic + potential) / 3.0)
    return list(phi) + prime + [a0, a0 * a0 * h_code, 0.0]


def classical_rk4_step(rhs, t, y, h):
    """The classical RK4 tableau shown in the cited public Notion page."""
    k1 = rhs(t, y)
    k2 = rhs(t + h / 2.0, add_scaled(y, [(h / 2.0, k1)]))
    k3 = rhs(t + h / 2.0, add_scaled(y, [(h / 2.0, k2)]))
    k4 = rhs(t + h, add_scaled(y, [(h, k3)]))
    return add_scaled(y, [
        (h / 6.0, k1), (h / 3.0, k2),
        (h / 3.0, k3), (h / 6.0, k4),
    ])


def notion_background_rows(step_count=4):
    """Exact background algorithm in Notion blocks 75c6..., 3be8..., 4c60...."""
    mass_phi, mass_psi = 9.0e-6, 1.0e-6

    def diagnostics(y):
        phi, phi_dot, psi, psi_dot, efolds = y
        potential = 0.5 * ((mass_phi * phi) ** 2 + (mass_psi * psi) ** 2)
        H = math.sqrt((0.5 * (phi_dot ** 2 + psi_dot ** 2) + potential) / 3.0)
        epsilon = 0.5 * (phi_dot ** 2 + psi_dot ** 2) / (H * H)
        return potential, H, epsilon, efolds

    def rhs(_time, y):
        phi, phi_dot, psi, psi_dot, _efolds = y
        _potential, H, _epsilon, _ = diagnostics(y)
        return [
            phi_dot,
            -3.0 * H * phi_dot - mass_phi ** 2 * phi,
            psi_dot,
            -3.0 * H * psi_dot - mass_psi ** 2 * psi,
            H,
        ]

    y = [13.0, 0.0, 13.0, 0.0, 0.0]
    time = 0.0
    B = math.sqrt(diagnostics(y)[0])
    rows = []
    for step in range(step_count + 1):
        _potential, H, epsilon, efolds = diagnostics(y)
        dt = 0.01 / H
        rows.append({
            "step": step,
            "time_physical": time,
            "time_code": B * time,
            "efolds": efolds,
            "a": math.exp(efolds),
            "H_physical": H,
            "epsilon": epsilon,
            "phi": y[0],
            "phi_dot_physical": y[1],
            "psi": y[2],
            "psi_dot_physical": y[3],
            "dt_next_physical": dt,
            "B": B,
            "mass_phi_physical": mass_phi,
            "mass_psi_physical": mass_psi,
        })
        if step < step_count:
            y = classical_rk4_step(rhs, time, y, dt)
            time += dt
    return rows


def trajectory_rows(B, masses, phi0, velocity0, a0, targets, scale_equation):
    rhs = homogeneous_rhs(masses, scale_equation)
    snapshots = integrate(rhs, initial_state(masses, phi0, velocity0, a0), targets)
    nfields = len(masses)
    rows = []
    for tau, state in snapshots:
        phi = state[:nfields]
        prime = state[nfields:2 * nfields]
        a, a_prime, cosmic_time = state[2 * nfields:]
        kinetic = 0.5 * sum(value * value for value in prime) / (a * a)
        potential = 0.5 * sum((mass * value) ** 2
                              for mass, value in zip(masses, phi))
        rho = kinetic + potential
        pressure = kinetic - potential
        row = {
            "tau_code": tau,
            "cosmic_time_code": cosmic_time,
            "efolds": math.log(a / a0),
            "a": a,
            "a_prime": a_prime,
            "H_code": a_prime / (a * a),
            "kinetic_code": kinetic,
            "gradient_code": 0.0,
            "potential_code": potential,
            "rho_code": rho,
            "pressure_code": pressure,
            "B": B,
        }
        for index, (mass, field, field_prime) in enumerate(zip(masses, phi, prime), 1):
            row[f"mass_code_{index}"] = mass
            row[f"phi_{index}"] = field
            row[f"phi_prime_{index}"] = field_prime
        rows.append(row)
    return rows


def nearest_integer_bin(norm):
    # C++ lround for nonnegative mode norms; no possible sqrt(integer) tie here.
    return int(math.floor(norm + 0.5))


def convention_row():
    N, L, B = 8, 2.0, 0.25
    a = 1.25
    mass_code = 0.3
    field_mean, field_amplitude = 1.4, 0.2
    prime_mean, prime_amplitude = -0.1, 0.04
    dx = L / N
    k_mode_lattice = 2.0 * math.sin(math.pi / N) / dx

    kinetic = 0.5 * (prime_mean ** 2 + prime_amplitude ** 2 / 2.0) / a ** 2
    gradient = field_amplitude ** 2 * k_mode_lattice ** 2 / (4.0 * a ** 2)
    potential = 0.5 * mass_code ** 2 * (field_mean ** 2 + field_amplitude ** 2 / 2.0)
    rho = kinetic + gradient + potential
    pressure = kinetic - gradient / 3.0 - potential
    a_prime = a * a * math.sqrt(rho / 3.0)

    selected = []
    for ix in range(N):
        nx = ix if ix <= N // 2 else ix - N
        for iy in range(N):
            ny = iy if iy <= N // 2 else iy - N
            for iz in range(N):
                nz = iz if iz <= N // 2 else iz - N
                squared = nx * nx + ny * ny + nz * nz
                if squared and nearest_integer_bin(math.sqrt(squared)) == 1:
                    k_discrete = 2.0 * math.pi * math.sqrt(squared) / L
                    k_lattice = (2.0 / dx) * math.sqrt(
                        math.sin(math.pi * nx / N) ** 2
                        + math.sin(math.pi * ny / N) ** 2
                        + math.sin(math.pi * nz / N) ** 2)
                    selected.append((k_discrete, k_lattice))

    mode_count = len(selected)
    k_mean_discrete = sum(pair[0] for pair in selected) / mode_count
    k_mean_lattice = sum(pair[1] for pair in selected) / mode_count
    # output.cpp:313 and the two nonzero unnormalised FFT coefficients N^3*A/2.
    inflationeasy_power = (L / B) ** 3 * field_amplitude ** 2 / (2.0 * mode_count)
    dimensionless_discrete = ((B * k_mean_discrete) ** 3
                              * inflationeasy_power / (2.0 * math.pi ** 2))
    dimensionless_lattice = ((B * k_mean_lattice) ** 3
                             * inflationeasy_power / (2.0 * math.pi ** 2))
    return {
        "N": N,
        "L_code": L,
        "B": B,
        "a": a,
        "mass_code": mass_code,
        "field_mean": field_mean,
        "field_amplitude": field_amplitude,
        "prime_mean": prime_mean,
        "prime_amplitude": prime_amplitude,
        "a_prime": a_prime,
        "kinetic_code": kinetic,
        "gradient_code": gradient,
        "potential_code": potential,
        "rho_code": rho,
        "pressure_code": pressure,
        "shell_bin": 1,
        "shell_mode_count": mode_count,
        "inflationeasy_k_output_code": 2.0 * math.pi / L,
        "inflationeasy_power_dimensional": inflationeasy_power,
        "k_mean_discrete_code": k_mean_discrete,
        "k_mean_lattice_code": k_mean_lattice,
        "dimensionless_discrete": dimensionless_discrete,
        "dimensionless_lattice": dimensionless_lattice,
    }


def write_rows(filename, rows):
    path = ROOT / filename
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17e") for key, value in row.items()})
    print(path.name)


def main():
    write_rows("single_field_background.csv", trajectory_rows(
        B=0.25,
        masses=[0.3],
        phi0=[1.4],
        velocity0=[-0.08],
        a0=1.1,
        targets=[0.0, 0.05, 0.1, 0.2],
        scale_equation="inflationeasy",
    ))
    physical_masses = [9.0e-6, 1.0e-6]
    two_field_initial = [13.0, 13.0]
    B_two_field = math.sqrt(0.5 * sum(
        (mass * field) ** 2
        for mass, field in zip(physical_masses, two_field_initial)
    ))
    write_rows("two_field_background.csv", trajectory_rows(
        B=B_two_field,
        masses=[mass / B_two_field for mass in physical_masses],
        phi0=two_field_initial,
        velocity0=[1.0e-10 / B_two_field, 0.0],
        a0=1.0,
        targets=[0.0, 0.05, 0.1, 0.2],
        scale_equation="einstein",
    ))
    write_rows("notion_background_rk4.csv", notion_background_rows())
    write_rows("single_field_conventions.csv", [convention_row()])


if __name__ == "__main__":
    main()
