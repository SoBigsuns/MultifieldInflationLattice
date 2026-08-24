const REFERENCE_DIR = @__DIR__

function read_numeric_reference(filename)
    lines = filter(line -> !isempty(strip(line)),
                   readlines(joinpath(REFERENCE_DIR, filename)))
    names = Tuple(Symbol.(split(first(lines), ',')))
    return [NamedTuple{names}(Tuple(parse(Float64, value) for value in split(line, ',')))
            for line in Iterators.drop(lines, 1)]
end

function homogeneous_reference_state(row, nfields; lattice_size=8, box_size=2.0)
    lattice = Lattice3D(lattice_size, box_size)
    masses = [getproperty(row, Symbol("mass_code_", q)) for q in 1:nfields]
    fields = zeros(Float64, lattice_size, lattice_size, lattice_size, nfields)
    primes = similar(fields)
    for q in 1:nfields
        fill!(@view(fields[:, :, :, q]), getproperty(row, Symbol("phi_", q)))
        fill!(@view(primes[:, :, :, q]), getproperty(row, Symbol("phi_prime_", q)))
    end
    model = SimulationModel(
        QuadraticPotential(masses), row.B, row.a,
        ["phi_$q" for q in 1:nfields],
    )
    state = SimulationState(
        fields, primes, row.a, row.a_prime, row.tau_code,
        row.cosmic_time_code, row.efolds, 0,
    )
    return state, model, lattice
end

function compare_homogeneous_trajectory(filename, nfields)
    rows = read_numeric_reference(filename)
    state, model, lattice = homogeneous_reference_state(first(rows), nfields)
    integrator = RK4Integrator(state; deterministic_reductions=true)

    for expected in rows
        while state.tau_code < expected.tau_code - 8eps(expected.tau_code)
            MFIL.step!(integrator, state, model, lattice,
                       min(1.0e-3, expected.tau_code - state.tau_code))
        end
        energy = energy_summary(state, model, lattice, integrator.dynamics)
        background = background_summary(state, model, lattice, energy)
        for q in 1:nfields
            @test mean(@view(state.field[:, :, :, q])) ≈
                  getproperty(expected, Symbol("phi_", q)) rtol=3.0e-10 atol=3.0e-12
            @test mean(@view(state.field_prime[:, :, :, q])) ≈
                  getproperty(expected, Symbol("phi_prime_", q)) rtol=3.0e-10 atol=3.0e-12
            @test background.field_velocities[q] ≈
                  expected.B * getproperty(expected, Symbol("phi_prime_", q)) / expected.a
        end
        @test state.tau_code ≈ expected.tau_code atol=2eps(Float64)
        @test state.cosmic_time_code ≈ expected.cosmic_time_code rtol=3.0e-10 atol=3.0e-12
        @test state.efolds ≈ expected.efolds rtol=3.0e-10 atol=3.0e-12
        @test state.a ≈ expected.a rtol=3.0e-10 atol=3.0e-12
        @test state.a_prime ≈ expected.a_prime rtol=3.0e-10 atol=3.0e-12
        @test MFIL.hubble_code(state) ≈ expected.H_code rtol=3.0e-10 atol=3.0e-12
        @test energy.kinetic_total ≈ expected.kinetic_code rtol=3.0e-10 atol=3.0e-12
        @test energy.gradient_total ≈ expected.gradient_code atol=2.0e-14
        @test energy.potential_total ≈ expected.potential_code rtol=3.0e-10 atol=3.0e-12
        @test energy.rho ≈ expected.rho_code rtol=3.0e-10 atol=3.0e-12
        @test energy.p ≈ expected.pressure_code rtol=3.0e-10 atol=3.0e-12
        @test background.H ≈ expected.B * expected.H_code rtol=3.0e-10 atol=3.0e-12
        @test background.rho ≈ expected.B^2 * expected.rho_code rtol=3.0e-10 atol=3.0e-12
        @test background.p ≈ expected.B^2 * expected.pressure_code rtol=3.0e-10 atol=3.0e-12
    end
end

@testset "pinned reference provenance" begin
    provenance = TOML.parsefile(joinpath(REFERENCE_DIR, "PROVENANCE.toml"))
    @test provenance["source"]["commit"] == MFIL.INFLATION_EASY_REFERENCE
    @test provenance["source"]["repository"] ==
          "https://github.com/caravangelo/inflation-easy"
    @test all(length(digest) == 64 for digest in values(provenance["source"]["sha256"]))
    @test provenance["generation"]["python_dependencies"] == "standard library only"
    @test provenance["notion_reference"]["url"] ==
          "https://pie-supply-358.notion.site/6-4-Runge-Kutta-c969f1759b8b4d9a894506418c6dda7f"
    @test provenance["notion_reference"]["page_id"] ==
          "c969f175-9b8b-4d9a-8945-06418c6dda7f"
    @test count(endswith("_block"), keys(provenance["notion_reference"])) == 5
end

@testset "InflationEasy-compatible single-field background" begin
    compare_homogeneous_trajectory("single_field_background.csv", 1)
end

@testset "independent two-field homogeneous background" begin
    initial = first(read_numeric_reference("two_field_background.csv"))
    sample = load_config(SAMPLE_CONFIG)
    expected_B = MFIL.resolve_rescaling_B(sample)
    @test initial.B ≈ expected_B rtol=2eps(Float64)
    for q in 1:2
        @test getproperty(initial, Symbol("phi_", q)) == sample.initial.field_means[q]
        @test initial.B * getproperty(initial, Symbol("mass_code_", q)) ≈
              sample.model.masses[q] rtol=2eps(Float64)
        @test initial.B * getproperty(initial, Symbol("phi_prime_", q)) / initial.a ≈
              sample.initial.cosmic_velocities[q] atol=2eps(Float64)
    end
    compare_homogeneous_trajectory("two_field_background.csv", 2)
end

function notion_observation(state, model, lattice, workspace)
    energy = energy_summary(state, model, lattice, workspace)
    background = background_summary(state, model, lattice, energy)
    epsilon = (energy.rho + energy.p) / (2 * MFIL.hubble_code(state)^2)
    return (
        time_code=state.cosmic_time_code,
        efolds=state.efolds,
        a=state.a,
        H_physical=background.H,
        epsilon,
        phi=background.field_means[1],
        phi_dot_physical=background.field_velocities[1],
        psi=background.field_means[2],
        psi_dot_physical=background.field_velocities[2],
    )
end

@testset "public Notion two-field background RK4" begin
    rows = read_numeric_reference("notion_background_rk4.csv")
    first_row = first(rows)
    @test first_row.phi_dot_physical == 0.0
    @test first_row.psi_dot_physical == 0.0
    @test first_row.efolds == 0.0
    @test first_row.dt_next_physical ≈ 0.01 / first_row.H_physical

    n = 2
    lattice = Lattice3D(n, 2.0)
    masses_code = [
        first_row.mass_phi_physical / first_row.B,
        first_row.mass_psi_physical / first_row.B,
    ]
    model = SimulationModel(
        QuadraticPotential(masses_code), first_row.B, 1.0, ["phi", "psi"],
    )
    field = zeros(Float64, n, n, n, 2)
    prime = zeros(Float64, n, n, n, 2)
    fill!(@view(field[:, :, :, 1]), first_row.phi)
    fill!(@view(field[:, :, :, 2]), first_row.psi)
    state = SimulationState(
        field, prime, 1.0, first_row.H_physical / first_row.B,
        0.0, 0.0, 0.0, 0,
    )
    integrator = RK4Integrator(state; deterministic_reductions=true)
    step_size = 2.0e-5
    provenance = TOML.parsefile(joinpath(REFERENCE_DIR, "PROVENANCE.toml"))
    rtol = provenance["notion_background"]["comparison_relative_tolerance"]
    atol = provenance["notion_background"]["comparison_absolute_tolerance"]

    for expected in rows
        target = expected.time_code
        before = notion_observation(state, model, lattice, integrator.dynamics)
        after = before
        while state.cosmic_time_code < target
            before = notion_observation(state, model, lattice, integrator.dynamics)
            MFIL.step!(integrator, state, model, lattice, step_size)
            after = notion_observation(state, model, lattice, integrator.dynamics)
        end
        fraction = after.time_code == before.time_code ? 0.0 :
                   (target - before.time_code) / (after.time_code - before.time_code)
        interpolate(name) = getproperty(before, name) + fraction *
            (getproperty(after, name) - getproperty(before, name))
        for name in (:efolds, :a, :H_physical, :epsilon, :phi,
                     :phi_dot_physical, :psi, :psi_dot_physical)
            @test isapprox(interpolate(name), getproperty(expected, name); rtol, atol)
        end
        @test expected.dt_next_physical ≈ 0.01 / expected.H_physical rtol=4eps(Float64)
    end
end

@testset "InflationEasy energy and analytic plane-wave spectrum conventions" begin
    expected = only(read_numeric_reference("single_field_conventions.csv"))
    n = round(Int, expected.N)
    lattice = Lattice3D(n, expected.L_code)
    model = SimulationModel(
        QuadraticPotential([expected.mass_code]), expected.B, 1.0, ["phi"],
    )
    field = zeros(Float64, n, n, n, 1)
    prime = similar(field)
    for k in 1:n, j in 1:n, i in 1:n
        phase = 2pi * (i - 1) / n
        field[i, j, k, 1] = expected.field_mean + expected.field_amplitude * cos(phase)
        prime[i, j, k, 1] = expected.prime_mean + expected.prime_amplitude * sin(phase)
    end
    state = SimulationState(
        field, prime, expected.a, expected.a_prime,
        0.0, 0.0, 0.0, 0,
    )
    workspace = MFIL.DynamicsWorkspace(state; deterministic_reductions=true)
    energy = energy_summary(state, model, lattice, workspace)
    physical = physical_energy_summary(energy, model)
    @test energy.kinetic_total ≈ expected.kinetic_code rtol=3.0e-14
    @test energy.gradient_total ≈ expected.gradient_code rtol=3.0e-14
    @test energy.potential_total ≈ expected.potential_code rtol=3.0e-14
    @test energy.rho ≈ expected.rho_code rtol=3.0e-14
    @test energy.p ≈ expected.pressure_code rtol=3.0e-14
    @test physical.rho ≈ expected.B^2 * expected.rho_code rtol=3.0e-14
    @test physical.rho != expected.B * expected.rho_code
    @test MFIL.friedmann_residual(state, energy.rho) <= 64eps(Float64)

    plane_wave = @view field[:, :, :, 1]
    lattice_rows = scalar_spectrum(plane_wave, state, model, lattice)
    continuum_rows = scalar_cross_spectrum(
        plane_wave, plane_wave, state, model, lattice;
        use_lattice_momentum=false,
    )
    bin = round(Int, expected.shell_bin)
    lattice_shell = only(row for row in lattice_rows if row.k_bin == bin)
    continuum_shell = only(row for row in continuum_rows if row.k_bin == bin)
    @test lattice_shell.n_modes == round(Int, expected.shell_mode_count)
    @test continuum_shell.n_modes == round(Int, expected.shell_mode_count)
    @test expected.inflationeasy_k_output_code ≈ 2pi / expected.L_code
    @test lattice_shell.k_comoving ≈ expected.B * expected.k_mean_lattice_code rtol=2.0e-14
    @test continuum_shell.k_comoving ≈ expected.B * expected.k_mean_discrete_code rtol=2.0e-14
    @test lattice_shell.power_dimensionless ≈ expected.dimensionless_lattice rtol=4.0e-14
    @test continuum_shell.power_dimensionless ≈ expected.dimensionless_discrete rtol=4.0e-14

    recovered_ie_power = 2pi^2 * continuum_shell.power_dimensionless /
                         continuum_shell.k_comoving^3
    @test recovered_ie_power ≈ expected.inflationeasy_power_dimensional rtol=4.0e-14
end
