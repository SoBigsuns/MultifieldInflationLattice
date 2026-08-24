@testset "Bunch-Davies initialization" begin
    data = effective_config_dict(test_config(vacuum=true))
    data["parallel"]["fft_threads"] = 2
    data["parallel"]["deterministic_reductions"] = true
    config = MFIL.config_from_dict(data)
    MFIL.FFTW.set_num_threads(1)
    first_run = initialize_simulation(config)
    second_run = initialize_simulation(config)
    state = first_run.state
    @test MFIL.FFTW.get_num_threads() == 2
    @test first_run.workspace.deterministic_reductions
    @test size(state.field) == (8, 8, 8, 2)
    @test state.field == second_run.state.field
    @test state.field_prime == second_run.state.field_prime
    @test mean(@view(state.field[:, :, :, 1])) ≈ 13.0 atol=64eps(Float64)
    @test mean(@view(state.field[:, :, :, 2])) ≈ 13.0 atol=64eps(Float64)
    @test !(@view(state.field[:, :, :, 1]) == @view(state.field[:, :, :, 2]))
    energy = energy_summary(state, first_run.model, first_run.lattice)
    @test MFIL.friedmann_residual(state, energy.rho) <= 1.0e-12
    @test all(isfinite, state.field)
    @test all(isfinite, state.field_prime)
end

@testset "energy, slow roll, and spectra" begin
    lattice = Lattice3D(8, 2.0)
    potential = QuadraticPotential([1.0])
    model = SimulationModel(potential, 1.0, 1.0, ["phi"])
    field = fill(2.0, 8, 8, 8, 1)
    prime = fill(3.0, 8, 8, 8, 1)
    rho = 0.5 * 3.0^2 / 2.0^2 + 0.5 * 2.0^2
    state = SimulationState(field, prime, 2.0, 4sqrt(rho / 3), 0.0, 0.0, log(2.0), 0)
    energy = energy_summary(state, model, lattice)
    @test energy.kinetic_total ≈ 1.125
    @test energy.gradient_total == 0.0
    @test energy.potential_total == 2.0
    @test energy.rho ≈ 3.125
    @test energy.p ≈ -0.875
    slowroll = slowroll_summary(state, model, lattice, energy)
    @test slowroll.epsilon_H ≈ (energy.rho + energy.p) / (2 * MFIL.hubble_code(state)^2)
    @test length(slowroll.etaV_eigenvalues) == 1

    map = [cos(2pi * (i - 1) / 8) for i in 1:8, j in 1:8, k in 1:8]
    spectrum = scalar_spectrum(map, state, model, lattice)
    @test sum(row.n_modes for row in spectrum) == 8^3 - 1
    @test all(row.power_dimensionless >= 0 for row in spectrum)
    @test any(row.power_dimensionless > 0 for row in spectrum)
end

@testset "linear curvature decomposition" begin
    config = test_config(vacuum=false)
    initialized = initialize_simulation(config)
    state = initialized.state
    lattice = initialized.lattice
    for k in 1:8, j in 1:8, i in 1:8
        state.field[i, j, k, 1] += 1.0e-4 * cos(2pi * (i - 1) / 8)
        state.field[i, j, k, 2] += 2.0e-4 * cos(2pi * (i - 1) / 8 + 0.3)
        state.field_prime[i, j, k, 2] = 0.5 * state.field_prime[i, j, k, 1]
    end
    rows = linear_curvature_spectra(state, initialized.model, lattice, config, 0)
    bins = unique(row.k_bin for row in rows)
    for bin in bins
        selected = filter(row -> row.k_bin == bin, rows)
        total = only(row.power_dimensionless for row in selected if row.component == "total")
        decomposed = sum(row.power_dimensionless for row in selected if row.component == "self") +
                     2sum(row.power_dimensionless for row in selected if row.component == "cross")
        @test isapprox(total, decomposed; rtol=2.0e-11, atol=1.0e-14)
    end
    maps = MFIL.linear_curvature_maps(state; min_speed2=1.0e-30)
    @test maps.total ≈ reduce(+, maps.components)

    fill!(state.field_prime, 0.0)
    @test all(isnan, linear_curvature_map(state; min_speed2=1.0e-30))
end

@testset "delta-N uniform reference surface" begin
    config = test_config(vacuum=false, deltaN=true)
    initialized = initialize_simulation(config)
    result = compute_deltaN(initialized.state, initialized.model, initialized.lattice, config)
    @test isempty(result.failed_indices)
    @test result.local_efolds == zeros(8, 8, 8)
    @test result.zeta == zeros(8, 8, 8)
    @test abs(result.mean) <= eps(Float64)
    @test result.variance == 0.0
    @test sum(row.n_modes for row in result.spectrum) == 8^3 - 1
end

@testset "uniform-density crossing interpolation" begin
    @test MFIL.interpolate_density_crossing(0.0, 0.1, 2.0, 1.0, 1.5) == 0.05
    @test MFIL.interpolate_density_crossing(0.0, -0.1, 1.0, 2.0, 1.5) == -0.05
    @test_throws ArgumentError MFIL.interpolate_density_crossing(0.0, 0.1, 2.0, 1.0, 3.0)
end
