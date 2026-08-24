@testset "deterministic reductions use fixed z-slab slots" begin
    n, nf = 8, 2
    lattice = Lattice3D(n, 2.5)
    potential = QuadraticPotential([0.31, 0.07])
    model = SimulationModel(potential, 0.2, 1.0, ["phi", "psi"])
    field = Array{Float64}(undef, n, n, n, nf)
    prime = similar(field)
    for q in 1:nf, k in 1:n, j in 1:n, i in 1:n
        field[i, j, k, q] = (-1.0)^(i + q) *
            (0.13q + 0.007i - 0.003j + 0.011k + 1.0e-7i * j * k)
        prime[i, j, k, q] = 0.004q + (-1.0)^(j + k) *
            (0.021q - 0.0004i + 0.0009j + 0.0002k)
    end
    state = SimulationState(field, prime, 1.3, 0.2, 0.0, 0.0, 0.0, 0)
    workspace = MFIL.DynamicsWorkspace(state; deterministic_reductions=true)
    dfield, dprime = similar(field), similar(field)
    MFIL.dynamics_rhs!(
        dfield, dprime, field, prime, state.a, state.a_prime,
        model, lattice, workspace,
    )

    slots = max(n, Threads.maxthreadid())
    expected_kinetic = zeros(Float64, nf, slots)
    expected_gradient = zeros(Float64, nf, slots)
    expected_potential = zeros(Float64, slots)
    local_fields = zeros(Float64, nf)
    inva2 = inv(state.a * state.a)
    for k in 1:n, j in 1:n, i in 1:n
        for q in 1:nf
            local_fields[q] = field[i, j, k, q]
        end
        expected_potential[k] += potential_value(potential, local_fields)
        for q in 1:nf
            expected_kinetic[q, k] += 0.5 * prime[i, j, k, q]^2 * inva2
            expected_gradient[q, k] +=
                -0.5 * field[i, j, k, q] * workspace.laplacian[i, j, k, q] * inva2
        end
    end
    @test workspace.deterministic_reductions
    @test workspace.kinetic_partial == expected_kinetic
    @test workspace.gradient_partial == expected_gradient
    @test workspace.potential_partial == expected_potential

    # Energy.jl owns separate reductions; compare it with an explicitly
    # z-slabbed forward-edge sum rather than with Dynamics.jl's result.
    edge_kinetic = zeros(Float64, nf, slots)
    edge_gradient = zeros(Float64, nf, slots)
    edge_potential = zeros(Float64, slots)
    invdx2 = inv(lattice.dx * lattice.dx)
    for k in 1:n, j in 1:n, i in 1:n
        ip, jp, kp = mod1(i + 1, n), mod1(j + 1, n), mod1(k + 1, n)
        for q in 1:nf
            local_fields[q] = field[i, j, k, q]
            edge_kinetic[q, k] += 0.5 * prime[i, j, k, q]^2 * inva2
            dx = field[ip, j, k, q] - field[i, j, k, q]
            dy = field[i, jp, k, q] - field[i, j, k, q]
            dz = field[i, j, kp, q] - field[i, j, k, q]
            edge_gradient[q, k] += 0.5 * inva2 * invdx2 *
                                    (dx * dx + dy * dy + dz * dz)
        end
        edge_potential[k] += potential_value(potential, local_fields)
    end
    energy = energy_summary(state, model, lattice, workspace)
    expected_kinetic_by_field = vec(sum(edge_kinetic; dims=2)) ./ n^3
    expected_gradient_by_field = vec(sum(edge_gradient; dims=2)) ./ n^3
    @test energy.kinetic_by_field == expected_kinetic_by_field
    @test energy.gradient_by_field == expected_gradient_by_field
    @test energy.potential_total == sum(edge_potential) / n^3

    gradient_partial = zeros(Float64, nf, slots)
    for k in 1:n, j in 1:n, i in 1:n, q in 1:nf
        gradient_partial[q, k] += potential.mass_squared[q] * field[i, j, k, q]
    end
    expected_mean_gradient = vec(sum(gradient_partial; dims=2)) ./ n^3
    @test MFIL._mean_potential_gradient(
        state, model; deterministic_reductions=true,
    ) == expected_mean_gradient
    slowroll = slowroll_summary(
        state, model, lattice, energy, workspace; min_speed2=1.0e-30,
    )
    @test all(isfinite, (slowroll.eta_parallel, slowroll.turn_rate))
end

function physically_equivalent_state(B)
    n = 8
    a = 1.1
    physical_box_size = 8.0
    lattice = Lattice3D(n, B * physical_box_size)
    physical_masses = [0.08, 0.03]
    physical_velocities = [2.0e-4, -1.0e-4]
    model = SimulationModel(
        QuadraticPotential(physical_masses ./ B), B, a, ["phi", "psi"],
    )
    field = zeros(Float64, n, n, n, 2)
    prime = similar(field)
    for k in 1:n, j in 1:n, i in 1:n
        phase = 2pi * (i - 1) / n
        field[i, j, k, 1] = 1.2 + 0.01cos(phase)
        field[i, j, k, 2] = -0.7 + 0.02sin(phase)
        prime[i, j, k, 1] = a * physical_velocities[1] / B
        prime[i, j, k, 2] = a * physical_velocities[2] / B
    end
    state = SimulationState(field, prime, a, 1.0, 0.0, 0.0, 0.0, 0)
    workspace = MFIL.DynamicsWorkspace(state; deterministic_reductions=true)
    energy = energy_summary(state, model, lattice, workspace)
    state.a_prime = a^2 * sqrt(energy.rho / 3)
    return state, model, lattice, workspace, energy
end

@testset "physical thresholds and slow-roll definitions are B invariant" begin
    one = physically_equivalent_state(0.5)
    two = physically_equivalent_state(0.125)
    state1, model1, lattice1, workspace1, energy1 = one
    state2, model2, lattice2, workspace2, energy2 = two
    physical1 = physical_energy_summary(energy1, model1)
    physical2 = physical_energy_summary(energy2, model2)
    for property in (:kinetic_total, :gradient_total, :potential_total, :rho, :p)
        @test getproperty(physical1, property) ≈ getproperty(physical2, property) rtol=5.0e-14
    end

    low_threshold = 1.0e-9
    slow1 = slowroll_summary(
        state1, model1, lattice1, energy1, workspace1;
        min_speed2=low_threshold,
    )
    slow2 = slowroll_summary(
        state2, model2, lattice2, energy2, workspace2;
        min_speed2=low_threshold,
    )
    for property in (:epsilon_H, :epsilon_V, :eta_parallel, :turn_rate,
                     :turn_rate_over_H)
        @test getproperty(slow1, property) ≈ getproperty(slow2, property) rtol=2.0e-12 atol=2.0e-14
    end
    @test slow1.epsilonH_by_field ≈ slow2.epsilonH_by_field rtol=2.0e-12
    @test slow1.etaV_diag ≈ slow2.etaV_diag rtol=2.0e-12
    @test slow1.etaV_eigenvalues ≈ slow2.etaV_eigenvalues rtol=2.0e-12

    defined_config = (spectra=(
        linear_zeta_min_speed2=low_threshold,
        use_lattice_momentum=true,
    ),)
    map1 = MFIL.linear_curvature_map(state1, model1, lattice1, defined_config)
    map2 = MFIL.linear_curvature_map(state2, model2, lattice2, defined_config)
    @test all(isfinite, map1)
    @test map1 ≈ map2 rtol=2.0e-13 atol=2.0e-14

    function full_config(B, threshold)
        data = effective_config_dict(load_config(SAMPLE_CONFIG))
        data["model"]["rescale_B"] = B
        data["spectra"]["linear_zeta_min_speed2"] = threshold
        return MFIL.config_from_dict(data)
    end
    public_config1 = full_config(model1.B, low_threshold)
    public_config2 = full_config(model2.B, low_threshold)
    @test MFIL.linear_curvature_map(state1, public_config1) ≈ map1 rtol=2.0e-13 atol=2.0e-14
    @test MFIL.linear_curvature_map(state2, public_config2) ≈ map2 rtol=2.0e-13 atol=2.0e-14

    high_threshold = 1.0e-6
    undefined_config = (spectra=(
        linear_zeta_min_speed2=high_threshold,
        use_lattice_momentum=true,
    ),)
    @test all(isnan, MFIL.linear_curvature_map(
        state1, model1, lattice1, undefined_config,
    ))
    @test all(isnan, MFIL.linear_curvature_map(
        state2, model2, lattice2, undefined_config,
    ))
    @test all(isnan, MFIL.linear_curvature_map(
        state1, full_config(model1.B, high_threshold),
    ))
    @test all(isnan, MFIL.linear_curvature_map(
        state2, full_config(model2.B, high_threshold),
    ))
    undefined1 = slowroll_summary(
        state1, model1, lattice1, energy1, workspace1;
        min_speed2=high_threshold,
    )
    undefined2 = slowroll_summary(
        state2, model2, lattice2, energy2, workspace2;
        min_speed2=high_threshold,
    )
    @test all(isnan, (undefined1.eta_parallel, undefined1.turn_rate,
                      undefined2.eta_parallel, undefined2.turn_rate))

    tiny_field = zeros(Float64, 1, 1, 1, 1)
    tiny_state = SimulationState(
        tiny_field, copy(tiny_field), 1.0, 1.0e-150, 0.0, 0.0, 0.0, 0,
    )
    @test MFIL.friedmann_residual(tiny_state, 3.3e-300) ≈ (0.1 / 1.1) rtol=2.0e-14
end

@testset "validation reports output capacity and I/O risk" begin
    data = effective_config_dict(load_config(SAMPLE_CONFIG))
    data["lattice"]["size"] = 8
    data["output"]["slice_index"] = 4
    data["output"]["save_3d_snapshots"] = true
    mktempdir() do directory
        path = joinpath(directory, "io_heavy.toml")
        open(path, "w") do stream
            TOML.print(stream, data)
        end
        stream = IOBuffer()
        returned = validate_config_file(path; io=stream)
        message = String(take!(stream))
        @test returned isa SimulationConfig
        @test occursin("estimated output capacity:", message)
        @test occursin("MiB", message)
        @test occursin("warning: configured output volume may dominate runtime or storage", message)
    end
end

@testset "nontrivial delta-N forward and backward density crossings" begin
    n = 2
    lattice = Lattice3D(n, 2.0)
    model = SimulationModel(QuadraticPotential([0.2]), 1.0, 1.0, ["phi"])
    field = fill(5.0, n, n, n, 1)
    prime = zeros(Float64, n, n, n, 1)
    for k in 1:n, j in 1:n, i in 1:n
        prime[i, j, k, 1] = i == 1 ? 0.5 : 1.0
    end
    rho_reference = 0.5 * (0.5 + 0.5^2 / 2 + 0.5 + 1.0^2 / 2)
    state = SimulationState(
        field, prime, 1.0, sqrt(rho_reference / 3),
        0.0, 0.0, 0.0, 0,
    )
    config = (
        reference_surface="uniform_density_at_end",
        dN=1.0e-3,
        max_extra_efolds=3.0,
        max_backward_efolds=3.0,
        integrator="rk4",
        separate_universe_min_ratio=1.0e-12,
        strict_separate_universe=true,
    )
    result = compute_deltaN(state, model, lattice, config)
    @test isempty(result.failed_indices)
    @test result.n_forward == n^3 ÷ 2
    @test result.n_backward == n^3 ÷ 2
    @test all(result.local_efolds[2, :, :] .> 0.0)
    @test all(result.local_efolds[1, :, :] .< 0.0)
    @test all(isfinite, result.zeta)
    @test abs(mean(result.zeta)) <= 32eps(Float64)
end
