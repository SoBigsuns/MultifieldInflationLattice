function homogeneous_state()
    lattice = Lattice3D(8, 2.0)
    model = SimulationModel(QuadraticPotential([0.2]), 1.0, 1.0, ["phi"])
    field = fill(1.0, 8, 8, 8, 1)
    prime = fill(0.1, 8, 8, 8, 1)
    energy0 = 0.5 * 0.1^2 + 0.5 * 0.2^2
    state = SimulationState(field, prime, 1.0, sqrt(energy0 / 3), 0.0, 0.0, 0.0, 0)
    return state, model, lattice
end

function integrate_fixed(integrator_type, h, steps)
    state, model, lattice = homogeneous_state()
    integrator = integrator_type(state)
    for _ in 1:steps
        MFIL.step!(integrator, state, model, lattice, h)
    end
    return state, integrator, model, lattice
end

@testset "fixed-step integrators" begin
    for integrator_type in (RK4Integrator, LeapfrogIntegrator)
        state, integrator, model, lattice = integrate_fixed(integrator_type, 0.002, 10)
        view = MFIL.synchronize!(integrator, state, model, lattice, nothing)
        @test view.step == 10
        @test state.tau_code ≈ 0.02
        @test all(isfinite, state.field)
        energy = energy_summary(state, model, lattice)
        @test MFIL.friedmann_residual(state, energy.rho) < 1.0e-6
    end

    for integrator_type in (RK4Integrator, LeapfrogIntegrator)
        coarse, _, _, _ = integrate_fixed(integrator_type, 0.04, 5)
        medium, _, _, _ = integrate_fixed(integrator_type, 0.02, 10)
        fine, _, _, _ = integrate_fixed(integrator_type, 0.01, 20)
        ratio = abs(mean(coarse.field) - mean(medium.field)) /
                abs(mean(medium.field) - mean(fine.field))
        if integrator_type === RK4Integrator
            @test 8.0 < ratio < 24.0 # approaches 2^4
        else
            @test 3.5 < ratio < 4.5 # approaches 2^2
        end
    end
end

@testset "checkpoint round trip and staggered restart" begin
    continuous, _, _, _ = integrate_fixed(LeapfrogIntegrator, 0.002, 8)
    split_state, split_integrator, model, lattice = integrate_fixed(LeapfrogIntegrator, 0.002, 4)
    mktempdir() do directory
        path = joinpath(directory, "checkpoint.jld2")
        save_checkpoint(path, Dict(
            "state" => split_state,
            "integrator" => split_integrator,
            "model" => model,
            "lattice" => lattice,
            "last_output_ids" => Dict(:background => -1),
        ))
        restored = load_checkpoint(path)
        state = restored["state"]
        integrator = restored["integrator"]
        for _ in 1:4
            MFIL.step!(integrator, state, restored["model"], restored["lattice"], 0.002)
        end
        @test state.field == continuous.field
        @test state.field_prime == continuous.field_prime
        @test state.a == continuous.a
        @test state.a_prime == continuous.a_prime
        @test isfile(path)
    end
end
