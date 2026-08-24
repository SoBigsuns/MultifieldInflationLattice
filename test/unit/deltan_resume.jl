function deltaN_resume_fixture(; integrator="rk4", scale=1.0)
    n = 2
    lattice = Lattice3D(n, 2.0 * scale)
    model = SimulationModel(
        QuadraticPotential([1.0 / scale]), scale, 1.0, ["phi"],
    )
    field = fill(1.9, n, n, n, 1)
    field[1, :, :, :] .= 2.1
    state = SimulationState(
        field,
        fill(-1.0 / scale, n, n, n, 1),
        10.0,
        80.0 / scale,
        0.0,
        0.0,
        log(10.0),
        0,
    )
    config = (
        parallel=(deterministic_reductions=true,),
        deltaN=(
            enabled=true,
            integrator=integrator,
            reference_surface="uniform_density_at_end",
            dN=1.0e-3,
            max_extra_efolds=2.0,
            max_backward_efolds=2.0,
            separate_universe_min_ratio=0.0,
            strict_separate_universe=false,
        ),
    )
    return state, model, lattice, config
end

@testset "delta-N progress checkpoint and exact unfinished-site resume" begin
    state, model, lattice, config = deltaN_resume_fixture()
    reference = compute_deltaN(state, model, lattice, config)
    progress = MFIL.initialize_deltaN_progress(state, model, lattice, config)
    @test progress.finished isa Array{Bool,3}
    deterministic_cache = MFIL.SimulationCache(
        state; deterministic_reductions=true,
    )
    @test progress.rho_reference_code ==
          energy_summary(state, model, lattice, deterministic_cache).rho

    mktempdir() do directory
        checkpoint = joinpath(directory, "deltaN-progress.jld2")
        callback_count = Ref(0)
        callback = function (current)
            callback_count[] += 1
            save_checkpoint(checkpoint, Dict(
                "phase" => "deltaN",
                "deltaN_progress" => current,
                "last_output_ids" => Dict(:background => -1),
            ))
            throw(InterruptException())
        end
        @test_throws InterruptException compute_deltaN(
            state,
            model,
            lattice,
            config;
            progress=progress,
            checkpoint_callback=callback,
            checkpoint_every_sweeps=37,
        )
        @test callback_count[] == 1
        @test progress.sweep == 37
        @test !MFIL.deltaN_progress_complete(progress)

        restored = load_checkpoint(checkpoint)["deltaN_progress"]
        @test restored.sweep == progress.sweep
        @test restored.local_field == progress.local_field
        @test restored.local_u == progress.local_u
        @test restored.local_efolds == progress.local_efolds
        @test restored.rho_previous == progress.rho_previous
        @test restored.finished == progress.finished

        resumed = compute_deltaN(
            state, model, lattice, config; progress=restored,
        )
        @test resumed.local_efolds == reference.local_efolds
        @test resumed.zeta == reference.zeta
        @test isempty(resumed.failed_indices)
        @test MFIL.deltaN_progress_complete(restored)

        terminal_sweep = restored.sweep
        terminal_field = copy(restored.local_field)
        MFIL.advance_deltaN!(
            restored, state, model, lattice, config; max_sweeps=10,
        )
        @test restored.sweep == terminal_sweep
        @test restored.local_field == terminal_field
    end
end

@testset "delta-N density classification is B-rescaling invariant" begin
    state1, model1, lattice1, config = deltaN_resume_fixture(scale=1.0)
    state2, model2, lattice2, _ = deltaN_resume_fixture(scale=1.0e12)
    progress1 = MFIL.initialize_deltaN_progress(state1, model1, lattice1, config)
    progress2 = MFIL.initialize_deltaN_progress(state2, model2, lattice2, config)
    @test progress1.status == progress2.status
    @test progress1.directions == progress2.directions
    @test count(==(Int8(1)), progress2.directions) == 4
    @test count(==(Int8(-1)), progress2.directions) == 4
end

@testset "delta-N staggered leapfrog is reversible and second order" begin
    potential = QuadraticPotential([1.0, 0.7])
    workspace = MFIL.DeltaNPatchWorkspace(2)
    workspace.phi .= (2.0, 1.4)
    workspace.u .= (-0.1, 0.03)
    initial_phi = copy(workspace.phi)
    initial_u = copy(workspace.u)
    @test MFIL._patch_leapfrog_step!(workspace, potential, 1.0e-3)
    @test MFIL._patch_leapfrog_step!(workspace, potential, -1.0e-3)
    @test workspace.phi ≈ initial_phi atol=1.0e-13
    @test workspace.u ≈ initial_u atol=1.0e-13

    function evolved_patch(h)
        work = MFIL.DeltaNPatchWorkspace(2)
        work.phi .= (2.0, 1.4)
        work.u .= (-0.1, 0.03)
        for _ in 1:round(Int, 0.2 / h)
            @test MFIL._patch_leapfrog_step!(work, potential, h)
        end
        return vcat(work.phi, work.u)
    end
    coarse = evolved_patch(0.02)
    medium = evolved_patch(0.01)
    fine = evolved_patch(0.005)
    convergence_ratio = norm(coarse - medium) / norm(medium - fine)
    @test 3.8 < convergence_ratio < 4.2

    state, model, lattice, config = deltaN_resume_fixture(integrator="leapfrog")
    result = compute_deltaN(state, model, lattice, config)
    @test isempty(result.failed_indices)
    @test result.n_forward == 4
    @test result.n_backward == 4
    @test abs(sum(result.zeta)) < 1.0e-12
end
