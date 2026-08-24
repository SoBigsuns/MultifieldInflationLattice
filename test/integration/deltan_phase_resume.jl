@testset "delta-N output phase replay is idempotent after a stale checkpoint" begin
    mktempdir() do directory
        config = test_config(
            vacuum=false,
            integrator="rk4",
            max_steps=1,
            spectra=false,
            deltaN=true,
            output_root=joinpath(directory, "runs"),
        )
        config_data = effective_config_dict(config)
        config_data["output"]["run_name"] = "deltaN_resume"
        input_path = joinpath(directory, "config.toml")
        open(input_path, "w") do io
            TOML.print(io, config_data; sorted=true)
            println(io)
        end
        config = load_config(input_path)
        initialized = initialize_simulation(config)
        state = initialized.state
        model = initialized.model
        lattice = initialized.lattice
        runctx = MFIL.create_run_directory(config, input_path)
        MFIL.initialize_outputs!(runctx, config, input_path, model, lattice)
        cache = MFIL.SimulationCache(state)
        MFIL.write_observation!(runctx, state, model, lattice, config, cache)
        integrator = MFIL.build_integrator(config.evolution.integrator, state)

        progress = MFIL.initialize_deltaN_progress(state, model, lattice, config)
        result = compute_deltaN(
            state, model, lattice, config; progress=progress,
        )
        MFIL._save_run_checkpoint(
            config,
            state,
            model,
            lattice,
            integrator,
            initialized.rng,
            runctx,
            state.step;
            phase="deltaN_output",
            deltaN_progress=progress,
            deltaN_result=result,
            deltaN_outputs_written=false,
        )

        # Simulate a hard crash after the durable CSV transaction completed but
        # before Runner could checkpoint deltaN_outputs_written=true.
        MFIL.write_deltaN_outputs!(runctx, result, state, model, lattice, config)
        summary_before = read(runctx.files[:deltaN_summary])
        spectrum_before = read(runctx.files[:deltaN_spectrum])
        # Also cover a hard stop after metadata finalization but before the
        # terminal phase checkpoint replaces the stale output-phase checkpoint.
        MFIL.finalize_metadata!(runctx, "epsilon_H_reached")
        resumed_dir = resume_simulation(runctx.files[:checkpoint])
        @test resumed_dir == runctx.run_dir
        @test read(runctx.files[:deltaN_summary]) == summary_before
        @test read(runctx.files[:deltaN_spectrum]) == spectrum_before

        final_payload = load_checkpoint(runctx.files[:checkpoint])
        @test final_payload["phase"] == "completed"
        @test final_payload["deltaN_outputs_written"]
        @test haskey(TOML.parsefile(runctx.files[:metadata]), "deltaN")
        @test resume_simulation(runctx.files[:checkpoint]) == runctx.run_dir
    end
end

@testset "Runner preserves delta-N phase and progress on interrupt" begin
    mktempdir() do directory
        config = test_config(
            vacuum=false,
            integrator="rk4",
            max_steps=1,
            spectra=false,
            deltaN=true,
            output_root=joinpath(directory, "runs"),
        )
        config_data = effective_config_dict(config)
        config_data["evolution"]["end_epsilon_H"] = 1.0e-20
        config_data["output"]["run_name"] = "deltaN_interrupt"
        input_path = joinpath(directory, "config.toml")
        open(input_path, "w") do io
            TOML.print(io, config_data; sorted=true)
            println(io)
        end

        MFIL._OUTPUT_FAULT_HOOK[] = function (event, kind, transaction_id,
                                               part_index, key)
            if event == :after_part && kind == :deltaN && part_index == 1
                throw(InterruptException())
            end
        end
        try
            @test_throws InterruptException run_simulation(input_path)
        finally
            MFIL._OUTPUT_FAULT_HOOK[] = nothing
        end

        run_dirs = filter(isdir, readdir(config_data["output"]["root"]; join=true))
        @test length(run_dirs) == 1
        run_dir = only(run_dirs)
        checkpoint_path = joinpath(run_dir, "checkpoint.jld2")
        interrupted = load_checkpoint(checkpoint_path)
        @test interrupted["phase"] == "deltaN_output"
        @test interrupted["deltaN_progress"] isa MFIL.DeltaNProgress
        @test MFIL.deltaN_progress_complete(interrupted["deltaN_progress"])
        @test interrupted["deltaN_result"] isa MFIL.DeltaNResult
        @test !interrupted["deltaN_outputs_written"]
        @test isdir(joinpath(run_dir, ".output-transaction"))

        @test resume_simulation(checkpoint_path) == run_dir
        completed = load_checkpoint(checkpoint_path)
        @test completed["phase"] == "completed"
        @test completed["deltaN_outputs_written"]
        @test haskey(TOML.parsefile(joinpath(run_dir, "metadata.toml")), "deltaN")
        @test !ispath(joinpath(run_dir, ".output-transaction"))
    end
end
