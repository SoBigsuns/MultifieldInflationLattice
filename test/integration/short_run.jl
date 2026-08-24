@testset "short CLI-equivalent run, resume, and analysis" begin
    mktempdir() do directory
        config = test_config(
            vacuum=false,
            integrator="rk4",
            max_steps=1,
            spectra=false,
            deltaN=false,
            output_root=joinpath(directory, "runs"),
        )
        data = effective_config_dict(config)
        data["output"]["run_name"] = "integration"
        path = joinpath(directory, "config.toml")
        open(path, "w") do io
            TOML.print(io, data; sorted=true)
            println(io)
        end

        run_dir = run_simulation(path)
        required = [
            "config.input.toml", "config.effective.toml", "metadata.toml",
            "data_dictionary.md", "run.log", "background.csv", "energies.csv",
            "slowroll.csv", "diagnostics.csv", "field_spectra.csv",
            "linear_curvature_spectra.csv", "deltaN_summary.csv",
            "deltaN_spectrum.csv", "checkpoint.jld2",
        ]
        @test all(name -> isfile(joinpath(run_dir, name)), required)
        @test first(readlines(joinpath(run_dir, "background.csv"))) ==
              "output_id,step,tau,t,efolds,a,H,Hdot,rho,p,w,phi_mean,phi_velocity,phi_variance,psi_mean,psi_velocity,psi_variance"
        metadata = TOML.parsefile(joinpath(run_dir, "metadata.toml"))
        @test metadata["run"]["end_reason"] == "max_steps"
        @test metadata["software"]["julia_version"] == string(VERSION)

        expected_fft_threads = Int(data["parallel"]["fft_threads"])
        MFIL.FFTW.set_num_threads(expected_fft_threads == 1 ? 2 : 1)
        @test resume_simulation(joinpath(run_dir, "checkpoint.jld2")) == run_dir
        @test MFIL.FFTW.get_num_threads() == expected_fft_threads
        MFIL.FFTW.set_num_threads(expected_fft_threads == 1 ? 2 : 1)
        analysis = analyze_run(run_dir)
        @test MFIL.FFTW.get_num_threads() == expected_fft_threads
        @test analysis.state.step == 1
        @test isfinite(analysis.background.H)
        @test isfinite(analysis.energy.rho)

        checkpoint_payload = load_checkpoint(joinpath(run_dir, "checkpoint.jld2"))
        code_energy = energy_summary(
            checkpoint_payload["state"], checkpoint_payload["model"],
            checkpoint_payload["lattice"],
        )
        diagnostic_lines = readlines(joinpath(run_dir, "diagnostics.csv"))
        diagnostic_header = split(first(diagnostic_lines), ',')
        residual_column = findfirst(==("friedmann_residual"), diagnostic_header)
        diagnostic_values = split(last(diagnostic_lines), ',')
        @test parse(Float64, diagnostic_values[residual_column]) ==
              MFIL.friedmann_residual(checkpoint_payload["state"], code_energy.rho)

        # A completed run must still pass lineage and every CSV-tail check.
        copied_run = joinpath(directory, "copied-run")
        cp(run_dir, copied_run)
        @test_throws MFIL.OutputError resume_simulation(
            joinpath(copied_run, "checkpoint.jld2"),
        )

        # Missing committed data in a completed run is not hidden by the
        # metadata status shortcut.
        energies_path = joinpath(run_dir, "energies.csv")
        energy_lines = readlines(energies_path; keep=true)
        open(energies_path, "w") do io
            write(io, join(energy_lines[1:end-1]))
        end
        @test_throws MFIL.OutputError resume_simulation(
            joinpath(run_dir, "checkpoint.jld2"),
        )
    end
end

@testset "interrupted multi-CSV observation is replayed exactly once" begin
    mktempdir() do directory
        config = test_config(
            vacuum=false,
            integrator="rk4",
            max_steps=2,
            spectra=false,
            deltaN=false,
            output_root=joinpath(directory, "runs"),
        )
        data = effective_config_dict(config)
        data["output"]["run_name"] = "fault_injection"
        path = joinpath(directory, "config.toml")
        open(path, "w") do io
            TOML.print(io, data; sorted=true)
            println(io)
        end

        MFIL._OUTPUT_FAULT_HOOK[] = function (event, kind, transaction_id, part_index, key)
            if event == :after_part && kind == :observation &&
               transaction_id == 1 && part_index == 2
                throw(InterruptException())
            end
        end
        try
            @test_throws InterruptException run_simulation(path)
        finally
            MFIL._OUTPUT_FAULT_HOOK[] = nothing
        end

        run_dirs = filter(isdir, readdir(data["output"]["root"]; join=true))
        @test length(run_dirs) == 1
        run_dir = only(run_dirs)
        @test isdir(joinpath(run_dir, ".output-transaction"))
        @test isfile(joinpath(run_dir, "checkpoint.jld2"))

        # Crash again after pending recovery has removed the journal but before
        # Runner can save the refreshed checkpoint.  The committed receipt must
        # make the following resume equivalent and idempotent.
        MFIL._OUTPUT_FAULT_HOOK[] = function (event, kind, transaction_id, part_index, key)
            if event == :after_restore && kind == :observation && transaction_id == 1
                throw(InterruptException())
            end
        end
        try
            @test_throws InterruptException resume_simulation(
                joinpath(run_dir, "checkpoint.jld2"),
            )
        finally
            MFIL._OUTPUT_FAULT_HOOK[] = nothing
        end
        @test !ispath(joinpath(run_dir, ".output-transaction"))
        @test isfile(joinpath(run_dir, ".output-transaction.last.toml"))

        @test resume_simulation(joinpath(run_dir, "checkpoint.jld2")) == run_dir
        @test !ispath(joinpath(run_dir, ".output-transaction"))
        for filename in ("background.csv", "energies.csv", "slowroll.csv", "diagnostics.csv")
            lines = readlines(joinpath(run_dir, filename))
            header = split(first(lines), ',')
            output_column = findfirst(==("output_id"), header)
            ids = [parse(Int, split(line, ',')[output_column]) for line in lines[2:end]]
            @test ids == [0, 1, 2]
            @test length(ids) == length(unique(ids))
        end
        for histogram in filter(
            path -> endswith(path, ".csv"),
            readdir(joinpath(run_dir, "histograms"); join=true),
        )
            lines = readlines(histogram)
            header = split(first(lines), ',')
            output_column = findfirst(==("output_id"), header)
            ids = [parse(Int, split(line, ',')[output_column]) for line in lines[2:end]]
            @test sort(unique(ids)) == [0, 1, 2]
        end
    end
end
