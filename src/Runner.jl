"""Construct the configured fixed-step integrator and its reusable storage."""
function build_integrator(
    name::AbstractString,
    state::SimulationState;
    deterministic_reductions::Bool=false,
)
    normalized = lowercase(strip(String(name)))
    normalized == "rk4" && return RK4Integrator(state; deterministic_reductions)
    normalized == "leapfrog" && return LeapfrogIntegrator(state; deterministic_reductions)
    throw(ConfigError(["evolution.integrator must be leapfrog or rk4"]))
end

function _configuration_estimate(config::SimulationConfig)
    B = resolve_rescaling_B(config)
    lattice = build_lattice(config.lattice)
    nf = length(config.model.field_names)
    primary_bytes = nf * lattice.N^3 * sizeof(Float64) * 2
    # RK4 has ten reusable field-sized stage arrays; leapfrog has seven.
    workspace_factor = lowercase(config.evolution.integrator) == "rk4" ? 11 : 8
    estimated_bytes = primary_bytes + nf * lattice.N^3 * sizeof(Float64) * workspace_factor
    potential_code = sum(
        0.5 * (config.model.masses[i] / B)^2 * config.initial.field_means[i]^2
        for i in eachindex(config.model.masses)
    )
    kinetic_code = sum(
        0.5 * (config.initial.cosmic_velocities[i] / B)^2
        for i in eachindex(config.initial.cosmic_velocities)
    )
    hcode = sqrt((potential_code + kinetic_code) / 3)
    means = Float64.(config.initial.field_means)
    potential = build_potential(config, B)
    mass = characteristic_mass(potential, means)
    dt_cosmic = config.evolution.timestep_factor * min(
        hcode > 0 ? inv(hcode) : Inf,
        config.initial.scale_factor / lattice.kmax_lat,
        mass > 0 ? inv(mass) : Inf,
    )
    dtau = min(dt_cosmic / config.initial.scale_factor, lattice.dx / sqrt(3))

    # CSV sizes depend mostly on scheduled row counts.  Use a deliberately
    # conservative textual-row estimate (24 bytes/column including delimiter)
    # so validation can flag configurations where diagnostics may dominate
    # compute/storage.  This is an estimate, not a quota.
    row_bytes(columns) = 24.0 * columns + 1.0
    output_count(interval) = min(
        Float64(config.evolution.max_steps) + 1.0,
        floor(config.evolution.max_efolds / interval) + 2.0,
    )
    background_outputs = output_count(config.output.background_interval_efolds)
    spectra_outputs = config.spectra.enabled ?
        output_count(config.output.spectra_interval_efolds) : 0.0
    histogram_outputs = output_count(config.output.histogram_interval_efolds)
    shell_count = floor(Int, sqrt(3) * lattice.N / 2) + 1
    field_pairs = nf * (nf + 1) ÷ 2
    curvature_components = nf + nf * (nf - 1) ÷ 2 + 1
    tabular_bytes = background_outputs * (
        row_bytes(11 + 3nf) + row_bytes(7 + 2nf) +
        row_bytes(9 + 3nf) + row_bytes(8)
    )
    spectral_bytes = spectra_outputs * shell_count * (
        field_pairs * row_bytes(12) + curvature_components * row_bytes(13)
    )
    histogram_bytes = histogram_outputs * (nf + 1) *
        config.output.histogram_bins * row_bytes(7)
    slice_bytes = config.output.save_2d_slices ?
        histogram_outputs * nf * lattice.N^2 * row_bytes(5) : 0.0
    snapshot_bytes = config.output.save_3d_snapshots ?
        histogram_outputs * nf * lattice.N^3 * row_bytes(7) : 0.0
    deltaN_bytes = config.deltaN.enabled ?
        row_bytes(12) + shell_count * row_bytes(7) +
        config.output.histogram_bins * row_bytes(7) +
        (config.output.save_2d_slices ? lattice.N^2 * row_bytes(5) : 0.0) +
        (config.output.save_3d_snapshots ? lattice.N^3 * row_bytes(7) : 0.0) : 0.0
    # Include two generations of the checkpoint plus metadata/log overhead.
    checkpoint_bytes = 2.0 * estimated_bytes
    estimated_output_bytes = tabular_bytes + spectral_bytes + histogram_bytes +
        slice_bytes + snapshot_bytes + deltaN_bytes + checkpoint_bytes + 1024.0^2
    return (;
        B,
        lattice,
        estimated_bytes,
        H_initial_approx=B * hcode,
        kmin_over_aH=lattice.kmin_lat / (config.initial.scale_factor * hcode),
        kmax_over_aH=lattice.kmax_lat / (config.initial.scale_factor * hcode),
        dtau_code=dtau,
        cfl_ratio=sqrt(3) * dtau / lattice.dx,
        estimated_output_bytes,
        io_dominant=config.output.save_3d_snapshots ||
            estimated_output_bytes > 4.0 * estimated_bytes,
    )
end

"""
Load and fully validate a TOML file without creating output.  A concise resource
and scale estimate is printed for CLI use, and the typed effective
configuration is returned.
"""
function validate_config_file(path::AbstractString; io::IO=stdout)
    config = load_config(path)
    validate_config(config)
    estimate = _configuration_estimate(config)
    println(io, "configuration valid")
    @printf(io, "  lattice: %d^3, fields: %d, estimated state/work memory: %.2f MiB\n",
        estimate.lattice.N, length(config.model.field_names),
        estimate.estimated_bytes / 1024^2)
    @printf(io, "  B: %.8e, initial H (background estimate): %.8e\n",
        estimate.B, estimate.H_initial_approx)
    @printf(io, "  k_min/(aH): %.6g, k_max/(aH): %.6g, CFL ratio: %.6g\n",
        estimate.kmin_over_aH, estimate.kmax_over_aH, estimate.cfl_ratio)
    @printf(io, "  estimated output capacity: %.2f MiB\n",
        estimate.estimated_output_bytes / 1024^2)
    estimate.io_dominant && println(io,
        "  warning: configured output volume may dominate runtime or storage")
    return config
end

function _checkpoint_payload(config, state, model, lattice, integrator, rng,
                             runctx, last_observation_step;
                             phase::AbstractString="main",
                             deltaN_progress=nothing,
                             deltaN_result=nothing,
                             deltaN_outputs_written::Bool=false)
    resumable_ids = Dict{Symbol,Int}()
    for key in keys(_RESUMABLE_CSV_NAMES)
        resumable_ids[key] = get(runctx.last_output_ids, key, -1)
    end
    return Dict{String,Any}(
        "config" => config,
        "state" => state,
        "model" => model,
        "lattice" => lattice,
        "integrator" => integrator,
        "rng" => rng,
        "runctx" => runctx,
        "run_dir" => runctx.run_dir,
        "run_id" => runctx.run_id,
        "output_id" => runctx.output_id,
        "last_output_ids" => resumable_ids,
        "csv_integrity" => checkpoint_csv_integrity(runctx),
        "last_observation_step" => last_observation_step,
        "effective_config_sha256" => runctx.effective_hash,
        "phase" => String(phase),
        "deltaN_progress" => deltaN_progress,
        "deltaN_result" => deltaN_result,
        "deltaN_outputs_written" => deltaN_outputs_written,
    )
end

function _save_run_checkpoint(config, state, model, lattice, integrator, rng,
                              runctx, last_observation_step;
                              phase::AbstractString="main",
                              deltaN_progress=nothing,
                              deltaN_result=nothing,
                              deltaN_outputs_written::Bool=false)
    payload = _checkpoint_payload(
        config, state, model, lattice, integrator, rng, runctx, last_observation_step;
        phase=phase,
        deltaN_progress=deltaN_progress,
        deltaN_result=deltaN_result,
        deltaN_outputs_written=deltaN_outputs_written,
    )
    save_checkpoint(runctx.files[:checkpoint], payload)
    write_run_log!(runctx, "INFO", "checkpoint committed at step $(state.step), phase=$(phase)")
    return payload
end

function _progress_line(state, model, epsilon_H, dtau, residual)
    return @sprintf(
        "step=%d N=%.8f a=%.8e H=%.8e epsilon_H=%.8e dtau=%.4e C_F=%.3e",
        state.step, state.efolds, state.a, model.B * hubble_code(state),
        epsilon_H, dtau, residual,
    )
end

function _run_loop!(config, state, model, lattice, integrator, rng, runctx;
                    last_observation_step::Int=-1,
                    resume_phase::AbstractString="main",
                    deltaN_progress=nothing,
                    deltaN_result=nothing,
                    deltaN_outputs_written::Bool=false)
    started_wall = time()
    cache = SimulationCache(
        state;
        deterministic_reductions=config.parallel.deterministic_reductions,
    )
    phase = String(resume_phase)
    phase in ("main", "deltaN", "deltaN_output", "completed") ||
        throw(OutputError("unsupported checkpoint phase $(repr(phase))"))
    reason = phase == "main" ? "unknown" : "epsilon_H_reached"
    warning_emitted = false

    try
        if phase == "main"
            if state.step != last_observation_step
                Base.disable_sigint() do
                    write_observation!(runctx, state, model, lattice, config, cache)
                    last_observation_step = state.step
                end
            end
        while true
            energy = energy_summary(state, model, lattice, cache)
            slowroll = slowroll_summary(
                state, model, lattice, energy, cache;
                min_speed2=config.spectra.linear_zeta_min_speed2,
            )
            if isfinite(slowroll.epsilon_H) &&
               slowroll.epsilon_H >= config.evolution.end_epsilon_H
                reason = "epsilon_H_reached"
                break
            elseif state.efolds >= config.evolution.max_efolds
                reason = "max_efolds"
                break
            elseif state.step >= config.evolution.max_steps
                reason = "max_steps"
                break
            elseif config.evolution.max_wall_time > 0 &&
                   time() - started_wall >= config.evolution.max_wall_time
                reason = "max_wall_time"
                break
            end

            dtau = choose_dtau(state, model, lattice, config)
            step!(integrator, state, model, lattice, dtau, cache)
            finite_state(state) || throw(NumericalError("non-finite state after step $(state.step)"))

            energy = energy_summary(state, model, lattice, cache)
            residual = friedmann_residual(state, energy.rho)
            if residual > config.evolution.friedmann_error_tolerance
                throw(NumericalError(@sprintf(
                    "Friedmann residual %.6e exceeds error tolerance %.6e at step %d",
                    residual, config.evolution.friedmann_error_tolerance, state.step,
                )))
            elseif residual > config.evolution.friedmann_warning_tolerance && !warning_emitted
                record_warning!(runctx, @sprintf(
                    "Friedmann residual %.6e exceeded warning tolerance %.6e",
                    residual, config.evolution.friedmann_warning_tolerance,
                ))
                warning_emitted = true
            end

            tolerance = 16eps(Float64) * max(abs(state.efolds), 1.0)
            if state.efolds + tolerance >= runctx.next_background_efolds
                Base.disable_sigint() do
                    write_observation!(runctx, state, model, lattice, config, cache)
                    last_observation_step = state.step
                end
                slowroll = slowroll_summary(
                    state, model, lattice, energy, cache;
                    min_speed2=config.spectra.linear_zeta_min_speed2,
                )
                line = _progress_line(state, model, slowroll.epsilon_H, dtau, residual)
                println(line)
                write_run_log!(runctx, "INFO", line)
            end

            if state.efolds + tolerance >= runctx.next_checkpoint_efolds
                _save_run_checkpoint(
                    config, state, model, lattice, integrator, rng,
                    runctx, last_observation_step;
                    phase=phase,
                    deltaN_progress=deltaN_progress,
                    deltaN_result=deltaN_result,
                    deltaN_outputs_written=deltaN_outputs_written,
                )
                while runctx.next_checkpoint_efolds <= state.efolds + tolerance
                    runctx.next_checkpoint_efolds += config.output.checkpoint_interval_efolds
                end
            end
        end

        if state.step != last_observation_step
            Base.disable_sigint() do
                write_observation!(runctx, state, model, lattice, config, cache)
                last_observation_step = state.step
            end
        end
        _save_run_checkpoint(
            config, state, model, lattice, integrator, rng, runctx, last_observation_step,
        )
        end

        if reason == "epsilon_H_reached" && config.deltaN.enabled
            if phase in ("main", "deltaN")
                if deltaN_progress === nothing
                    deltaN_progress = initialize_deltaN_progress(
                        state, model, lattice, config,
                    )
                end
                phase = "deltaN"
                _save_run_checkpoint(
                    config, state, model, lattice, integrator, rng,
                    runctx, last_observation_step;
                    phase=phase,
                    deltaN_progress=deltaN_progress,
                )
                checkpoint_sweeps = max(
                    1,
                    ceil(Int, config.output.checkpoint_interval_efolds / config.deltaN.dN),
                )
                checkpoint_callback = progress -> _save_run_checkpoint(
                    config, state, model, lattice, integrator, rng,
                    runctx, last_observation_step;
                    phase="deltaN",
                    deltaN_progress=progress,
                )
                deltaN_result = compute_deltaN(
                    state,
                    model,
                    lattice,
                    config;
                    progress=deltaN_progress,
                    checkpoint_callback=checkpoint_callback,
                    checkpoint_every_sweeps=checkpoint_sweeps,
                )
                phase = "deltaN_output"
                _save_run_checkpoint(
                    config, state, model, lattice, integrator, rng,
                    runctx, last_observation_step;
                    phase=phase,
                    deltaN_progress=deltaN_progress,
                    deltaN_result=deltaN_result,
                    deltaN_outputs_written=false,
                )
            end
            deltaN_result === nothing && throw(OutputError(
                "delta-N output phase has no checkpointed result",
            ))
            # The output layer commits all delta-N artifacts as one durable
            # transaction.  Always replay/verify its receipt in this phase,
            # even when recovery set the flag: besides proving the files are
            # exact, the writer reconstructs metadata that may post-date the
            # stale serialized RunContext.  Matching receipts are a no-op for
            # CSV bytes and mismatches fail closed.
            write_deltaN_outputs!(runctx, deltaN_result, state, model, lattice, config)
            deltaN_outputs_written = true
            _save_run_checkpoint(
                config, state, model, lattice, integrator, rng,
                runctx, last_observation_step;
                phase=phase,
                deltaN_progress=deltaN_progress,
                deltaN_result=deltaN_result,
                deltaN_outputs_written=true,
            )
            if !isempty(deltaN_result.failed_indices)
                throw(NumericalError(
                    "delta-N failed to reach the reference density at " *
                    "$(length(deltaN_result.failed_indices)) lattice sites",
                ))
            end
        elseif config.deltaN.enabled
            record_warning!(runctx,
                "delta-N skipped because the main evolution stopped at $(reason), not epsilon_H")
        end
        finalize_metadata!(runctx, reason)
        phase = "completed"
        _save_run_checkpoint(
            config, state, model, lattice, integrator, rng, runctx, last_observation_step;
            phase=phase,
            deltaN_progress=deltaN_progress,
            deltaN_result=deltaN_result,
            deltaN_outputs_written=deltaN_outputs_written,
        )
        return runctx.run_dir
    catch err
        if err isa InterruptException
            try
                _save_run_checkpoint(
                    config, state, model, lattice, integrator, rng,
                    runctx, last_observation_step;
                    phase=phase,
                    deltaN_progress=deltaN_progress,
                    deltaN_result=deltaN_result,
                    deltaN_outputs_written=deltaN_outputs_written,
                )
                finalize_metadata!(runctx, "interrupted")
            catch checkpoint_error
                @error "failed to checkpoint interrupted run" exception=(checkpoint_error, catch_backtrace())
            end
            rethrow()
        end
        try
            _save_run_checkpoint(
                config, state, model, lattice, integrator, rng,
                runctx, last_observation_step;
                phase=phase,
                deltaN_progress=deltaN_progress,
                deltaN_result=deltaN_result,
                deltaN_outputs_written=deltaN_outputs_written,
            )
            finalize_metadata!(runctx, "error"; warnings=[sprint(showerror, err)])
        catch checkpoint_error
            @error "failed to save diagnostic checkpoint" exception=(checkpoint_error, catch_backtrace())
        end
        rethrow()
    end
end

"""Run a new simulation from a validated TOML configuration path."""
function run_simulation(config_path::AbstractString)
    input_path = abspath(String(config_path))
    config = load_config(input_path)
    validate_config(config)
    initialized = initialize_simulation(config)
    runctx = create_run_directory(config, input_path)
    initialize_outputs!(
        runctx, config, input_path, initialized.model, initialized.lattice,
    )
    diagnostics = initialized.initial_diagnostics
    runctx.metadata["initial_diagnostics"] = _io_plain(diagnostics)
    write_metadata!(runctx)
    write_run_log!(runctx, "INFO", @sprintf(
        "initial H=%.8e, kmin/(aH)=%.6g, kmax/(aH)=%.6g",
        diagnostics.H, diagnostics.kmin_over_aH, diagnostics.kmax_over_aH,
    ))
    integrator = build_integrator(
        config.evolution.integrator,
        initialized.state;
        deterministic_reductions=config.parallel.deterministic_reductions,
    )
    return _run_loop!(
        config, initialized.state, initialized.model, initialized.lattice,
        integrator, initialized.rng, runctx,
    )
end

function _payload_value(payload, key::AbstractString)
    payload isa AbstractDict && haskey(payload, key) && return payload[key]
    payload isa AbstractDict && haskey(payload, Symbol(key)) && return payload[Symbol(key)]
    hasproperty(payload, Symbol(key)) && return getproperty(payload, Symbol(key))
    throw(OutputError("checkpoint payload is missing $(key)"))
end

function _payload_optional(payload, key::AbstractString, default=nothing)
    payload isa AbstractDict && haskey(payload, key) && return payload[key]
    payload isa AbstractDict && haskey(payload, Symbol(key)) && return payload[Symbol(key)]
    hasproperty(payload, Symbol(key)) && return getproperty(payload, Symbol(key))
    return default
end

"""Resume a simulation only after all committed CSV output IDs are verified."""
function resume_simulation(checkpoint_path::AbstractString)
    checkpoint = abspath(String(checkpoint_path))
    payload = load_checkpoint(checkpoint)
    run_dir = dirname(checkpoint)
    config = _payload_value(payload, "config")
    validate_config(config)
    FFTW.set_num_threads(config.parallel.fft_threads)
    state = _payload_value(payload, "state")
    model = _payload_value(payload, "model")
    lattice = _payload_value(payload, "lattice")
    integrator = _payload_value(payload, "integrator")
    rng = _payload_value(payload, "rng")
    runctx = _payload_value(payload, "runctx")
    metadata = verify_checkpoint_lineage(checkpoint, payload, runctx)
    restore_output_progress!(runctx, payload)
    saved_phase = _payload_optional(payload, "phase", nothing)
    phase = saved_phase === nothing ? "main" : String(saved_phase)
    run_metadata = get(metadata, "run", Dict{String,Any}())
    # A hard stop may occur after metadata was finalized but before the
    # terminal checkpoint replaced a delta-N output-phase checkpoint. Continue
    # that one idempotent phase; every other completed run returns only after
    # lineage, transaction, output-ID, and whole-CSV integrity verification.
    metadata_completed = get(run_metadata, "status", "") == "completed"
    if phase == "completed" || (metadata_completed && phase != "deltaN_output")
        return run_dir
    end
    hasproperty(integrator, :dynamics) || throw(OutputError(
        "checkpointed integrator has no dynamics workspace",
    ))
    integrator.dynamics.deterministic_reductions =
        config.parallel.deterministic_reductions
    runctx.finalized = false
    runctx.metadata["run"]["status"] = "running"
    write_metadata!(runctx)
    last_step = csv_last_integer(runctx.files[:background], "step")
    last_step < 0 && (last_step = Int(_payload_value(payload, "last_observation_step")))
    deltaN_progress = _payload_optional(payload, "deltaN_progress", nothing)
    deltaN_result = _payload_optional(payload, "deltaN_result", nothing)
    deltaN_outputs_written = Bool(_payload_optional(
        payload, "deltaN_outputs_written", false,
    ))
    return _run_loop!(
        config, state, model, lattice, integrator, rng, runctx;
        last_observation_step=last_step,
        resume_phase=phase,
        deltaN_progress=deltaN_progress,
        deltaN_result=deltaN_result,
        deltaN_outputs_written=deltaN_outputs_written,
    )
end

"""
Recompute final diagnostics from a saved checkpoint.  The function is
read-only with respect to the original CSV stream and returns analysis-ready
Julia values to callers.
"""
function analyze_run(input_dir::AbstractString)
    run_dir = abspath(String(input_dir))
    checkpoint = isdir(run_dir) ? joinpath(run_dir, "checkpoint.jld2") : run_dir
    payload = load_checkpoint(checkpoint)
    config = _payload_value(payload, "config")
    validate_config(config)
    FFTW.set_num_threads(config.parallel.fft_threads)
    state = _payload_value(payload, "state")
    model = _payload_value(payload, "model")
    lattice = _payload_value(payload, "lattice")
    cache = SimulationCache(
        state;
        deterministic_reductions=config.parallel.deterministic_reductions,
    )
    energy = energy_summary(state, model, lattice, cache)
    slowroll = slowroll_summary(
        state, model, lattice, energy, cache;
        min_speed2=config.spectra.linear_zeta_min_speed2,
    )
    spectra = config.spectra.enabled ?
        field_spectra(state, model, lattice, config, 0) : NamedTuple[]
    curvature = config.spectra.enabled ?
        linear_curvature_spectra(state, model, lattice, config, 0) : NamedTuple[]
    return (;
        state,
        background=background_summary(state, model, lattice, energy),
        energy=physical_energy_summary(energy, model),
        slowroll,
        field_spectra=spectra,
        linear_curvature_spectra=curvature,
    )
end
