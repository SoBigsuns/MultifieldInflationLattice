import Dates
import SHA
import TOML

const MFIL_OUTPUT_SCHEMA_VERSION = "MFIL-CSV-1"
const MFIL_CHECKPOINT_SCHEMA_VERSION = "MFIL-JLD2-1"

"""An I/O failure that should map to CLI exit code 3."""
struct OutputError <: Exception
    message::String
end

Base.showerror(io::IO, err::OutputError) = print(io, err.message)

"""
Mutable bookkeeping for one run directory.

No file handles are retained: every committed CSV row is flushed and closed before
the corresponding `last_output_ids` entry is advanced.  This makes checkpoint
bookkeeping unambiguous and keeps a normally interrupted run readable.
"""
mutable struct RunContext
    run_dir::String
    run_id::String
    started_at::Dates.DateTime
    output_id::Int
    warnings::Vector{String}
    files::Dict{Symbol,String}
    headers::Dict{Symbol,Vector{String}}
    last_output_ids::Dict{Symbol,Int}
    next_background_efolds::Float64
    next_spectra_efolds::Float64
    next_histogram_efolds::Float64
    next_checkpoint_efolds::Float64
    metadata::Dict{String,Any}
    input_hash::String
    effective_hash::String
    finalized::Bool
end

function RunContext(run_dir::AbstractString, run_id::AbstractString,
                    started_at::Dates.DateTime, input_hash::AbstractString)
    return RunContext(
        abspath(String(run_dir)), String(run_id), started_at, -1, String[],
        Dict{Symbol,String}(), Dict{Symbol,Vector{String}}(), Dict{Symbol,Int}(),
        0.0, 0.0, 0.0, 0.0, Dict{String,Any}(), String(input_hash), "", false,
    )
end

"""Return a property or dictionary entry without constraining configuration types."""
function _io_get(value, key::Symbol, default=nothing)
    if value isa AbstractDict
        haskey(value, key) && return value[key]
        skey = String(key)
        haskey(value, skey) && return value[skey]
    elseif value !== nothing && hasproperty(value, key)
        return getproperty(value, key)
    end
    return default
end

function _io_get_any(value, keys, default=nothing)
    for key in keys
        result = _io_get(value, key, nothing)
        result === nothing || return result
    end
    return default
end

function _io_cfg(config, section::Symbol, key::Symbol, default=nothing)
    return _io_get(_io_get(config, section, nothing), key, default)
end

function _io_plain(value)
    if value === nothing
        return ""
    elseif value isa Symbol || value isa Dates.TimeType
        return string(value)
    elseif value isa Unsigned
        value <= UInt64(typemax(Int64)) || return string(value)
        return Int64(value)
    elseif value isa AbstractString || value isa Number || value isa Bool
        return value
    elseif value isa AbstractDict
        result = Dict{String,Any}()
        for (key, item) in value
            result[string(key)] = _io_plain(item)
        end
        return result
    elseif value isa NamedTuple
        return Dict(string(key) => _io_plain(getfield(value, key)) for key in keys(value))
    elseif value isa AbstractVector{<:Unsigned}
        # TOML integers are signed Int64; keep derived UInt64 seed vectors homogeneous.
        return [string(item) for item in value]
    elseif value isa Tuple || value isa AbstractVector
        return [_io_plain(item) for item in value]
    elseif isstructtype(typeof(value))
        result = Dict{String,Any}()
        for key in fieldnames(typeof(value))
            result[string(key)] = _io_plain(getfield(value, key))
        end
        return result
    end
    return string(value)
end

function _io_sha256_bytes(bytes)
    return bytes2hex(SHA.sha256(bytes))
end

function _io_sha256_file(path::AbstractString)
    isfile(path) || throw(OutputError("cannot hash missing file: $(abspath(path))"))
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _io_atomic_write(writer::Function, path::AbstractString)
    target = abspath(String(path))
    mkpath(dirname(target))
    temporary = target * ".tmp"
    ispath(temporary) && rm(temporary; force=true, recursive=isdir(temporary))
    try
        open(temporary, "w") do io
            writer(io)
            flush(io)
        end
        mv(temporary, target; force=true)
    catch err
        ispath(temporary) && rm(temporary; force=true, recursive=isdir(temporary))
        err isa InterruptException && rethrow()
        err isa OutputError && rethrow()
        throw(OutputError("failed to write $(target): $(sprint(showerror, err))"))
    end
    return target
end

function _io_write_toml(path::AbstractString, value)
    plain = _io_plain(value)
    plain isa AbstractDict || throw(OutputError("TOML root must be a table"))
    return _io_atomic_write(path) do io
        TOML.print(io, plain; sorted=true)
        write(io, "\n")
    end
end

function _io_config_bytes(config)
    return Vector{UInt8}(codeunits(sprint(io -> TOML.print(io, _io_plain(config); sorted=true))))
end

function _io_safe_identifier(value; fallback::AbstractString="run")
    text = String(value)
    text = replace(text, r"[^A-Za-z0-9_]+" => "_")
    text = replace(text, r"^_+|_+$" => "")
    isempty(text) && (text = String(fallback))
    occursin(r"^[0-9]", text) && (text = "_" * text)
    return text
end

"""
Create a unique run directory named `<run_name>_<UTC timestamp>_<hash>`.

Existing results are never modified unless `output.overwrite=true`.  Validation
belongs before this call so an invalid configuration cannot create a run folder.
"""
function create_run_directory(config, input_path::AbstractString)
    input_hash = if isfile(input_path)
        _io_sha256_file(input_path)
    else
        _io_sha256_bytes(_io_config_bytes(config))
    end
    run_name = _io_safe_identifier(_io_cfg(config, :output, :run_name, "run"))
    timestamp = Dates.format(Dates.now(Dates.UTC), Dates.DateFormat("yyyymmddTHHMMSSZ"))
    run_id = "$(run_name)_$(timestamp)_$(first(input_hash, 12))"
    root = abspath(expanduser(String(_io_cfg(config, :output, :root, "runs"))))
    run_dir = joinpath(root, run_id)
    overwrite = Bool(_io_cfg(config, :output, :overwrite, false))

    mkpath(root)
    if ispath(run_dir)
        overwrite || throw(OutputError(
            "run directory already exists and output.overwrite=false: $(run_dir)",
        ))
        isdir(run_dir) || throw(OutputError("run output path is not a directory: $(run_dir)"))
        rm(run_dir; recursive=true, force=true)
    end
    try
        mkdir(run_dir)
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError("failed to create exclusive run directory $(run_dir): $(sprint(showerror, err))"))
    end
    return RunContext(run_dir, run_id, Dates.now(Dates.UTC), input_hash)
end

function _io_copy_input_config(input_path::AbstractString, destination::AbstractString)
    isfile(input_path) || throw(OutputError("input configuration not found: $(abspath(input_path))"))
    source = abspath(String(input_path))
    target = abspath(String(destination))
    source == target && return target
    try
        cp(source, target; force=false)
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError("failed to preserve input configuration: $(sprint(showerror, err))"))
    end
    return target
end

function _io_command_output(command::Cmd)
    try
        return strip(read(pipeline(command; stderr=devnull), String))
    catch err
        err isa InterruptException && rethrow()
        return "unknown"
    end
end

function _io_dependency_versions(project_root::AbstractString)
    result = Dict{String,Any}()
    manifest_path = joinpath(project_root, "Manifest.toml")
    isfile(manifest_path) || return Dict{String,Any}("status" => "Manifest.toml unavailable")
    try
        manifest = TOML.parsefile(manifest_path)
        dependencies = get(manifest, "deps", Dict{String,Any}())
        for (name, entries) in dependencies
            entries isa AbstractVector && isempty(entries) && continue
            entry = entries isa AbstractVector ? first(entries) : entries
            entry isa AbstractDict || continue
            version = get(entry, "version", nothing)
            version === nothing || (result[string(name)] = string(version))
        end
    catch err
        err isa InterruptException && rethrow()
        result["status"] = "unavailable"
    end
    return result
end

function _io_cpu_name()
    try
        info = Sys.cpu_info()
        isempty(info) || return string(first(info).model)
    catch err
        err isa InterruptException && rethrow()
    end
    return string(Sys.ARCH)
end

function _io_git_metadata(project_root::AbstractString)
    commit = _io_command_output(`git -C $project_root rev-parse HEAD`)
    status = _io_command_output(`git -C $project_root status --porcelain`)
    return Dict{String,Any}(
        "commit" => commit,
        "dirty" => status != "unknown" && !isempty(status),
    )
end

"""Append a warning to both the run context and `run.log`."""
function record_warning!(runctx::RunContext, warning::AbstractString)
    message = String(warning)
    is_new = !(message in runctx.warnings)
    is_new && push!(runctx.warnings, message)
    haskey(runctx.files, :log) && write_run_log!(runctx, "WARN", message)
    is_new && !isempty(runctx.metadata) && haskey(runctx.files, :metadata) && write_metadata!(runctx)
    return runctx
end

"""Append one UTC-stamped line to the run log."""
function write_run_log!(runctx::RunContext, level::AbstractString, message::AbstractString)
    path = get(runctx.files, :log, joinpath(runctx.run_dir, "run.log"))
    stamp = Dates.format(Dates.now(Dates.UTC), Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.sssZ"))
    try
        open(path, "a") do io
            println(io, stamp, " [", uppercase(String(level)), "] ", replace(String(message), '\n' => " "))
            flush(io)
        end
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError("failed to append run log $(path): $(sprint(showerror, err))"))
    end
    return path
end

function _io_manifest_hash(project_root::AbstractString)
    path = joinpath(project_root, "Manifest.toml")
    return isfile(path) ? _io_sha256_file(path) : "unavailable"
end

function _io_base_metadata(runctx::RunContext, config, model, lattice)
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    field_names = String.(collect(_io_cfg(config, :model, :field_names, String[])))
    size_value = _io_get_any(lattice, (:N, :size, :n), _io_cfg(config, :lattice, :size, 0))
    box_size = _io_get_any(lattice, (:box_size, :L, :length), _io_cfg(config, :lattice, :box_size, NaN))
    spacing = _io_get_any(lattice, (:dx, :spacing),
                          (size_value isa Number && size_value != 0) ? box_size / size_value : NaN)
    scale_B = _io_get_any(model, (:B, :rescale_B, :scale_B),
                          _io_cfg(config, :model, :rescale_B, "unknown"))
    return Dict{String,Any}(
        "schema_version" => MFIL_OUTPUT_SCHEMA_VERSION,
        "run" => Dict{String,Any}(
            "id" => runctx.run_id,
            "started_at_utc" => Dates.format(runctx.started_at, Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.sssZ")),
            "status" => "running",
            "end_reason" => "",
        ),
        "configuration" => Dict{String,Any}(
            "input_sha256" => runctx.input_hash,
            "effective_sha256" => runctx.effective_hash,
        ),
        "software" => merge(
            Dict{String,Any}(
                "julia_version" => string(VERSION),
                "manifest_sha256" => _io_manifest_hash(project_root),
                "dependencies" => _io_dependency_versions(project_root),
            ),
            _io_git_metadata(project_root),
        ),
        "platform" => Dict{String,Any}(
            "os" => string(Sys.KERNEL),
            "architecture" => string(Sys.ARCH),
            "cpu" => _io_cpu_name(),
            "julia_threads" => Threads.nthreads(),
            "fft_threads" => Int(_io_cfg(config, :parallel, :fft_threads, 1)),
        ),
        "random" => Dict{String,Any}(
            "rng_type" => "MersenneTwister with deterministic per-field seeds",
            "root_seed" => string(_io_cfg(config, :initial, :seed, 0)),
            "field_seed_derivation" => "SplitMix64(root xor golden-ratio*field-index)",
        ),
        "lattice" => Dict{String,Any}(
            "field_names" => field_names,
            "B" => _io_plain(scale_B),
            "N" => Int(size_value),
            "L" => Float64(box_size),
            "dx" => Float64(spacing),
            "L_units" => "code length (divide by B for reduced-Planck length)",
            "dx_units" => "code length (divide by B for reduced-Planck length)",
            "L_physical" => Float64(box_size) / Float64(scale_B),
            "dx_physical" => Float64(spacing) / Float64(scale_B),
            "boundary" => string(_io_cfg(config, :lattice, :boundary, "periodic")),
        ),
        "conventions" => Dict{String,Any}(
            "units" => "reduced Planck units; internal rescaling documented by B",
            "action" => "canonical_real_scalar_fields_flat_FLRW_v1",
            "fourier" => "forward_dx3_inverse_1_over_volume_v1",
            "spectrum" => "dimensionless_k_eff_cubed_over_2pi2V_v1",
            "curvature_sign" => "zeta_I=-H*phidotbar_I*delta_phi_I/sum_J(phidotbar_J^2)",
            "deltaN" => "zeta_deltaN=N_local-mean(N_local)",
        ),
        "reference" => Dict{String,Any}(
            "inflation_easy_commit" => "d4f0cfdde0d148fa8ffaa14fe37d705cd79b2366",
        ),
        "warnings" => String[],
    )
end

"""Persist the current metadata dictionary atomically."""
function write_metadata!(runctx::RunContext)
    runctx.metadata["warnings"] = copy(runctx.warnings)
    return _io_write_toml(get(runctx.files, :metadata,
                              joinpath(runctx.run_dir, "metadata.toml")), runctx.metadata)
end

"""Mark a run complete or stopped and atomically update `metadata.toml`."""
function finalize_metadata!(runctx::RunContext, reason; warnings=[])
    for warning in warnings
        record_warning!(runctx, string(warning))
    end
    ended_at = Dates.now(Dates.UTC)
    run_table = get!(runctx.metadata, "run", Dict{String,Any}())
    reason_text = string(reason)
    run_table["status"] = if reason_text in ("completed", "epsilon_H_reached")
        "completed"
    elseif reason_text == "error"
        "failed"
    elseif reason_text == "interrupted"
        "interrupted"
    else
        "stopped"
    end
    run_table["end_reason"] = reason_text
    run_table["ended_at_utc"] = Dates.format(ended_at, Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.sssZ"))
    run_table["elapsed_seconds"] = Dates.value(ended_at - runctx.started_at) / 1000
    runctx.finalized = true
    write_metadata!(runctx)
    write_run_log!(runctx, "INFO", "run finalized: $(reason)")
    return runctx
end
