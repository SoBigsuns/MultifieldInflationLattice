import Dates
import JLD2

"""
Atomically save a complete restart payload to a JLD2 checkpoint.

The new generation is written and reopened before it replaces `path`.  If a
checkpoint already exists, its previous generation is retained as `path.bak`.
The caller should include configuration hashes, the full simulation and RNG
state, integrator half-step state, and `last_output_ids` in `payload`.
"""
function save_checkpoint(path::AbstractString, payload)
    target = abspath(String(path))
    directory = dirname(target)
    mkpath(directory)
    temporary = target * ".tmp"
    backup = target * ".bak"
    backup_temporary = backup * ".tmp"

    ispath(temporary) && rm(temporary; force=true, recursive=isdir(temporary))
    ispath(backup_temporary) && rm(backup_temporary; force=true, recursive=isdir(backup_temporary))
    try
        JLD2.jldopen(temporary, "w") do file
            file["schema_version"] = MFIL_CHECKPOINT_SCHEMA_VERSION
            file["created_at_utc"] = Dates.format(
                Dates.now(Dates.UTC), Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.sssZ"),
            )
            file["payload"] = payload
        end

        # Reopen the temporary file before touching the committed generation.
        JLD2.jldopen(temporary, "r") do file
            haskey(file, "schema_version") || throw(OutputError(
                "temporary checkpoint has no schema_version: $(temporary)",
            ))
            haskey(file, "payload") || throw(OutputError(
                "temporary checkpoint has no payload: $(temporary)",
            ))
            file["payload"] # force a complete deserialize before replacement
        end

        if isfile(target)
            cp(target, backup_temporary; force=true)
            mv(backup_temporary, backup; force=true)
        end
        mv(temporary, target; force=true)
    catch err
        ispath(temporary) && rm(temporary; force=true, recursive=isdir(temporary))
        ispath(backup_temporary) && rm(backup_temporary; force=true, recursive=isdir(backup_temporary))
        if !isfile(target) && isfile(backup)
            try
                cp(backup, target; force=false)
            catch
                # Preserve the original exception; the retained .bak is still recoverable.
            end
        end
        err isa InterruptException && rethrow()
        err isa OutputError && rethrow()
        throw(OutputError("failed to save checkpoint $(target): $(sprint(showerror, err))"))
    end
    return target
end

"""Load and schema-check a JLD2 checkpoint, returning its saved payload."""
function load_checkpoint(path::AbstractString)
    target = abspath(String(path))
    isfile(target) || throw(OutputError("checkpoint not found: $(target)"))
    try
        return JLD2.jldopen(target, "r") do file
            haskey(file, "schema_version") || throw(OutputError(
                "checkpoint is missing schema_version: $(target)",
            ))
            schema = string(file["schema_version"])
            schema == MFIL_CHECKPOINT_SCHEMA_VERSION || throw(OutputError(
                "unsupported checkpoint schema $(repr(schema)); expected $(MFIL_CHECKPOINT_SCHEMA_VERSION)",
            ))
            haskey(file, "payload") || throw(OutputError(
                "checkpoint is missing payload: $(target)",
            ))
            file["payload"]
        end
    catch err
        err isa InterruptException && rethrow()
        err isa OutputError && rethrow()
        throw(OutputError("failed to load checkpoint $(target): $(sprint(showerror, err))"))
    end
end

function _checkpoint_same_path(left::AbstractString, right::AbstractString)
    a = normpath(abspath(String(left)))
    b = normpath(abspath(String(right)))
    return Sys.iswindows() ? lowercase(a) == lowercase(b) : a == b
end

function _checkpoint_required(payload, key::Symbol)
    value = _io_get(payload, key, nothing)
    value === nothing && throw(OutputError("checkpoint payload is missing $(key)"))
    return value
end

"""
Verify that a checkpoint belongs to the run directory in which it is found.

The effective-config digest and run ID must agree across the payload,
serialized RunContext, on-disk effective configuration, and metadata. This
deliberately rejects a checkpoint copied into a different run directory even
when its CSV output IDs happen to match.
"""
function verify_checkpoint_lineage(checkpoint_path::AbstractString, payload, runctx::RunContext)
    checkpoint = abspath(String(checkpoint_path))
    run_dir = dirname(checkpoint)
    saved_dir = String(_checkpoint_required(payload, :run_dir))
    _checkpoint_same_path(saved_dir, run_dir) || throw(OutputError(
        "checkpoint lineage mismatch: payload run_dir=$(saved_dir), actual run_dir=$(run_dir)",
    ))
    _checkpoint_same_path(runctx.run_dir, run_dir) || throw(OutputError(
        "checkpoint lineage mismatch: RunContext belongs to $(runctx.run_dir), not $(run_dir)",
    ))
    expected_checkpoint = get(runctx.files, :checkpoint, joinpath(runctx.run_dir, "checkpoint.jld2"))
    _checkpoint_same_path(expected_checkpoint, checkpoint) || throw(OutputError(
        "checkpoint lineage mismatch: RunContext checkpoint path is $(expected_checkpoint)",
    ))

    payload_run_id = String(_checkpoint_required(payload, :run_id))
    payload_run_id == runctx.run_id || throw(OutputError(
        "checkpoint lineage mismatch: payload run_id does not match RunContext",
    ))
    payload_hash = String(_checkpoint_required(payload, :effective_config_sha256))
    payload_hash == runctx.effective_hash || throw(OutputError(
        "checkpoint lineage mismatch: effective configuration hash differs from RunContext",
    ))

    effective_path = joinpath(run_dir, "config.effective.toml")
    isfile(effective_path) || throw(OutputError(
        "checkpoint lineage cannot be verified: missing $(effective_path)",
    ))
    disk_hash = _io_sha256_file(effective_path)
    disk_hash == payload_hash || throw(OutputError(
        "checkpoint lineage mismatch: config.effective.toml SHA-256 differs from checkpoint",
    ))

    metadata_path = joinpath(run_dir, "metadata.toml")
    isfile(metadata_path) || throw(OutputError(
        "checkpoint lineage cannot be verified: missing $(metadata_path)",
    ))
    metadata = try
        TOML.parsefile(metadata_path)
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError("cannot parse metadata for checkpoint lineage: $(sprint(showerror, err))"))
    end
    metadata_run = get(metadata, "run", Dict{String,Any}())
    get(metadata_run, "id", nothing) == payload_run_id || throw(OutputError(
        "checkpoint lineage mismatch: metadata run ID differs from checkpoint",
    ))
    metadata_config = get(metadata, "configuration", Dict{String,Any}())
    get(metadata_config, "effective_sha256", nothing) == payload_hash || throw(OutputError(
        "checkpoint lineage mismatch: metadata effective-config SHA-256 differs from checkpoint",
    ))
    return metadata
end

function _checkpoint_csv_fields(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if quoted
            if character == '"'
                nextindex_value = nextind(line, index)
                if nextindex_value <= lastindex(line) && line[nextindex_value] == '"'
                    print(buffer, '"')
                    index = nextindex_value
                else
                    quoted = false
                end
            else
                print(buffer, character)
            end
        elseif character == '"'
            quoted = true
        elseif character == ','
            push!(fields, String(take!(buffer)))
        else
            print(buffer, character)
        end
        index = nextind(line, index)
    end
    quoted && throw(OutputError("unterminated quoted CSV field"))
    push!(fields, String(take!(buffer)))
    return fields
end

"""Return the final committed `output_id` in a CSV, or `-1` for a header-only file."""
function csv_last_output_id(path::AbstractString; column::AbstractString="output_id")
    target = abspath(String(path))
    isfile(target) || throw(OutputError("CSV output is missing: $(target)"))
    header = String[]
    last_record = nothing
    try
        open(target, "r") do io
            eof(io) && throw(OutputError("CSV output is empty: $(target)"))
            header = _checkpoint_csv_fields(chomp(readline(io)))
            for line in eachline(io)
                isempty(strip(line)) || (last_record = line)
            end
        end
    catch err
        err isa InterruptException && rethrow()
        err isa OutputError && rethrow()
        throw(OutputError("failed to inspect CSV $(target): $(sprint(showerror, err))"))
    end
    position = findfirst(==(String(column)), header)
    position === nothing && throw(OutputError(
        "CSV $(target) has no $(repr(column)) column",
    ))
    last_record === nothing && return -1
    values = _checkpoint_csv_fields(last_record)
    length(values) == length(header) || throw(OutputError(
        "CSV $(target) ends with a partial row",
    ))
    try
        return parse(Int, values[position])
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError(
            "invalid final output_id in $(target): $(sprint(showerror, err))",
        ))
    end
end

"""Read an integer column from the final complete CSV row, or `-1` if header-only."""
function csv_last_integer(path::AbstractString, column::AbstractString)
    target = abspath(String(path))
    isfile(target) || throw(OutputError("CSV output is missing: $(target)"))
    header = String[]
    last_record = nothing
    open(target, "r") do io
        eof(io) && throw(OutputError("CSV output is empty: $(target)"))
        header = _checkpoint_csv_fields(chomp(readline(io)))
        for line in eachline(io)
            isempty(strip(line)) || (last_record = line)
        end
    end
    position = findfirst(==(String(column)), header)
    position === nothing && throw(OutputError(
        "CSV $(target) has no $(repr(column)) column",
    ))
    last_record === nothing && return -1
    values = _checkpoint_csv_fields(last_record)
    length(values) == length(header) || throw(OutputError(
        "CSV $(target) ends with a partial row",
    ))
    try
        return parse(Int, values[position])
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError(
            "invalid final $(column) in $(target): $(sprint(showerror, err))",
        ))
    end
end

function _checkpoint_progress(payload)
    progress = _io_get_any(payload, (:last_output_ids, :output_ids, :csv_output_ids), nothing)
    context = _io_get_any(payload, (:runctx, :run_context, :output_progress), nothing)
    if progress === nothing
        progress = _io_get_any(context, (:last_output_ids, :output_ids, :csv_output_ids), nothing)
    end
    if progress === nothing
        scalar = _io_get(payload, :output_id, nothing)
        scalar === nothing || return Dict{Symbol,Int}(:all => Int(scalar))
        throw(OutputError("checkpoint payload has no CSV output progress"))
    end

    result = Dict{Symbol,Int}()
    if progress isa AbstractDict || progress isa NamedTuple
        pairs_iter = progress isa NamedTuple ? pairs(progress) : pairs(progress)
        for (key, value) in pairs_iter
            normalized = replace(lowercase(string(key)), ".csv" => "")
            result[Symbol(normalized)] = Int(value)
        end
    else
        scalar = _io_get(progress, :output_id, nothing)
        scalar === nothing && throw(OutputError("unrecognized checkpoint CSV progress value"))
        result[:all] = Int(scalar)
    end
    # The top-level checkpoint table contains the fixed resumable streams.  The
    # serialized RunContext additionally retains dynamically named histograms.
    nested = _io_get_any(context, (:last_output_ids, :output_ids, :csv_output_ids), nothing)
    if nested isa AbstractDict || nested isa NamedTuple
        for (key, value) in pairs(nested)
            normalized = Symbol(replace(lowercase(string(key)), ".csv" => ""))
            get!(result, normalized, Int(value))
        end
    end
    return result
end

const _RESUMABLE_CSV_NAMES = Dict{Symbol,String}(
    :background => "background.csv",
    :energies => "energies.csv",
    :slowroll => "slowroll.csv",
    :diagnostics => "diagnostics.csv",
    :field_spectra => "field_spectra.csv",
    :linear_curvature_spectra => "linear_curvature_spectra.csv",
)

"""Capture byte length and SHA-256 for every CSV artifact in the run tree."""
function checkpoint_csv_integrity(runctx::RunContext)
    result = Dict{String,Any}()
    for (directory, _, filenames) in walkdir(runctx.run_dir)
        for filename in filenames
            endswith(lowercase(filename), ".csv") || continue
            path = joinpath(directory, filename)
            relative = replace(_transaction_target_relative(runctx, path), '\\' => '/')
            result[relative] = Dict{String,Any}(
                "path" => relative,
                "size" => filesize(path),
                "sha256" => _io_sha256_file(path),
            )
        end
    end
    return result
end

function _payload_csv_integrity(payload)
    integrity = _io_get(payload, :csv_integrity, nothing)
    integrity isa AbstractDict || throw(OutputError(
        "checkpoint payload has no CSV integrity manifest",
    ))
    return integrity
end

function _refresh_payload_csv_integrity!(payload, runctx::RunContext, paths)
    payload isa AbstractDict || return nothing
    integrity_key = haskey(payload, "csv_integrity") ? "csv_integrity" :
                    haskey(payload, :csv_integrity) ? :csv_integrity : "csv_integrity"
    integrity = get!(payload, integrity_key, Dict{String,Any}())
    integrity isa AbstractDict || throw(OutputError("checkpoint CSV integrity value is invalid"))
    for path in paths
        endswith(lowercase(path), ".csv") || continue
        relative = replace(_transaction_target_relative(runctx, path), '\\' => '/')
        entry = Dict{String,Any}(
            "path" => relative,
            "size" => filesize(path),
            "sha256" => _io_sha256_file(path),
        )
        integrity[relative] = entry
    end
    return nothing
end

function verify_checkpoint_csv_integrity(runctx::RunContext, payload)
    integrity = _payload_csv_integrity(payload)
    expected_paths = Set{String}()
    for (raw_key, raw_entry) in pairs(integrity)
        raw_entry isa AbstractDict || throw(OutputError(
            "invalid CSV integrity entry for $(raw_key)",
        ))
        relative = String(_io_get(raw_entry, :path, ""))
        isempty(relative) && throw(OutputError("CSV integrity entry $(raw_key) has no path"))
        push!(expected_paths, replace(relative, '\\' => '/'))
        path = normpath(joinpath(runctx.run_dir, relative))
        _transaction_target_relative(runctx, path)
        isfile(path) || throw(OutputError(
            "resume refused: checkpointed CSV is missing: $(path)",
        ))
        expected_size = Int(_io_get(raw_entry, :size, -1))
        filesize(path) == expected_size || throw(OutputError(
            "resume refused: CSV byte length differs from checkpoint: $(path)",
        ))
        expected_hash = String(_io_get(raw_entry, :sha256, ""))
        _io_sha256_file(path) == expected_hash || throw(OutputError(
            "resume refused: CSV SHA-256 differs from checkpoint: $(path)",
        ))
    end
    actual_paths = Set{String}()
    for (directory, _, filenames) in walkdir(runctx.run_dir)
        for filename in filenames
            endswith(lowercase(filename), ".csv") || continue
            relative = replace(
                _transaction_target_relative(runctx, joinpath(directory, filename)),
                '\\' => '/',
            )
            push!(actual_paths, relative)
        end
    end
    actual_paths == expected_paths || throw(OutputError(
        "resume refused: run CSV set differs from checkpoint integrity manifest",
    ))
    return true
end

"""
Verify that every checkpointed CSV output ID agrees with the on-disk final row.

The function performs no truncation or repair.  Any missing, partial, ahead, or
behind CSV causes an `OutputError`, preventing an ambiguous automatic append.
It returns the actual per-file IDs after a successful check.
"""
function verify_resume_output_ids(run_dir::AbstractString, payload;
                                  filenames::AbstractDict=_RESUMABLE_CSV_NAMES)
    directory = abspath(String(run_dir))
    isdir(directory) || throw(OutputError("run directory not found: $(directory)"))
    expected = _checkpoint_progress(payload)
    actual = Dict{Symbol,Int}()

    if haskey(expected, :all)
        expected_id = expected[:all]
        for (key, filename) in filenames
            path = joinpath(directory, filename)
            isfile(path) || continue
            id = csv_last_output_id(path)
            actual[key] = id
            id == expected_id || throw(OutputError(
                "resume refused: $(filename) ends at output_id=$(id), checkpoint expects $(expected_id)",
            ))
        end
    else
        for (raw_key, expected_id) in expected
            raw_key in (:deltaN_summary, :deltan_summary, :deltaN_spectrum, :deltan_spectrum) && continue
            key = raw_key == :linear_curvature ? :linear_curvature_spectra : raw_key
            filename = get(filenames, key, string(key) * ".csv")
            path = joinpath(directory, filename)
            id = csv_last_output_id(path)
            actual[key] = id
            id == expected_id || throw(OutputError(
                "resume refused: $(filename) ends at output_id=$(id), checkpoint expects $(expected_id)",
            ))
        end
    end
    isempty(actual) && throw(OutputError("resume refused: no checkpointed CSV outputs were found"))
    context = _io_get_any(payload, (:runctx, :run_context), nothing)
    context isa RunContext && verify_checkpoint_csv_integrity(context, payload)
    return actual
end

const check_resume_output_consistency = verify_resume_output_ids

function _resume_transaction_is_new(info, payload; pending::Bool=false)
    checkpoint_id = Int(_io_get(payload, :output_id, -1))
    state = _checkpoint_required(payload, :state)
    checkpoint_step = Int(_io_get(state, :step, -1))
    phase = String(_io_get(payload, :phase, "main"))
    deltaN_written = Bool(_io_get(payload, :deltaN_outputs_written, false))

    if info.kind == :observation
        if pending
            info.transaction_id == checkpoint_id + 1 || throw(OutputError(
                "pending observation transaction id=$(info.transaction_id) is not the " *
                "single successor of checkpoint output_id=$(checkpoint_id)",
            ))
        elseif info.transaction_id <= checkpoint_id
            return false
        elseif info.transaction_id != checkpoint_id + 1
            throw(OutputError(
                "completed observation receipt is more than one output ahead of the checkpoint",
            ))
        end
        info.step == checkpoint_step || throw(OutputError(
            "output transaction step=$(info.step) does not match checkpoint state step=$(checkpoint_step)",
        ))
        return true
    elseif info.kind == :deltaN
        if deltaN_written
            pending && throw(OutputError(
                "pending delta-N transaction conflicts with checkpointed completed output",
            ))
            return false
        end
        phase == "deltaN_output" || throw(OutputError(
            "delta-N output transaction conflicts with checkpoint phase $(repr(phase))",
        ))
        info.transaction_id == checkpoint_id || throw(OutputError(
            "delta-N output transaction id differs from checkpoint output_id",
        ))
        info.step == checkpoint_step || throw(OutputError(
            "delta-N output transaction step differs from checkpoint state step",
        ))
        return true
    end

    pending && throw(OutputError(
        "unsupported pending output transaction kind $(info.kind) during resume",
    ))
    return false
end

function _apply_recovered_output_to_payload!(payload, runctx::RunContext, recovery)
    payload isa AbstractDict || throw(OutputError(
        "checkpoint payload cannot be updated after output transaction recovery",
    ))
    progress_key = haskey(payload, "last_output_ids") ? "last_output_ids" :
                   haskey(payload, :last_output_ids) ? :last_output_ids : "last_output_ids"
    progress = get!(payload, progress_key, Dict{Symbol,Int}())
    progress isa AbstractDict || throw(OutputError(
        "checkpoint CSV progress cannot be updated after transaction recovery",
    ))
    for (key, value) in runctx.last_output_ids
        if keytype(typeof(progress)) <: AbstractString
            progress[String(key)] = value
        else
            progress[key] = value
        end
    end
    if haskey(payload, "output_id")
        payload["output_id"] = runctx.output_id
    else
        payload[:output_id] = runctx.output_id
    end
    if recovery.kind == :observation
        if haskey(payload, "last_observation_step")
            payload["last_observation_step"] = recovery.step
        else
            payload[:last_observation_step] = recovery.step
        end
    elseif recovery.kind == :deltaN
        if haskey(payload, "deltaN_outputs_written")
            payload["deltaN_outputs_written"] = true
        else
            payload[:deltaN_outputs_written] = true
        end
    end
    _refresh_payload_csv_integrity!(payload, runctx, recovery.paths)
    return payload
end

"""Verify CSVs and restore `RunContext` counters from a checkpoint payload."""
function restore_output_progress!(runctx::RunContext, payload)
    recovery = nothing
    pending = pending_output_transaction_info(runctx)
    if pending !== nothing
        _resume_transaction_is_new(pending, payload; pending=true)
        recovery = recover_pending_output_transaction!(runctx)
    else
        completed = completed_output_transaction_info(runctx)
        if completed !== nothing && _resume_transaction_is_new(completed, payload)
            recovery = recover_completed_output_transaction!(runctx)
        end
    end
    if recovery !== nothing
        _apply_recovered_output_to_payload!(payload, runctx, recovery)
        # Test-only fault injection point for the crash window after the
        # pending journal was finalized but before a new checkpoint is saved.
        _invoke_output_fault_hook(
            :after_restore, recovery.kind, recovery.transaction_id,
            length(recovery.keys), :none,
        )
    end
    actual = verify_resume_output_ids(runctx.run_dir, payload)
    empty!(runctx.last_output_ids)
    merge!(runctx.last_output_ids, actual)
    runctx.output_id = maximum(values(actual))
    return runctx
end
