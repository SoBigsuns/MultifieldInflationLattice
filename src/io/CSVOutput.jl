import Statistics

const _PRIMARY_CSV_FILENAMES = Dict{Symbol,String}(
    :background => "background.csv",
    :energies => "energies.csv",
    :slowroll => "slowroll.csv",
    :diagnostics => "diagnostics.csv",
    :field_spectra => "field_spectra.csv",
    :linear_curvature_spectra => "linear_curvature_spectra.csv",
    :deltaN_summary => "deltaN_summary.csv",
    :deltaN_spectrum => "deltaN_spectrum.csv",
)

const _FIELD_SPECTRUM_HEADER = [
    "output_id", "step", "efolds", "a", "k_bin", "k_comoving", "k_physical",
    "n_modes", "field_i", "field_j", "power_dimensionless",
]
const _LINEAR_CURVATURE_HEADER = [
    "output_id", "step", "efolds", "a", "k_bin", "k_comoving", "k_physical",
    "n_modes", "component", "field_i", "field_j", "power_dimensionless",
]
const _DELTAN_SUMMARY_HEADER = [
    "efolds_reference", "rho_reference", "mean", "variance", "stddev", "skewness",
    "excess_kurtosis", "minimum", "maximum", "n_sites", "n_failed",
    "separate_universe_ratio", "separate_universe_ok",
]
const _DELTAN_SPECTRUM_HEADER = [
    "efolds_reference", "rho_reference", "k_bin", "k_comoving", "k_physical",
    "n_modes", "power_dimensionless",
]
const _HISTOGRAM_HEADER = [
    "quantity", "output_id", "bin_left", "bin_right", "bin_center", "count", "density",
]
const _SLICE_HEADER = ["i", "j", "x", "y", "value"]
const _VOLUME_HEADER = ["i", "j", "k", "x", "y", "z", "value"]
const _WINDOWS_RESERVED_STEMS = Set([
    "con", "prn", "aux", "nul", "clock\$",
    ("com$(index)" for index in 1:9)...,
    ("lpt$(index)" for index in 1:9)...,
])

function _csv_filename_stem(value)
    stem = lowercase(_io_safe_identifier(value; fallback="quantity"))
    return stem in _WINDOWS_RESERVED_STEMS ? stem * "_quantity" : stem
end

function _csv_safe_field_names(field_names)
    safe = String[]
    used = Set{String}()
    for (index, name) in enumerate(field_names)
        base = _io_safe_identifier(name; fallback="field_$(index)")
        candidate = base
        suffix = 2
        while lowercase(candidate) in used
            candidate = "$(base)_$(suffix)"
            suffix += 1
        end
        push!(used, lowercase(candidate))
        push!(safe, candidate)
    end
    return safe
end

function _csv_headers(field_ids::Vector{String})
    background = ["output_id", "step", "tau", "t", "efolds", "a", "H", "Hdot", "rho", "p", "w"]
    energies = [
        "output_id", "step", "efolds", "kinetic_total", "gradient_total",
        "potential_total", "rho_total",
    ]
    slowroll = [
        "output_id", "step", "efolds", "epsilon_H", "epsilon_V", "eta_parallel",
        "turn_rate", "turn_rate_over_H",
    ]
    for field in field_ids
        append!(background, ["$(field)_mean", "$(field)_velocity", "$(field)_variance"])
        append!(energies, ["kinetic_$(field)", "gradient_$(field)"])
        push!(slowroll, "epsilonH_$(field)")
    end
    for field in field_ids
        push!(slowroll, "etaV_diag_$(field)")
    end
    for index in eachindex(field_ids)
        push!(slowroll, "etaV_eig_$(index)")
    end
    diagnostics = [
        "output_id", "step", "efolds", "dtau", "cfl_ratio", "H2_constraint",
        "friedmann_residual", "field_min", "field_max", "finite_state",
    ]
    return Dict{Symbol,Vector{String}}(
        :background => background,
        :energies => energies,
        :slowroll => slowroll,
        :diagnostics => diagnostics,
        :field_spectra => copy(_FIELD_SPECTRUM_HEADER),
        :linear_curvature_spectra => copy(_LINEAR_CURVATURE_HEADER),
        :deltaN_summary => copy(_DELTAN_SUMMARY_HEADER),
        :deltaN_spectrum => copy(_DELTAN_SPECTRUM_HEADER),
    )
end

function _csv_validate_header(header::AbstractVector{<:AbstractString})
    isempty(header) && throw(OutputError("CSV header must not be empty"))
    length(unique(header)) == length(header) || throw(OutputError("CSV header contains duplicate columns"))
    for column in header
        occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", column) || throw(OutputError(
            "unsafe CSV column name $(repr(column)); only ASCII identifiers are allowed",
        ))
    end
    return String.(header)
end

function _csv_cell(value)
    value === nothing && return ""
    value === missing && return ""
    text = if value isa Bool
        value ? "true" : "false"
    elseif value isa AbstractFloat
        isnan(value) ? "NaN" : isinf(value) ? (signbit(value) ? "-Inf" : "Inf") : string(value)
    elseif value isa Real
        string(value)
    else
        replace(string(value), '\r' => "\\r", '\n' => "\\n")
    end
    if occursin(',', text) || occursin('"', text) || occursin('\r', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function _csv_row_value(row, column::AbstractString)
    key = Symbol(column)
    if row isa AbstractDict
        haskey(row, key) && return row[key]
        haskey(row, column) && return row[column]
    elseif hasproperty(row, key)
        return getproperty(row, key)
    end
    return nothing
end

function _csv_line(header, row)
    return join((_csv_cell(_csv_row_value(row, column)) for column in header), ',') * "\n"
end

function _csv_create(path::AbstractString, header)
    columns = _csv_validate_header(header)
    ispath(path) && throw(OutputError("refusing to overwrite existing CSV: $(abspath(path))"))
    return _io_atomic_write(path) do io
        write(io, join(columns, ','), "\n")
    end
end

function _csv_assert_header(path::AbstractString, expected)
    isfile(path) || throw(OutputError("CSV output is missing: $(abspath(path))"))
    actual = try
        open(path, "r") do io
            eof(io) && throw(OutputError("CSV output is empty: $(abspath(path))"))
            split(chomp(readline(io)), ','; keepempty=true)
        end
    catch err
        err isa InterruptException && rethrow()
        err isa OutputError && rethrow()
        throw(OutputError("failed to read CSV header $(abspath(path)): $(sprint(showerror, err))"))
    end
    actual == expected || throw(OutputError(
        "CSV header mismatch in $(abspath(path)); refusing unsafe append",
    ))
    return true
end

function _csv_has_data(path::AbstractString)
    isfile(path) || return false
    return open(path, "r") do io
        eof(io) && return false
        readline(io) # header
        for line in eachline(io)
            isempty(strip(line)) || return true
        end
        return false
    end
end

function _csv_append_rows!(runctx::RunContext, key::Symbol, rows)
    path = get(runctx.files, key, nothing)
    path === nothing && throw(OutputError("no output path registered for $(key)"))
    header = get(runctx.headers, key, nothing)
    header === nothing && throw(OutputError("no CSV schema registered for $(key)"))
    _csv_assert_header(path, header)

    sequence = (rows isa NamedTuple || rows isa AbstractDict) ? (rows,) : rows
    buffer = IOBuffer()
    count_rows = 0
    final_id = nothing
    requires_output_id = "output_id" in header
    for row in sequence
        id = _csv_row_value(row, "output_id")
        requires_output_id && id === nothing && throw(OutputError(
            "row for $(key) is missing required output_id",
        ))
        write(buffer, _csv_line(header, row))
        count_rows += 1
        id === nothing || (final_id = Int(id))
    end
    count_rows == 0 && return 0

    bytes = take!(buffer)
    try
        open(path, "a") do io
            write(io, bytes)
            flush(io)
        end
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError("failed to append CSV $(path): $(sprint(showerror, err))"))
    end
    final_id === nothing || (runctx.last_output_ids[key] = final_id)
    return count_rows
end

const _OUTPUT_TRANSACTION_SCHEMA = "MFIL-OUTPUT-TXN-2"
const _OUTPUT_FAULT_HOOK = Ref{Any}(nothing)

struct _OutputTransactionPart
    key::Symbol
    path::String
    header::Vector{String}
    bytes::Vector{UInt8}
    allow_create::Bool
    transient::Bool
    once::Bool
    row_count::Int
    data_sha256::String
    last_output_id::Union{Nothing,Int}
end

function _csv_serialized_rows(header, rows; include_header::Bool=false)
    columns = _csv_validate_header(header)
    sequence = (rows isa NamedTuple || rows isa AbstractDict) ? (rows,) : rows
    buffer = IOBuffer()
    include_header && write(buffer, join(columns, ','), "\n")
    count_rows = 0
    final_id = nothing
    requires_output_id = "output_id" in columns
    for row in sequence
        id = _csv_row_value(row, "output_id")
        requires_output_id && id === nothing && throw(OutputError(
            "transaction row is missing required output_id",
        ))
        write(buffer, _csv_line(columns, row))
        count_rows += 1
        id === nothing || (final_id = Int(id))
    end
    return (bytes=take!(buffer), rows=count_rows, last_output_id=final_id)
end

function _transaction_part(runctx::RunContext, key::Symbol, rows;
                           path=get(runctx.files, key, ""),
                           header=get(runctx.headers, key, String[]),
                           allow_create::Bool=false, transient::Bool=false,
                           once::Bool=false)
    isempty(path) && throw(OutputError("no output path registered for transaction part $(key)"))
    columns = _csv_validate_header(header)
    exists = isfile(path)
    if exists
        _csv_assert_header(path, columns)
    elseif !allow_create
        throw(OutputError("transaction target is missing: $(abspath(path))"))
    end
    serialized = _csv_serialized_rows(columns, rows; include_header=false)
    bytes = if exists
        serialized.bytes
    else
        header_bytes = Vector{UInt8}(codeunits(join(columns, ',') * "\n"))
        vcat(header_bytes, serialized.bytes)
    end
    return _OutputTransactionPart(
        key, abspath(String(path)), columns, bytes,
        allow_create, transient, once, serialized.rows,
        _io_sha256_bytes(serialized.bytes), serialized.last_output_id,
    )
end

function _output_transaction_paths(runctx::RunContext)
    return (
        pending=joinpath(runctx.run_dir, ".output-transaction"),
        staging=joinpath(runctx.run_dir, ".output-transaction.tmp"),
        receipt=joinpath(runctx.run_dir, ".output-transaction.last.toml"),
    )
end

function _safe_remove_output_transaction(path::AbstractString, runctx::RunContext)
    absolute = abspath(String(path))
    parent = abspath(dirname(absolute))
    parent == abspath(runctx.run_dir) || throw(OutputError(
        "refusing to remove transaction directory outside the run: $(absolute)",
    ))
    basename(absolute) in (".output-transaction", ".output-transaction.tmp") ||
        throw(OutputError("unrecognized output transaction directory: $(absolute)"))
    ispath(absolute) && rm(absolute; recursive=true, force=true)
    return nothing
end

function _transaction_target_relative(runctx::RunContext, target::AbstractString)
    relative = relpath(abspath(String(target)), abspath(runctx.run_dir))
    (relative == ".." || startswith(relative, "../") || startswith(relative, "..\\")) &&
        throw(OutputError("transaction target escapes the run directory: $(target)"))
    return relative
end

_transaction_portable_path(path::AbstractString) = replace(String(path), '\\' => '/')

function _transaction_part_identity(runctx::RunContext, part::_OutputTransactionPart)
    return Dict{String,Any}(
        "key" => String(part.key),
        "target" => _transaction_portable_path(
            _transaction_target_relative(runctx, part.path),
        ),
        "header" => part.header,
        "transient" => part.transient,
        "once" => part.once,
        "row_count" => part.row_count,
        "data_sha256" => part.data_sha256,
    )
end

function _transaction_entry_identity(entry::AbstractDict)
    return Dict{String,Any}(
        "key" => String(entry["key"]),
        "target" => _transaction_portable_path(String(entry["target"])),
        "header" => String.(entry["header"]),
        "transient" => Bool(entry["transient"]),
        "once" => Bool(get(entry, "once", false)),
        "row_count" => Int(entry["row_count"]),
        "data_sha256" => String(entry["data_sha256"]),
    )
end

function _output_transaction_identity(runctx::RunContext, kind::Symbol,
                                      transaction_id::Integer, step::Integer,
                                      parts::Vector{_OutputTransactionPart}, updates)
    return Dict{String,Any}(
        "schema_version" => _OUTPUT_TRANSACTION_SCHEMA,
        "kind" => String(kind),
        "transaction_id" => Int(transaction_id),
        "step" => Int(step),
        "updates" => Dict(string(key) => _io_plain(value) for (key, value) in pairs(updates)),
        "parts" => [_transaction_part_identity(runctx, part) for part in parts],
    )
end

function _manifest_transaction_identity(manifest::AbstractDict)
    return Dict{String,Any}(
        "schema_version" => String(manifest["schema_version"]),
        "kind" => String(manifest["kind"]),
        "transaction_id" => Int(manifest["transaction_id"]),
        "step" => Int(manifest["step"]),
        "updates" => _io_plain(get(manifest, "updates", Dict{String,Any}())),
        "parts" => [_transaction_entry_identity(entry) for entry in manifest["parts"]],
    )
end

function _output_transaction_digest(identity::AbstractDict)
    return _io_sha256_bytes(_io_config_bytes(identity))
end

function _stage_output_transaction!(runctx::RunContext, kind::Symbol,
                                    transaction_id::Integer, step::Integer,
                                    parts::Vector{_OutputTransactionPart}, updates)
    paths = _output_transaction_paths(runctx)
    isdir(paths.pending) && throw(OutputError(
        "an unfinished output transaction must be recovered before starting another",
    ))
    ispath(paths.staging) && _safe_remove_output_transaction(paths.staging, runctx)
    mkdir(paths.staging)
    manifest_parts = Vector{Dict{String,Any}}()
    try
        for (index, part) in enumerate(parts)
            chunk_name = lpad(string(index), 4, '0') * ".part"
            chunk_path = joinpath(paths.staging, chunk_name)
            _io_atomic_write(chunk_path) do io
                write(io, part.bytes)
            end
            target_exists = isfile(part.path)
            base_size = target_exists ? filesize(part.path) : 0
            target_exists || part.allow_create || throw(OutputError(
                "transaction target disappeared before staging: $(part.path)",
            ))
            entry = Dict{String,Any}(
                "key" => String(part.key),
                "target" => _transaction_target_relative(runctx, part.path),
                "header" => part.header,
                "chunk" => chunk_name,
                "chunk_size" => length(part.bytes),
                "chunk_sha256" => _io_sha256_bytes(part.bytes),
                "base_size" => base_size,
                "create" => !target_exists,
                "transient" => part.transient,
                "once" => part.once,
                "row_count" => part.row_count,
                "data_sha256" => part.data_sha256,
            )
            part.last_output_id === nothing ||
                (entry["last_output_id"] = part.last_output_id)
            push!(manifest_parts, entry)
        end
        manifest = Dict{String,Any}(
            "schema_version" => _OUTPUT_TRANSACTION_SCHEMA,
            "kind" => String(kind),
            "transaction_id" => Int(transaction_id),
            "step" => Int(step),
            "updates" => Dict(string(key) => _io_plain(value) for (key, value) in pairs(updates)),
            "parts" => manifest_parts,
        )
        manifest["transaction_sha256"] = _output_transaction_digest(
            _manifest_transaction_identity(manifest),
        )
        _io_write_toml(joinpath(paths.staging, "manifest.toml"), manifest)
        mv(paths.staging, paths.pending; force=false)
    catch
        # No target is changed before the staging directory becomes the durable pending directory.
        ispath(paths.staging) && _safe_remove_output_transaction(paths.staging, runctx)
        rethrow()
    end
    return paths.pending
end

function _read_tail(path::AbstractString, offset::Integer, count::Integer)
    count == 0 && return UInt8[]
    return open(path, "r") do io
        seek(io, offset)
        read(io, count)
    end
end

function _apply_transaction_part!(runctx::RunContext, directory::AbstractString,
                                  entry::AbstractDict)
    target = normpath(joinpath(runctx.run_dir, String(entry["target"])))
    _transaction_target_relative(runctx, target) # containment check
    chunk_path = joinpath(directory, String(entry["chunk"]))
    isfile(chunk_path) || throw(OutputError("output transaction chunk is missing: $(chunk_path)"))
    chunk = read(chunk_path)
    length(chunk) == Int(entry["chunk_size"]) || throw(OutputError(
        "output transaction chunk length mismatch: $(chunk_path)",
    ))
    _io_sha256_bytes(chunk) == String(entry["chunk_sha256"]) || throw(OutputError(
        "output transaction chunk hash mismatch: $(chunk_path)",
    ))
    base_size = Int(entry["base_size"])
    expected_size = base_size + length(chunk)
    create = Bool(entry["create"])
    current_size = isfile(target) ? filesize(target) : -1

    if create
        if current_size == -1
            _io_atomic_write(target) do io
                write(io, chunk)
            end
        elseif current_size == length(chunk) && _io_sha256_file(target) == String(entry["chunk_sha256"])
            # A previous attempt completed this new artifact.
        else
            throw(OutputError(
                "partial or conflicting output transaction target $(target); " *
                "expected absent or an exact $(length(chunk))-byte artifact, found $(current_size) bytes",
            ))
        end
    elseif current_size == base_size
        Base.disable_sigint() do
            open(target, "a") do io
                write(io, chunk)
                flush(io)
            end
        end
    elseif current_size == expected_size
        tail = _read_tail(target, base_size, length(chunk))
        _io_sha256_bytes(tail) == String(entry["chunk_sha256"]) || throw(OutputError(
            "output transaction target has the expected length but different trailing data: $(target)",
        ))
    else
        throw(OutputError(
            "partial or conflicting output transaction target $(target); " *
            "expected $(base_size) or $(expected_size) bytes, found $(current_size)",
        ))
    end

    # Re-read the exact committed bytes even after a nominally successful write.
    actual_size = filesize(target)
    actual_size == expected_size || throw(OutputError(
        "output transaction write was incomplete for $(target)",
    ))
    tail = _read_tail(target, base_size, length(chunk))
    _io_sha256_bytes(tail) == String(entry["chunk_sha256"]) || throw(OutputError(
        "output transaction verification failed for $(target)",
    ))

    key = Symbol(String(entry["key"]))
    transient = Bool(entry["transient"])
    if !transient
        runctx.files[key] = target
        runctx.headers[key] = String.(entry["header"])
    end
    haskey(entry, "last_output_id") &&
        (runctx.last_output_ids[key] = Int(entry["last_output_id"]))
    return (key=key, path=target)
end

function _apply_transaction_updates!(runctx::RunContext, updates::AbstractDict)
    haskey(updates, "output_id") && (runctx.output_id = Int(updates["output_id"]))
    haskey(updates, "next_background_efolds") &&
        (runctx.next_background_efolds = Float64(updates["next_background_efolds"]))
    haskey(updates, "next_spectra_efolds") &&
        (runctx.next_spectra_efolds = Float64(updates["next_spectra_efolds"]))
    haskey(updates, "next_histogram_efolds") &&
        (runctx.next_histogram_efolds = Float64(updates["next_histogram_efolds"]))
    haskey(updates, "next_checkpoint_efolds") &&
        (runctx.next_checkpoint_efolds = Float64(updates["next_checkpoint_efolds"]))
    return runctx
end

function _invoke_output_fault_hook(event::Symbol, kind::Symbol, transaction_id::Int,
                                   part_index::Int, key::Symbol)
    hook = _OUTPUT_FAULT_HOOK[]
    hook === nothing || hook(event, kind, transaction_id, part_index, key)
    return nothing
end

function _read_output_transaction_document(path::AbstractString, label::AbstractString)
    isfile(path) || throw(OutputError("$(label) is missing: $(path)"))
    document = try
        TOML.parsefile(path)
    catch err
        err isa InterruptException && rethrow()
        throw(OutputError("cannot parse $(label): $(sprint(showerror, err))"))
    end
    get(document, "schema_version", "") == _OUTPUT_TRANSACTION_SCHEMA || throw(OutputError(
        "unsupported output transaction schema in $(path)",
    ))
    recorded_digest = String(get(document, "transaction_sha256", ""))
    isempty(recorded_digest) && throw(OutputError("$(label) has no transaction digest"))
    actual_digest = _output_transaction_digest(_manifest_transaction_identity(document))
    actual_digest == recorded_digest || throw(OutputError(
        "$(label) transaction digest mismatch: $(path)",
    ))
    return document
end

function pending_output_transaction_info(runctx::RunContext)
    directory = _output_transaction_paths(runctx).pending
    isdir(directory) || return nothing
    manifest = _read_output_transaction_document(
        joinpath(directory, "manifest.toml"), "pending output transaction manifest",
    )
    return (
        kind=Symbol(String(manifest["kind"])),
        transaction_id=Int(manifest["transaction_id"]),
        step=Int(manifest["step"]),
    )
end

function completed_output_transaction_info(runctx::RunContext)
    path = _output_transaction_paths(runctx).receipt
    isfile(path) || return nothing
    receipt = _read_output_transaction_document(path, "completed output transaction receipt")
    get(receipt, "status", "") == "committed" || throw(OutputError(
        "completed output transaction receipt has invalid status: $(path)",
    ))
    return (
        kind=Symbol(String(receipt["kind"])),
        transaction_id=Int(receipt["transaction_id"]),
        step=Int(receipt["step"]),
    )
end

function _write_output_transaction_receipt!(runctx::RunContext, manifest::AbstractDict)
    receipt_parts = Vector{Dict{String,Any}}()
    for entry in manifest["parts"]
        identity = _transaction_entry_identity(entry)
        target = normpath(joinpath(runctx.run_dir, String(identity["target"])))
        _transaction_target_relative(runctx, target)
        isfile(target) || throw(OutputError(
            "cannot finalize output transaction; target is missing: $(target)",
        ))
        identity["final_size"] = filesize(target)
        identity["final_sha256"] = _io_sha256_file(target)
        haskey(entry, "last_output_id") &&
            (identity["last_output_id"] = Int(entry["last_output_id"]))
        push!(receipt_parts, identity)
    end
    receipt = Dict{String,Any}(
        "schema_version" => _OUTPUT_TRANSACTION_SCHEMA,
        "status" => "committed",
        "kind" => String(manifest["kind"]),
        "transaction_id" => Int(manifest["transaction_id"]),
        "step" => Int(manifest["step"]),
        "updates" => _io_plain(get(manifest, "updates", Dict{String,Any}())),
        "parts" => receipt_parts,
        "committed_at_utc" => Dates.format(
            Dates.now(Dates.UTC), Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.sssZ"),
        ),
    )
    receipt["transaction_sha256"] = _output_transaction_digest(
        _manifest_transaction_identity(receipt),
    )
    _io_write_toml(_output_transaction_paths(runctx).receipt, receipt)
    return receipt
end

function recover_completed_output_transaction!(runctx::RunContext)
    path = _output_transaction_paths(runctx).receipt
    isfile(path) || return nothing
    receipt = _read_output_transaction_document(path, "completed output transaction receipt")
    get(receipt, "status", "") == "committed" || throw(OutputError(
        "completed output transaction receipt has invalid status: $(path)",
    ))
    applied_keys = Symbol[]
    applied_paths = String[]
    for entry in receipt["parts"]
        target = normpath(joinpath(runctx.run_dir, String(entry["target"])))
        _transaction_target_relative(runctx, target)
        isfile(target) || throw(OutputError(
            "completed output transaction target is missing: $(target)",
        ))
        filesize(target) == Int(entry["final_size"]) || throw(OutputError(
            "completed output transaction target length mismatch: $(target)",
        ))
        _io_sha256_file(target) == String(entry["final_sha256"]) || throw(OutputError(
            "completed output transaction target hash mismatch: $(target)",
        ))
        key = Symbol(String(entry["key"]))
        if !Bool(entry["transient"])
            runctx.files[key] = target
            runctx.headers[key] = String.(entry["header"])
        end
        haskey(entry, "last_output_id") &&
            (runctx.last_output_ids[key] = Int(entry["last_output_id"]))
        push!(applied_keys, key)
        push!(applied_paths, target)
    end
    _apply_transaction_updates!(runctx, get(receipt, "updates", Dict{String,Any}()))
    return (
        kind=Symbol(String(receipt["kind"])),
        transaction_id=Int(receipt["transaction_id"]),
        step=Int(receipt["step"]),
        keys=applied_keys,
        paths=applied_paths,
    )
end

function _matching_completed_output_transaction!(runctx::RunContext, kind::Symbol,
                                                 transaction_id::Integer,
                                                 step::Integer,
                                                 parts::Vector{_OutputTransactionPart},
                                                 updates)
    info = completed_output_transaction_info(runctx)
    info === nothing && return nothing
    (info.kind == kind && info.transaction_id == Int(transaction_id) &&
     info.step == Int(step)) || return nothing
    receipt = _read_output_transaction_document(
        _output_transaction_paths(runctx).receipt,
        "completed output transaction receipt",
    )
    expected = _output_transaction_identity(
        runctx, kind, transaction_id, step, parts, updates,
    )
    expected_digest = _output_transaction_digest(expected)
    expected_digest == String(receipt["transaction_sha256"]) || throw(OutputError(
        "output transaction identity collision for $(kind) id=$(transaction_id), step=$(step)",
    ))
    return recover_completed_output_transaction!(runctx)
end

"""
Recover and finish the durable output transaction in `runctx.run_dir`.

Each target must be either untouched or contain the exact staged suffix. A
partial/different suffix is never guessed, truncated, or overwritten.
"""
function recover_pending_output_transaction!(runctx::RunContext)
    directory = _output_transaction_paths(runctx).pending
    isdir(directory) || return nothing
    manifest_path = joinpath(directory, "manifest.toml")
    manifest = _read_output_transaction_document(
        manifest_path, "pending output transaction manifest",
    )
    kind = Symbol(String(manifest["kind"]))
    transaction_id = Int(manifest["transaction_id"])
    step = Int(manifest["step"])
    entries = manifest["parts"]
    applied_keys = Symbol[]
    applied_paths = String[]
    for (index, entry) in enumerate(entries)
        applied = _apply_transaction_part!(runctx, directory, entry)
        push!(applied_keys, applied.key)
        push!(applied_paths, applied.path)
        _invoke_output_fault_hook(:after_part, kind, transaction_id, index, applied.key)
    end
    _invoke_output_fault_hook(:before_finalize, kind, transaction_id, length(entries), :none)
    Base.disable_sigint() do
        _apply_transaction_updates!(runctx, get(manifest, "updates", Dict{String,Any}()))
        _write_output_transaction_receipt!(runctx, manifest)
        _safe_remove_output_transaction(directory, runctx)
    end
    return (
        kind=kind, transaction_id=transaction_id, step=step,
        keys=applied_keys, paths=applied_paths,
    )
end

function commit_output_transaction!(runctx::RunContext, kind::Symbol,
                                    transaction_id::Integer, step::Integer,
                                    parts::Vector{_OutputTransactionPart}; updates=Dict{String,Any}())
    isempty(parts) && return nothing
    recovered = recover_pending_output_transaction!(runctx)
    if recovered !== nothing
        matched = _matching_completed_output_transaction!(
            runctx, kind, transaction_id, step, parts, updates,
        )
        matched === nothing && throw(OutputError(
            "recovered a previous transaction; retry the requested output operation explicitly",
        ))
        return matched
    end
    matched = _matching_completed_output_transaction!(
        runctx, kind, transaction_id, step, parts, updates,
    )
    matched === nothing || return matched
    for part in parts
        part.once && _csv_has_data(part.path) && throw(OutputError(
            "one-shot output already contains data without a matching transaction receipt: " *
            part.path,
        ))
    end
    _stage_output_transaction!(runctx, kind, transaction_id, step, parts, updates)
    return recover_pending_output_transaction!(runctx)
end

function _write_data_dictionary(path::AbstractString, field_names, field_ids)
    mappings = join(("- `$(name)` -> `$(identifier)`" for (name, identifier) in zip(field_names, field_ids)), "\n")
    content = """# Data dictionary

Schema version: `$(MFIL_OUTPUT_SCHEMA_VERSION)`

All files are RFC-4180-compatible, comma-delimited UTF-8 text with one header row.
Indices and `output_id` are integers. Floating-point undefined values are written
as IEEE `NaN`; strings that require quoting use doubled quote escaping. Unless a
column says otherwise, dimensionful values use reduced Planck units with
`M_Pl^red = 1`.

## Field-name mapping

$(mappings)

## Common conventions

- `step`: accepted integrator step number.
- `tau`, `t`: conformal and cosmic time.
- `efolds`: `log(a/a_initial)`.
- `H`, `Hdot`: Hubble rate and its cosmic-time derivative.
- Fourier transform: `X_k = dx^3 sum_x exp(-ikx) X(x)`.
- `k_comoving` and `k_physical=k_comoving/a` use the configured lattice momentum convention.
- `power_dimensionless = k_eff^3 <Re(X_k Y_k*)>/(2 pi^2 L^3)`; the zero mode is excluded.
- Cross-spectrum rows store `P_ij`, while a total linear-curvature spectrum includes `2 P_ij` for `i<j`.
- Linear curvature uses `zeta_I=-H phidotbar_I delta_phi_I/sum_J(phidotbar_J^2)`.
- Nonlinear curvature uses `zeta_deltaN=N_local-mean(N_local)` and is not split by field.

## background.csv

One wide row per background output. Base columns are `output_id,step,tau,t,efolds,a,H,Hdot,rho,p,w`.
For each normalized field name `f`, `f_mean`, `f_velocity`, and `f_variance`
are the spatial mean, physical cosmic-time mean velocity, and population variance.

## energies.csv

`kinetic_total`, `gradient_total`, `potential_total`, and `rho_total` are volume
averages. `kinetic_f` and `gradient_f` give the unambiguous field-wise pieces;
potential energy is intentionally not decomposed between interacting fields.

## slowroll.csv

Contains `epsilon_H`, `epsilon_V`, `eta_parallel`, physical `turn_rate`,
`turn_rate_over_H`, per-field `epsilonH_f` and `etaV_diag_f`, and the ascending
eigenvalues `etaV_eig_i` of `V_,IJ/V`. Undefined speed-dependent quantities are `NaN`.

## diagnostics.csv

`dtau` is the physical conformal step, `cfl_ratio=sqrt(3)*dtau_code/dx_code`,
`H2_constraint=rho_physical/3`; the dimensionless residual is evaluated as
`abs(H_code^2-rho_code/3)/max(H_code^2,rho_code/3,rho_floor)`.
`finite_state` reports the finite-value scan of the saved state.

## field_spectra.csv

Long-form field auto/cross spectra. `field_i` and `field_j` preserve the user-facing
field names; `k_bin` is the rounded integer-wavevector shell and `n_modes` its population.

## linear_curvature_spectra.csv

Long-form `self`, `cross`, and `total` rows. When the mean trajectory speed is below
the configured threshold, powers are `NaN`, not zero.

## deltaN_summary.csv and deltaN_spectrum.csv

The summary records the common reference surface, population moments, extrema,
site/failure counts, and separate-universe diagnostic. The spectrum contains only
the all-field nonlinear `deltaN` curvature map.

## histograms/*.csv

Columns are `quantity,output_id,bin_left,bin_right,bin_center,count,density`.
Density integrates to one over finite samples.

## slices/*.csv and optional snapshots/*.csv

A slice is one quantity and output time per file with `i,j,x,y,value`; `i,j` are
one-based in-plane lattice indices and `x,y` are physical in-plane coordinates.
Optional volumes use `i,j,k,x,y,z,value` and are disabled by default.
"""
    return _io_atomic_write(path) do io
        write(io, content)
    end
end

"""
Create schemas and immutable run provenance before the first numerical output.
"""
function initialize_outputs!(runctx::RunContext, config, input_path::AbstractString, model, lattice)
    isdir(runctx.run_dir) || throw(OutputError("run directory is missing: $(runctx.run_dir)"))
    field_names = String.(collect(_io_get(model, :field_names,
                                          _io_cfg(config, :model, :field_names, String[]))))
    isempty(field_names) && throw(OutputError("cannot initialize output without field names"))
    field_ids = _csv_safe_field_names(field_names)
    headers = _csv_headers(field_ids)

    for (key, filename) in _PRIMARY_CSV_FILENAMES
        runctx.files[key] = joinpath(runctx.run_dir, filename)
    end
    runctx.files[:config_input] = joinpath(runctx.run_dir, "config.input.toml")
    runctx.files[:config_effective] = joinpath(runctx.run_dir, "config.effective.toml")
    runctx.files[:metadata] = joinpath(runctx.run_dir, "metadata.toml")
    runctx.files[:data_dictionary] = joinpath(runctx.run_dir, "data_dictionary.md")
    runctx.files[:log] = joinpath(runctx.run_dir, "run.log")
    runctx.files[:checkpoint] = joinpath(runctx.run_dir, "checkpoint.jld2")
    merge!(runctx.headers, headers)

    mkpath(joinpath(runctx.run_dir, "histograms"))
    mkpath(joinpath(runctx.run_dir, "slices"))
    Bool(_io_cfg(config, :output, :save_3d_snapshots, false)) &&
        mkpath(joinpath(runctx.run_dir, "snapshots"))

    _io_copy_input_config(input_path, runctx.files[:config_input])
    effective = isdefined(@__MODULE__, :effective_config_dict) ? effective_config_dict(config) : _io_plain(config)
    _io_write_toml(runctx.files[:config_effective], effective)
    runctx.effective_hash = _io_sha256_file(runctx.files[:config_effective])
    _write_data_dictionary(runctx.files[:data_dictionary], field_names, field_ids)
    _io_atomic_write(runctx.files[:log]) do io
        write(io, "")
    end

    for key in keys(_PRIMARY_CSV_FILENAMES)
        _csv_create(runctx.files[key], runctx.headers[key])
        key in (:deltaN_summary, :deltaN_spectrum) || (runctx.last_output_ids[key] = -1)
    end

    runctx.next_background_efolds = 0.0
    runctx.next_spectra_efolds = 0.0
    runctx.next_histogram_efolds = 0.0
    runctx.next_checkpoint_efolds = Float64(_io_cfg(
        config, :output, :checkpoint_interval_efolds, Inf,
    ))
    runctx.metadata = _io_base_metadata(runctx, config, model, lattice)
    runctx.metadata["configuration"]["effective_sha256"] = runctx.effective_hash
    runctx.metadata["output"] = Dict{String,Any}(
        "directory" => runctx.run_dir,
        "field_identifiers" => Dict(field_names[index] => field_ids[index] for index in eachindex(field_names)),
        "csv_delimiter" => ",",
    )
    write_metadata!(runctx)
    write_run_log!(runctx, "INFO", "initialized run $(runctx.run_id)")
    return runctx
end

@inline function _vector_entry(value, index::Int, default=NaN)
    value === nothing && return default
    index <= length(value) || return default
    return value[index]
end

function _schedule_due_and_advance!(runctx::RunContext, field::Symbol,
                                    current::Real, interval::Real)
    next_value = Float64(getfield(runctx, field))
    tolerance = 16eps(Float64) * max(abs(Float64(current)), 1.0)
    Float64(current) + tolerance < next_value && return false
    step = Float64(interval)
    step > 0.0 || throw(OutputError("output interval must be positive"))
    while next_value <= Float64(current) + tolerance
        next_value += step
    end
    setfield!(runctx, field, next_value)
    return true
end

function _schedule_is_due(runctx::RunContext, field::Symbol, current::Real)
    next_value = Float64(getfield(runctx, field))
    tolerance = 16eps(Float64) * max(abs(Float64(current)), 1.0)
    return Float64(current) + tolerance >= next_value
end

function _schedule_value_after(runctx::RunContext, field::Symbol,
                               current::Real, interval::Real)
    next_value = Float64(getfield(runctx, field))
    tolerance = 16eps(Float64) * max(abs(Float64(current)), 1.0)
    Float64(current) + tolerance < next_value && return next_value
    spacing = Float64(interval)
    spacing > 0.0 || throw(OutputError("output interval must be positive"))
    while next_value <= Float64(current) + tolerance
        next_value += spacing
    end
    return next_value
end

function _diagnostic_row(output_id, state, model, lattice, energy)
    Hcode = state.a_prime / (state.a * state.a)
    constraint = energy.rho / 3.0
    residual = friedmann_residual(state, energy.rho)
    dtau_code = Float64(_io_get(state, :last_dtau, NaN))
    dtau = dtau_code / model.B
    cfl = sqrt(3.0) * dtau_code / lattice.dx
    finite_state = isfinite(state.a) && isfinite(state.a_prime) &&
                   all(isfinite, state.field) && all(isfinite, state.field_prime)
    return (
        output_id=output_id,
        step=state.step,
        efolds=state.efolds,
        dtau=dtau,
        cfl_ratio=cfl,
        H2_constraint=constraint * model.B * model.B,
        friedmann_residual=residual,
        field_min=minimum(state.field),
        field_max=maximum(state.field),
        finite_state=finite_state,
    )
end

"""Write one committed background observation and any due optional products."""
function write_observation!(runctx::RunContext, state, model, lattice, config, cache)
    output_id = runctx.output_id + 1
    energy_code = energy_summary(state, model, lattice, cache)
    energy = isdefined(@__MODULE__, :physical_energy_summary) ?
        physical_energy_summary(energy_code, model) : energy_code
    background = background_summary(state, model, lattice, energy_code)
    slowroll = slowroll_summary(
        state, model, lattice, energy_code, cache;
        min_speed2=Float64(_io_cfg(config, :spectra, :linear_zeta_min_speed2, 1.0e-30)),
    )
    field_ids = _csv_safe_field_names(String.(collect(model.field_names)))

    background_row = Dict{String,Any}(
        "output_id" => output_id, "step" => background.step, "tau" => background.tau,
        "t" => background.t, "efolds" => background.efolds, "a" => background.a,
        "H" => background.H, "Hdot" => background.Hdot, "rho" => background.rho,
        "p" => background.p, "w" => background.w,
    )
    energy_row = Dict{String,Any}(
        "output_id" => output_id, "step" => state.step, "efolds" => state.efolds,
        "kinetic_total" => energy.kinetic_total, "gradient_total" => energy.gradient_total,
        "potential_total" => energy.potential_total, "rho_total" => energy.rho,
    )
    slowroll_row = Dict{String,Any}(
        "output_id" => output_id, "step" => state.step, "efolds" => state.efolds,
        "epsilon_H" => slowroll.epsilon_H, "epsilon_V" => slowroll.epsilon_V,
        "eta_parallel" => slowroll.eta_parallel, "turn_rate" => slowroll.turn_rate,
        "turn_rate_over_H" => slowroll.turn_rate_over_H,
    )
    for index in eachindex(field_ids)
        field = field_ids[index]
        background_row["$(field)_mean"] = _vector_entry(background.field_means, index)
        background_row["$(field)_velocity"] = _vector_entry(background.field_velocities, index)
        background_row["$(field)_variance"] = _vector_entry(background.field_variances, index)
        energy_row["kinetic_$(field)"] = _vector_entry(energy.kinetic_by_field, index)
        energy_row["gradient_$(field)"] = _vector_entry(energy.gradient_by_field, index)
        slowroll_row["epsilonH_$(field)"] = _vector_entry(slowroll.epsilonH_by_field, index)
        slowroll_row["etaV_diag_$(field)"] = _vector_entry(slowroll.etaV_diag, index)
        slowroll_row["etaV_eig_$(index)"] = _vector_entry(slowroll.etaV_eigenvalues, index)
    end

    parts = _OutputTransactionPart[
        _transaction_part(runctx, :background, background_row),
        _transaction_part(runctx, :energies, energy_row),
        _transaction_part(runctx, :slowroll, slowroll_row),
        _transaction_part(
            runctx, :diagnostics,
            _diagnostic_row(output_id, state, model, lattice, energy_code),
        ),
    ]

    spectra_enabled = Bool(_io_cfg(config, :spectra, :enabled, true))
    spectra_interval = Float64(_io_cfg(config, :output, :spectra_interval_efolds, 0.01))
    spectra_due = spectra_enabled && _schedule_is_due(
        runctx, :next_spectra_efolds, state.efolds,
    )
    if spectra_due
        push!(parts, _transaction_part(
            runctx, :field_spectra,
            field_spectra(state, model, lattice, config, output_id),
        ))
        curvature_rows = linear_curvature_spectra(state, model, lattice, config, output_id)
        push!(parts, _transaction_part(runctx, :linear_curvature_spectra, curvature_rows))
        any(row -> !isfinite(row.power_dimensionless), curvature_rows) && record_warning!(
            runctx, "linear curvature output contains undefined values because the mean field speed is too small",
        )
    end

    histogram_interval = Float64(_io_cfg(config, :output, :histogram_interval_efolds, 0.01))
    histogram_due = _schedule_is_due(runctx, :next_histogram_efolds, state.efolds)
    if histogram_due
        bins = Int(_io_cfg(config, :output, :histogram_bins, 256))
        for index in eachindex(field_ids)
            push!(parts, _histogram_transaction_part(
                runctx, field_ids[index], output_id,
                @view(state.field[:, :, :, index]), bins,
            ))
        end
        if isdefined(@__MODULE__, :linear_curvature_maps)
            maps = linear_curvature_maps(state, model, lattice, config)
            maps.defined && push!(parts, _histogram_transaction_part(
                runctx, "zeta_lin", output_id, maps.total, bins,
            ))
        end
        append!(parts, _configured_slice_transaction_parts(
            runctx, state, model, lattice, config, output_id,
        ))
    end

    background_interval = Float64(_io_cfg(
        config, :output, :background_interval_efolds, 0.01,
    ))
    updates = Dict{String,Any}(
        "output_id" => output_id,
        "next_background_efolds" => _schedule_value_after(
            runctx, :next_background_efolds, state.efolds, background_interval,
        ),
        "next_spectra_efolds" => spectra_due ? _schedule_value_after(
            runctx, :next_spectra_efolds, state.efolds, spectra_interval,
        ) : runctx.next_spectra_efolds,
        "next_histogram_efolds" => histogram_due ? _schedule_value_after(
            runctx, :next_histogram_efolds, state.efolds, histogram_interval,
        ) : runctx.next_histogram_efolds,
    )
    commit_output_transaction!(
        runctx, :observation, output_id, state.step, parts; updates=updates,
    )
    return output_id
end

function _finite_values(values)
    result = Float64[]
    sizehint!(result, length(values))
    for value in values
        number = Float64(value)
        isfinite(number) && push!(result, number)
    end
    return result
end

"""Compute and append a normalized finite-value histogram."""
function _histogram_rows(quantity::AbstractString, output_id::Integer, values, bins::Integer)
    bins > 0 || throw(OutputError("histogram bin count must be positive"))
    samples = _finite_values(values)
    isempty(samples) && throw(OutputError("cannot histogram an all-nonfinite quantity: $(quantity)"))
    minimum_value, maximum_value = extrema(samples)
    if minimum_value == maximum_value
        half_width = max(abs(minimum_value), 1.0) * sqrt(eps(Float64))
        minimum_value -= half_width
        maximum_value += half_width
    end
    width = (maximum_value - minimum_value) / bins
    isfinite(width) && width > 0.0 || throw(OutputError(
        "histogram range is not representable for quantity $(quantity)",
    ))
    counts = zeros(Int, bins)
    for value in samples
        index = value == maximum_value ? bins : floor(Int, (value - minimum_value) / width) + 1
        counts[clamp(index, 1, bins)] += 1
    end

    rows = NamedTuple[]
    total = length(samples)
    for index in 1:bins
        left = minimum_value + (index - 1) * width
        right = index == bins ? maximum_value : minimum_value + index * width
        push!(rows, (
            quantity=String(quantity), output_id=Int(output_id), bin_left=left,
            bin_right=right, bin_center=(left + right) / 2.0, count=counts[index],
            density=counts[index] / (total * width),
        ))
    end
    return rows
end

function _histogram_transaction_part(runctx::RunContext, quantity::AbstractString,
                                     output_id::Integer, values, bins::Integer;
                                     once::Bool=false)
    safe = _csv_filename_stem(quantity)
    key = Symbol("histograms/$(safe)")
    path = joinpath(runctx.run_dir, "histograms", "$(safe).csv")
    rows = _histogram_rows(quantity, output_id, values, bins)
    return _transaction_part(
        runctx, key, rows; path=path, header=copy(_HISTOGRAM_HEADER),
        allow_create=true, once=once,
    )
end

function write_histogram_output!(runctx::RunContext, quantity::AbstractString,
                                 output_id::Integer, values, bins::Integer)
    part = _histogram_transaction_part(runctx, quantity, output_id, values, bins)
    commit_output_transaction!(
        runctx, :histogram, Int(output_id), -1, _OutputTransactionPart[part],
    )
    return part.path
end

function _slice_plane(map3, axis::AbstractString, index::Integer)
    normalized = lowercase(String(axis))
    if normalized == "x"
        1 <= index <= size(map3, 1) || throw(OutputError("slice index is outside x extent"))
        return @view(map3[index, :, :])
    elseif normalized == "y"
        1 <= index <= size(map3, 2) || throw(OutputError("slice index is outside y extent"))
        return @view(map3[:, index, :])
    elseif normalized == "z"
        1 <= index <= size(map3, 3) || throw(OutputError("slice index is outside z extent"))
        return @view(map3[:, :, index])
    end
    throw(OutputError("slice axis must be x, y, or z"))
end

"""Write one 2-D slice file; existing files are never overwritten."""
function _slice_transaction_part(runctx::RunContext, quantity::AbstractString,
                                 output_id::Integer, map3, lattice;
                                 axis::AbstractString="z", index::Integer=1,
                                 coordinate_scale::Real=1.0)
    ndims(map3) == 3 || throw(OutputError("slice source must be a three-dimensional array"))
    plane = _slice_plane(map3, axis, index)
    safe = _csv_filename_stem(quantity)
    filename = "$(safe)_$(lpad(string(output_id), 8, '0'))_$(lowercase(axis))$(index).csv"
    path = joinpath(runctx.run_dir, "slices", filename)
    rows = NamedTuple[]
    spacing = lattice.dx * Float64(coordinate_scale)
    sizehint!(rows, length(plane))
    for second in axes(plane, 2), first_index in axes(plane, 1)
        push!(rows, (
            i=first_index, j=second, x=(first_index - 1) * spacing,
            y=(second - 1) * spacing, value=plane[first_index, second],
        ))
    end
    key = Symbol("slice_$(safe)_$(output_id)_$(lowercase(axis))$(index)")
    return _transaction_part(
        runctx, key, rows; path=path, header=copy(_SLICE_HEADER),
        allow_create=true, transient=true, once=true,
    )
end

function write_slice_output!(runctx::RunContext, quantity::AbstractString,
                             output_id::Integer, map3, lattice;
                             axis::AbstractString="z", index::Integer=1,
                             coordinate_scale::Real=1.0)
    part = _slice_transaction_part(
        runctx, quantity, output_id, map3, lattice;
        axis=axis, index=index, coordinate_scale=coordinate_scale,
    )
    commit_output_transaction!(
        runctx, :slice, Int(output_id), -1, _OutputTransactionPart[part],
    )
    return part.path
end

"""Write one optional full 3-D CSV snapshot."""
function _volume_transaction_part(runctx::RunContext, quantity::AbstractString,
                                  output_id::Integer, map3, lattice;
                                  coordinate_scale::Real=1.0)
    ndims(map3) == 3 || throw(OutputError("snapshot source must be a three-dimensional array"))
    safe = _csv_filename_stem(quantity)
    path = joinpath(runctx.run_dir, "snapshots", "$(safe)_$(lpad(string(output_id), 8, '0')).csv")
    mkpath(dirname(path))
    spacing = lattice.dx * Float64(coordinate_scale)
    rows = NamedTuple[]
    sizehint!(rows, length(map3))
    for k in axes(map3, 3), j in axes(map3, 2), i in axes(map3, 1)
        push!(rows, (
            i=i, j=j, k=k, x=(i - 1) * spacing, y=(j - 1) * spacing,
            z=(k - 1) * spacing, value=map3[i, j, k],
        ))
    end
    key = Symbol("snapshot_$(safe)_$(output_id)")
    return _transaction_part(
        runctx, key, rows; path=path, header=copy(_VOLUME_HEADER),
        allow_create=true, transient=true, once=true,
    )
end

function write_volume_output!(runctx::RunContext, quantity::AbstractString,
                              output_id::Integer, map3, lattice;
                              coordinate_scale::Real=1.0)
    part = _volume_transaction_part(
        runctx, quantity, output_id, map3, lattice; coordinate_scale=coordinate_scale,
    )
    commit_output_transaction!(
        runctx, :snapshot, Int(output_id), -1, _OutputTransactionPart[part],
    )
    return part.path
end

function _configured_slice_transaction_parts(runctx, state, model, lattice, config, output_id)
    parts = _OutputTransactionPart[]
    save_slices = Bool(_io_cfg(config, :output, :save_2d_slices, false))
    save_volumes = Bool(_io_cfg(config, :output, :save_3d_snapshots, false))
    (save_slices || save_volumes) || return parts
    axis = String(_io_cfg(config, :output, :slice_axis, "z"))
    index = Int(_io_cfg(config, :output, :slice_index, cld(lattice.N, 2)))
    coordinate_scale = inv(model.B)
    field_ids = _csv_safe_field_names(String.(collect(model.field_names)))
    for field_index in axes(state.field, 4)
        name = field_ids[field_index]
        map3 = @view(state.field[:, :, :, field_index])
        save_slices && push!(parts, _slice_transaction_part(
            runctx, name, output_id, map3, lattice;
            axis=axis, index=index, coordinate_scale=coordinate_scale,
        ))
        save_volumes && push!(parts, _volume_transaction_part(
            runctx, name, output_id, map3, lattice; coordinate_scale=coordinate_scale,
        ))
    end
    return parts
end

function _deltaN_statistics(map)
    values = _finite_values(map)
    isempty(values) && throw(OutputError("delta-N map contains no finite sites"))
    count_values = length(values)
    mean_value = Statistics.mean(values)
    variance = Statistics.var(values; corrected=false)
    standard_deviation = sqrt(max(variance, 0.0))
    if standard_deviation > 0.0
        skewness = sum(((value - mean_value) / standard_deviation)^3 for value in values) / count_values
        excess = sum(((value - mean_value) / standard_deviation)^4 for value in values) / count_values - 3.0
    else
        skewness = NaN
        excess = NaN
    end
    return (
        mean=mean_value, variance=variance, stddev=standard_deviation,
        skewness=skewness, excess_kurtosis=excess,
        minimum=minimum(values), maximum=maximum(values), n_sites=count_values,
    )
end

function _deltaN_failure_count(result)
    count_value = _io_get_any(result, (:n_failed, :failed_count, :failure_count), nothing)
    count_value === nothing || return Int(count_value)
    failed_indices = _io_get(result, :failed_indices, nothing)
    failed_indices === nothing || return length(failed_indices)
    converged = _io_get_any(result, (:converged, :success_mask, :finished), nothing)
    converged === nothing && return 0
    return count(value -> !Bool(value), converged)
end

"""Write nonlinear delta-N summary, spectrum, histogram, and configured slice."""
function write_deltaN_outputs!(runctx::RunContext, result, state, model, lattice, config)
    zeta = _io_get_any(result, (:zeta, :zeta_deltaN, :deltaN, :map), nothing)
    zeta === nothing && throw(OutputError("delta-N result has no curvature map"))
    ndims(zeta) == 3 || throw(OutputError("delta-N curvature map must be three-dimensional"))
    statistics = _deltaN_statistics(zeta)
    reference_efolds = Float64(_io_get_any(
        result, (:efolds_reference, :reference_efolds), state.efolds,
    ))
    reference_density = Float64(_io_get_any(
        result, (:rho_reference, :reference_density), NaN,
    ))
    ratio = Float64(_io_get_any(
        result, (:separate_universe_ratio, :su_ratio, :R_SU, :separate_universe_R), NaN,
    ))
    ratio_ok = Bool(_io_get_any(
        result, (:separate_universe_ok, :separate_universe_valid, :su_valid),
        isfinite(ratio) && ratio >= Float64(_io_cfg(
            config, :deltaN, :separate_universe_min_ratio, 1.0,
        )),
    ))
    failed = _deltaN_failure_count(result)
    summary = merge((
        efolds_reference=reference_efolds,
        rho_reference=reference_density,
    ), statistics, (
        n_sites=length(zeta),
        n_failed=failed,
        separate_universe_ratio=ratio,
        separate_universe_ok=ratio_ok,
    ))
    spectrum = _io_get_any(result, (:spectrum, :spectrum_rows), nothing)
    spectrum === nothing && (spectrum = scalar_spectrum(zeta, state, model, lattice))
    spectrum_rows = NamedTuple[]
    for row in spectrum
        push!(spectrum_rows, (
            efolds_reference=reference_efolds,
            rho_reference=reference_density,
            k_bin=_io_get(row, :k_bin, nothing),
            k_comoving=_io_get(row, :k_comoving, nothing),
            k_physical=_io_get(row, :k_physical, nothing),
            n_modes=_io_get(row, :n_modes, nothing),
            power_dimensionless=_io_get(row, :power_dimensionless, nothing),
        ))
    end
    output_id = max(runctx.output_id, 0)
    bins = Int(_io_cfg(config, :output, :histogram_bins, 256))
    parts = _OutputTransactionPart[
        _transaction_part(runctx, :deltaN_summary, summary; once=true),
        _transaction_part(runctx, :deltaN_spectrum, spectrum_rows; once=true),
        _histogram_transaction_part(
            runctx, "zeta_deltaN", output_id, zeta, bins; once=true,
        ),
    ]
    if Bool(_io_cfg(config, :output, :save_2d_slices, false))
        push!(parts, _slice_transaction_part(
            runctx, "zeta_deltaN", output_id, zeta, lattice;
            axis=String(_io_cfg(config, :output, :slice_axis, "z")),
            index=Int(_io_cfg(config, :output, :slice_index, cld(lattice.N, 2))),
            coordinate_scale=inv(model.B),
        ))
    end
    Bool(_io_cfg(config, :output, :save_3d_snapshots, false)) && push!(
        parts,
        _volume_transaction_part(
            runctx, "zeta_deltaN", output_id, zeta, lattice;
            coordinate_scale=inv(model.B),
        ),
    )
    commit_output_transaction!(
        runctx, :deltaN, output_id, state.step, parts,
    )

    runctx.metadata["deltaN"] = Dict{String,Any}(
        "efolds_reference" => reference_efolds,
        "rho_reference" => reference_density,
        "n_failed" => failed,
        "separate_universe_ratio" => ratio,
        "separate_universe_ok" => ratio_ok,
    )
    ratio_ok || record_warning!(runctx, "separate-universe applicability condition was not satisfied")
    write_metadata!(runctx)
    return summary
end
