"""All configuration errors discovered before a simulation starts."""
struct ConfigError <: Exception
    errors::Vector{String}
end

function Base.showerror(io::IO, error::ConfigError)
    print(io, "invalid simulation configuration")
    for message in error.errors
        print(io, "\n  - ", message)
    end
end

const _CONFIG_TABLE_KEYS = Set((
    "model", "lattice", "initial", "evolution", "deltaN", "spectra",
    "output", "parallel",
))
const _MODEL_KEYS = Set((
    "field_names", "potential", "masses", "reduced_planck_mass", "rescale_B",
))
const _LATTICE_KEYS = Set(("size", "box_size", "boundary", "laplacian"))
const _INITIAL_KEYS = Set((
    "field_means", "cosmic_velocities", "scale_factor", "seed",
    "vacuum_fluctuations", "include_effective_mass", "low_cutoff_index",
    "high_cutoff_index",
))
const _EVOLUTION_KEYS = Set((
    "integrator", "timestep_factor", "end_epsilon_H", "max_efolds",
    "max_steps", "max_wall_time", "friedmann_warning_tolerance",
    "friedmann_error_tolerance",
))
const _DELTAN_KEYS = Set((
    "enabled", "integrator", "reference_surface", "dN", "max_extra_efolds",
    "max_backward_efolds", "separate_universe_min_ratio",
    "strict_separate_universe",
))
const _SPECTRA_KEYS = Set((
    "enabled", "use_lattice_momentum", "linear_zeta_min_speed2",
))
const _OUTPUT_KEYS = Set((
    "root", "run_name", "background_interval_efolds", "spectra_interval_efolds",
    "histogram_interval_efolds", "checkpoint_interval_efolds", "histogram_bins",
    "save_2d_slices", "slice_axis", "slice_index", "save_3d_snapshots",
    "overwrite",
))
const _PARALLEL_KEYS = Set((
    "backend", "fft_threads", "deterministic_reductions",
))

"""Return a fresh configuration containing every documented default."""
function default_config()
    return SimulationConfig(
        ModelConfig(
            ["phi", "psi"],
            "quadratic",
            [9.0e-6, 1.0e-6],
            1.0,
            "auto",
        ),
        LatticeConfig(32, 2.0, "periodic", "second_order_7point"),
        InitialConfig(
            [13.0, 13.0],
            [1.0e-10, 0.0],
            1.0,
            UInt64(8),
            true,
            false,
            0.0,
            0.0,
        ),
        EvolutionConfig(
            "leapfrog",
            0.01,
            1.0,
            100.0,
            100_000_000,
            0.0,
            1.0e-6,
            1.0e-3,
        ),
        DeltaNConfig(
            true,
            "leapfrog",
            "uniform_density_at_end",
            1.0e-4,
            10.0,
            1.0,
            1.0,
            false,
        ),
        SpectraConfig(true, true, 1.0e-30),
        OutputConfig(
            "runs",
            "two_field_quadratic",
            0.01,
            0.01,
            0.01,
            0.5,
            256,
            true,
            "z",
            16,
            false,
            false,
        ),
        ParallelConfig("cpu_threads", 1, false),
    )
end

function _unknown_keys!(errors::Vector{String}, table::AbstractDict, allowed, path::String)
    unknown = sort!(String[string(key) for key in keys(table) if string(key) ∉ allowed])
    for key in unknown
        push!(errors, "$path.$key is not a recognized key")
    end
    return nothing
end

function _table(
    data::AbstractDict,
    key::String,
    errors::Vector{String},
)
    value = get(data, key, nothing)
    value === nothing && return Dict{String,Any}()
    if !(value isa AbstractDict)
        push!(errors, "$key must be a TOML table")
        return Dict{String,Any}()
    end
    return value
end

function _string_value(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa AbstractString)
        push!(errors, "$path.$key must be a string")
        return default
    end
    return String(value)
end

function _bool_value(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa Bool)
        push!(errors, "$path.$key must be a Boolean")
        return default
    end
    return value
end

function _float_value(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa Real) || value isa Bool
        push!(errors, "$path.$key must be a finite number")
        return default
    end
    converted = try
        Float64(value)
    catch
        push!(errors, "$path.$key cannot be represented as Float64")
        return default
    end
    if !isfinite(converted)
        push!(errors, "$path.$key must be finite")
        return default
    end
    return converted
end

function _int_value(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa Integer) || value isa Bool
        push!(errors, "$path.$key must be an integer")
        return default
    end
    return try
        Int(value)
    catch
        push!(errors, "$path.$key cannot be represented as Int")
        default
    end
end

function _seed_value(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa Integer) || value isa Bool || value < 0
        push!(errors, "$path.$key must be a non-negative integer convertible to UInt64")
        return default
    end
    return try
        UInt64(value)
    catch
        push!(errors, "$path.$key cannot be represented as UInt64")
        default
    end
end

function _string_vector(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa AbstractVector)
        push!(errors, "$path.$key must be an array of strings")
        return copy(default)
    end
    if any(item -> !(item isa AbstractString), value)
        push!(errors, "$path.$key must contain only strings")
        return copy(default)
    end
    return String[String(item) for item in value]
end

function _float_vector(table, key, default, errors, path)
    value = get(table, key, default)
    if !(value isa AbstractVector)
        push!(errors, "$path.$key must be an array of finite numbers")
        return copy(default)
    end
    result = Float64[]
    sizehint!(result, length(value))
    valid = true
    for (index, item) in pairs(value)
        if !(item isa Real) || item isa Bool
            push!(errors, "$path.$key[$index] must be a finite number")
            valid = false
            continue
        end
        converted = try
            Float64(item)
        catch
            push!(errors, "$path.$key[$index] cannot be represented as Float64")
            valid = false
            continue
        end
        if !isfinite(converted)
            push!(errors, "$path.$key[$index] must be finite")
            valid = false
        end
        push!(result, converted)
    end
    return valid ? result : copy(default)
end

function _rescale_value(table, default, errors)
    value = get(table, "rescale_B", default)
    if value isa AbstractString
        value == "auto" || push!(errors, "model.rescale_B must be \"auto\" or a finite positive number")
        return value == "auto" ? "auto" : default
    end
    if value isa Real && !(value isa Bool)
        converted = try
            Float64(value)
        catch
            push!(errors, "model.rescale_B cannot be represented as Float64")
            return default
        end
        if !(isfinite(converted) && converted > 0)
            push!(errors, "model.rescale_B must be finite and positive")
            return default
        end
        return converted
    end
    push!(errors, "model.rescale_B must be \"auto\" or a finite positive number")
    return default
end

"""
    config_from_dict(data) -> SimulationConfig

Apply defaults, reject unknown keys and type mismatches, and return a fully
validated typed configuration.  All discovered problems are reported together
in one [`ConfigError`](@ref).
"""
function config_from_dict(data::AbstractDict)
    defaults = default_config()
    errors = String[]
    _unknown_keys!(errors, data, _CONFIG_TABLE_KEYS, "configuration")

    model_data = _table(data, "model", errors)
    lattice_data = _table(data, "lattice", errors)
    initial_data = _table(data, "initial", errors)
    evolution_data = _table(data, "evolution", errors)
    deltaN_data = _table(data, "deltaN", errors)
    spectra_data = _table(data, "spectra", errors)
    output_data = _table(data, "output", errors)
    parallel_data = _table(data, "parallel", errors)

    _unknown_keys!(errors, model_data, _MODEL_KEYS, "model")
    _unknown_keys!(errors, lattice_data, _LATTICE_KEYS, "lattice")
    _unknown_keys!(errors, initial_data, _INITIAL_KEYS, "initial")
    _unknown_keys!(errors, evolution_data, _EVOLUTION_KEYS, "evolution")
    _unknown_keys!(errors, deltaN_data, _DELTAN_KEYS, "deltaN")
    _unknown_keys!(errors, spectra_data, _SPECTRA_KEYS, "spectra")
    _unknown_keys!(errors, output_data, _OUTPUT_KEYS, "output")
    _unknown_keys!(errors, parallel_data, _PARALLEL_KEYS, "parallel")

    dm = defaults.model
    model = ModelConfig(
        _string_vector(model_data, "field_names", dm.field_names, errors, "model"),
        _string_value(model_data, "potential", dm.potential, errors, "model"),
        _float_vector(model_data, "masses", dm.masses, errors, "model"),
        _float_value(model_data, "reduced_planck_mass", dm.reduced_planck_mass, errors, "model"),
        _rescale_value(model_data, dm.rescale_B, errors),
    )

    dl = defaults.lattice
    lattice = LatticeConfig(
        _int_value(lattice_data, "size", dl.size, errors, "lattice"),
        _float_value(lattice_data, "box_size", dl.box_size, errors, "lattice"),
        _string_value(lattice_data, "boundary", dl.boundary, errors, "lattice"),
        _string_value(lattice_data, "laplacian", dl.laplacian, errors, "lattice"),
    )

    di = defaults.initial
    initial = InitialConfig(
        _float_vector(initial_data, "field_means", di.field_means, errors, "initial"),
        _float_vector(initial_data, "cosmic_velocities", di.cosmic_velocities, errors, "initial"),
        _float_value(initial_data, "scale_factor", di.scale_factor, errors, "initial"),
        _seed_value(initial_data, "seed", di.seed, errors, "initial"),
        _bool_value(initial_data, "vacuum_fluctuations", di.vacuum_fluctuations, errors, "initial"),
        _bool_value(initial_data, "include_effective_mass", di.include_effective_mass, errors, "initial"),
        _float_value(initial_data, "low_cutoff_index", di.low_cutoff_index, errors, "initial"),
        _float_value(initial_data, "high_cutoff_index", di.high_cutoff_index, errors, "initial"),
    )

    de = defaults.evolution
    evolution = EvolutionConfig(
        _string_value(evolution_data, "integrator", de.integrator, errors, "evolution"),
        _float_value(evolution_data, "timestep_factor", de.timestep_factor, errors, "evolution"),
        _float_value(evolution_data, "end_epsilon_H", de.end_epsilon_H, errors, "evolution"),
        _float_value(evolution_data, "max_efolds", de.max_efolds, errors, "evolution"),
        _int_value(evolution_data, "max_steps", de.max_steps, errors, "evolution"),
        _float_value(evolution_data, "max_wall_time", de.max_wall_time, errors, "evolution"),
        _float_value(evolution_data, "friedmann_warning_tolerance", de.friedmann_warning_tolerance, errors, "evolution"),
        _float_value(evolution_data, "friedmann_error_tolerance", de.friedmann_error_tolerance, errors, "evolution"),
    )

    dd = defaults.deltaN
    deltaN = DeltaNConfig(
        _bool_value(deltaN_data, "enabled", dd.enabled, errors, "deltaN"),
        _string_value(deltaN_data, "integrator", dd.integrator, errors, "deltaN"),
        _string_value(deltaN_data, "reference_surface", dd.reference_surface, errors, "deltaN"),
        _float_value(deltaN_data, "dN", dd.dN, errors, "deltaN"),
        _float_value(deltaN_data, "max_extra_efolds", dd.max_extra_efolds, errors, "deltaN"),
        _float_value(deltaN_data, "max_backward_efolds", dd.max_backward_efolds, errors, "deltaN"),
        _float_value(deltaN_data, "separate_universe_min_ratio", dd.separate_universe_min_ratio, errors, "deltaN"),
        _bool_value(deltaN_data, "strict_separate_universe", dd.strict_separate_universe, errors, "deltaN"),
    )

    ds = defaults.spectra
    spectra = SpectraConfig(
        _bool_value(spectra_data, "enabled", ds.enabled, errors, "spectra"),
        _bool_value(spectra_data, "use_lattice_momentum", ds.use_lattice_momentum, errors, "spectra"),
        _float_value(spectra_data, "linear_zeta_min_speed2", ds.linear_zeta_min_speed2, errors, "spectra"),
    )

    dout = defaults.output
    # Keep the default slice at the middle of a user-selected lattice.  The
    # documented value 16 is the resulting default for the standard N=32.
    default_slice_index = max(1, fld(lattice.size, 2))
    output = OutputConfig(
        _string_value(output_data, "root", dout.root, errors, "output"),
        _string_value(output_data, "run_name", dout.run_name, errors, "output"),
        _float_value(output_data, "background_interval_efolds", dout.background_interval_efolds, errors, "output"),
        _float_value(output_data, "spectra_interval_efolds", dout.spectra_interval_efolds, errors, "output"),
        _float_value(output_data, "histogram_interval_efolds", dout.histogram_interval_efolds, errors, "output"),
        _float_value(output_data, "checkpoint_interval_efolds", dout.checkpoint_interval_efolds, errors, "output"),
        _int_value(output_data, "histogram_bins", dout.histogram_bins, errors, "output"),
        _bool_value(output_data, "save_2d_slices", dout.save_2d_slices, errors, "output"),
        _string_value(output_data, "slice_axis", dout.slice_axis, errors, "output"),
        _int_value(output_data, "slice_index", default_slice_index, errors, "output"),
        _bool_value(output_data, "save_3d_snapshots", dout.save_3d_snapshots, errors, "output"),
        _bool_value(output_data, "overwrite", dout.overwrite, errors, "output"),
    )

    dp = defaults.parallel
    parallel = ParallelConfig(
        _string_value(parallel_data, "backend", dp.backend, errors, "parallel"),
        _int_value(parallel_data, "fft_threads", dp.fft_threads, errors, "parallel"),
        _bool_value(parallel_data, "deterministic_reductions", dp.deterministic_reductions, errors, "parallel"),
    )

    config = SimulationConfig(model, lattice, initial, evolution, deltaN, spectra, output, parallel)
    append!(errors, config_errors(config))
    unique!(errors)
    isempty(errors) || throw(ConfigError(errors))
    return config
end

"""Return every semantic validation error in `config`, without throwing."""
function config_errors(config::SimulationConfig)
    errors = String[]
    model = config.model
    lattice = config.lattice
    initial = config.initial
    evolution = config.evolution
    deltaN = config.deltaN
    spectra = config.spectra
    output = config.output
    parallel = config.parallel

    isempty(model.field_names) && push!(errors, "model.field_names must not be empty")
    any(isempty, model.field_names) && push!(errors, "model.field_names must not contain empty names")
    length(unique(model.field_names)) == length(model.field_names) ||
        push!(errors, "model.field_names must not contain duplicates")
    supported_potentials = if isdefined(@__MODULE__, :available_potentials)
        available_potentials()
    else
        ["quadratic"]
    end
    model.potential in supported_potentials || push!(errors,
        "model.potential must name a registered potential (available: " *
        join(supported_potentials, ", ") * ")",
    )
    length(model.masses) == length(model.field_names) ||
        push!(errors, "model.masses length must equal model.field_names length")
    all(value -> isfinite(value) && value > 0, model.masses) ||
        push!(errors, "every model mass must be finite and positive")
    isfinite(model.reduced_planck_mass) && model.reduced_planck_mass == 1.0 ||
        push!(errors, "model.reduced_planck_mass must equal 1.0 in the current reduced-Planck convention")
    if model.rescale_B isa Float64
        isfinite(model.rescale_B) && model.rescale_B > 0 ||
            push!(errors, "model.rescale_B must be finite and positive")
    elseif model.rescale_B != "auto"
        push!(errors, "model.rescale_B must be \"auto\" or a finite positive number")
    end
    model.potential != "quadratic" && model.rescale_B == "auto" &&
        push!(errors, "model.rescale_B must be explicit for non-quadratic potentials")

    lattice.size >= 8 || push!(errors, "lattice.size must be at least 8")
    ispow2(lattice.size) || push!(errors, "lattice.size must be a power of two")
    isfinite(lattice.box_size) && lattice.box_size > 0 ||
        push!(errors, "lattice.box_size must be finite and positive")
    lattice.boundary == "periodic" ||
        push!(errors, "lattice.boundary must be \"periodic\"")
    lattice.laplacian == "second_order_7point" ||
        push!(errors, "lattice.laplacian must be \"second_order_7point\"")

    nf = length(model.field_names)
    length(initial.field_means) == nf ||
        push!(errors, "initial.field_means length must equal the field count")
    length(initial.cosmic_velocities) == nf ||
        push!(errors, "initial.cosmic_velocities length must equal the field count")
    all(isfinite, initial.field_means) ||
        push!(errors, "initial.field_means must contain only finite values")
    all(isfinite, initial.cosmic_velocities) ||
        push!(errors, "initial.cosmic_velocities must contain only finite values")
    isfinite(initial.scale_factor) && initial.scale_factor > 0 ||
        push!(errors, "initial.scale_factor must be finite and positive")
    isfinite(initial.low_cutoff_index) && initial.low_cutoff_index >= 0 ||
        push!(errors, "initial.low_cutoff_index must be finite and non-negative")
    isfinite(initial.high_cutoff_index) && initial.high_cutoff_index >= 0 ||
        push!(errors, "initial.high_cutoff_index must be finite and non-negative")
    initial.high_cutoff_index == 0 ||
        initial.high_cutoff_index >= initial.low_cutoff_index ||
        push!(errors, "initial.high_cutoff_index must be zero or at least low_cutoff_index")

    evolution.integrator in ("leapfrog", "rk4") ||
        push!(errors, "evolution.integrator must be \"leapfrog\" or \"rk4\"")
    isfinite(evolution.timestep_factor) && evolution.timestep_factor > 0 ||
        push!(errors, "evolution.timestep_factor must be finite and positive")
    isfinite(evolution.end_epsilon_H) && evolution.end_epsilon_H > 0 ||
        push!(errors, "evolution.end_epsilon_H must be finite and positive")
    isfinite(evolution.max_efolds) && evolution.max_efolds > 0 ||
        push!(errors, "evolution.max_efolds must be finite and positive")
    evolution.max_steps > 0 || push!(errors, "evolution.max_steps must be positive")
    isfinite(evolution.max_wall_time) && evolution.max_wall_time >= 0 ||
        push!(errors, "evolution.max_wall_time must be finite and non-negative (zero disables it)")
    isfinite(evolution.friedmann_warning_tolerance) &&
        evolution.friedmann_warning_tolerance > 0 ||
        push!(errors, "evolution.friedmann_warning_tolerance must be finite and positive")
    isfinite(evolution.friedmann_error_tolerance) &&
        evolution.friedmann_error_tolerance > evolution.friedmann_warning_tolerance ||
        push!(errors, "friedmann_error_tolerance must exceed friedmann_warning_tolerance")

    deltaN.integrator in ("leapfrog", "rk4") ||
        push!(errors, "deltaN.integrator must be \"leapfrog\" or \"rk4\"")
    deltaN.reference_surface == "uniform_density_at_end" ||
        push!(errors, "deltaN.reference_surface must be \"uniform_density_at_end\"")
    isfinite(deltaN.dN) && deltaN.dN > 0 ||
        push!(errors, "deltaN.dN must be finite and positive")
    isfinite(deltaN.max_extra_efolds) && deltaN.max_extra_efolds > 0 ||
        push!(errors, "deltaN.max_extra_efolds must be finite and positive")
    isfinite(deltaN.max_backward_efolds) && deltaN.max_backward_efolds >= 0 ||
        push!(errors, "deltaN.max_backward_efolds must be finite and non-negative")
    isfinite(deltaN.separate_universe_min_ratio) &&
        deltaN.separate_universe_min_ratio > 0 ||
        push!(errors, "deltaN.separate_universe_min_ratio must be finite and positive")

    isfinite(spectra.linear_zeta_min_speed2) && spectra.linear_zeta_min_speed2 > 0 ||
        push!(errors, "spectra.linear_zeta_min_speed2 must be finite and positive")

    isempty(strip(output.root)) && push!(errors, "output.root must not be empty")
    isempty(strip(output.run_name)) && push!(errors, "output.run_name must not be empty")
    occursin(r"[\\/]", output.run_name) &&
        push!(errors, "output.run_name must not contain a path separator")
    output.run_name in (".", "..") &&
        push!(errors, "output.run_name must not be a relative-directory token")
    for (name, value) in (
        ("background_interval_efolds", output.background_interval_efolds),
        ("spectra_interval_efolds", output.spectra_interval_efolds),
        ("histogram_interval_efolds", output.histogram_interval_efolds),
        ("checkpoint_interval_efolds", output.checkpoint_interval_efolds),
    )
        isfinite(value) && value > 0 ||
            push!(errors, "output.$name must be finite and positive")
    end
    output.histogram_bins >= 2 || push!(errors, "output.histogram_bins must be at least 2")
    output.slice_axis in ("x", "y", "z") ||
        push!(errors, "output.slice_axis must be \"x\", \"y\", or \"z\"")
    1 <= output.slice_index <= lattice.size ||
        push!(errors, "output.slice_index must be in 1:lattice.size")

    parallel.backend == "cpu_threads" ||
        push!(errors, "parallel.backend must be \"cpu_threads\"")
    parallel.fft_threads > 0 || push!(errors, "parallel.fft_threads must be positive")
    return errors
end

"""Validate an already typed configuration and return it unchanged."""
function validate_config(config::SimulationConfig)
    errors = config_errors(config)
    isempty(errors) || throw(ConfigError(errors))
    return config
end

"""Parse and validate a TOML configuration file."""
function load_config(path::AbstractString)
    isfile(path) || throw(ConfigError(["configuration file does not exist: $path"]))
    data = try
        TOML.parsefile(path)
    catch error
        error isa InterruptException && rethrow()
        throw(ConfigError(["could not parse TOML file $path: $(sprint(showerror, error))"]))
    end
    return config_from_dict(data)
end

"""Convert a validated configuration to a TOML-compatible effective dictionary."""
function effective_config_dict(config::SimulationConfig)
    validate_config(config)
    return Dict{String,Any}(
        "model" => Dict{String,Any}(
            "field_names" => copy(config.model.field_names),
            "potential" => config.model.potential,
            "masses" => copy(config.model.masses),
            "reduced_planck_mass" => config.model.reduced_planck_mass,
            "rescale_B" => config.model.rescale_B,
        ),
        "lattice" => Dict{String,Any}(
            "size" => config.lattice.size,
            "box_size" => config.lattice.box_size,
            "boundary" => config.lattice.boundary,
            "laplacian" => config.lattice.laplacian,
        ),
        "initial" => Dict{String,Any}(
            "field_means" => copy(config.initial.field_means),
            "cosmic_velocities" => copy(config.initial.cosmic_velocities),
            "scale_factor" => config.initial.scale_factor,
            "seed" => config.initial.seed,
            "vacuum_fluctuations" => config.initial.vacuum_fluctuations,
            "include_effective_mass" => config.initial.include_effective_mass,
            "low_cutoff_index" => config.initial.low_cutoff_index,
            "high_cutoff_index" => config.initial.high_cutoff_index,
        ),
        "evolution" => Dict{String,Any}(
            "integrator" => config.evolution.integrator,
            "timestep_factor" => config.evolution.timestep_factor,
            "end_epsilon_H" => config.evolution.end_epsilon_H,
            "max_efolds" => config.evolution.max_efolds,
            "max_steps" => config.evolution.max_steps,
            "max_wall_time" => config.evolution.max_wall_time,
            "friedmann_warning_tolerance" => config.evolution.friedmann_warning_tolerance,
            "friedmann_error_tolerance" => config.evolution.friedmann_error_tolerance,
        ),
        "deltaN" => Dict{String,Any}(
            "enabled" => config.deltaN.enabled,
            "integrator" => config.deltaN.integrator,
            "reference_surface" => config.deltaN.reference_surface,
            "dN" => config.deltaN.dN,
            "max_extra_efolds" => config.deltaN.max_extra_efolds,
            "max_backward_efolds" => config.deltaN.max_backward_efolds,
            "separate_universe_min_ratio" => config.deltaN.separate_universe_min_ratio,
            "strict_separate_universe" => config.deltaN.strict_separate_universe,
        ),
        "spectra" => Dict{String,Any}(
            "enabled" => config.spectra.enabled,
            "use_lattice_momentum" => config.spectra.use_lattice_momentum,
            "linear_zeta_min_speed2" => config.spectra.linear_zeta_min_speed2,
        ),
        "output" => Dict{String,Any}(
            "root" => config.output.root,
            "run_name" => config.output.run_name,
            "background_interval_efolds" => config.output.background_interval_efolds,
            "spectra_interval_efolds" => config.output.spectra_interval_efolds,
            "histogram_interval_efolds" => config.output.histogram_interval_efolds,
            "checkpoint_interval_efolds" => config.output.checkpoint_interval_efolds,
            "histogram_bins" => config.output.histogram_bins,
            "save_2d_slices" => config.output.save_2d_slices,
            "slice_axis" => config.output.slice_axis,
            "slice_index" => config.output.slice_index,
            "save_3d_snapshots" => config.output.save_3d_snapshots,
            "overwrite" => config.output.overwrite,
        ),
        "parallel" => Dict{String,Any}(
            "backend" => config.parallel.backend,
            "fft_threads" => config.parallel.fft_threads,
            "deterministic_reductions" => config.parallel.deterministic_reductions,
        ),
    )
end
