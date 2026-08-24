const _POTENTIAL_BUILDERS = Dict{String,Function}(
    "quadratic" => (model, B) -> QuadraticPotential(model.masses ./ B),
)

"""Names of all potential implementations available in this build."""
available_potentials() = sort!(collect(keys(_POTENTIAL_BUILDERS)))

"""
    register_potential!(name, builder; replace=false)

Register a trusted in-package potential constructor.  `builder(model_config,
B)` must return an [`AbstractPotential`](@ref) in code units.  Arbitrary code
loading from run configuration is intentionally unsupported.
"""
function register_potential!(
    name::AbstractString,
    builder::Function;
    replace::Bool=false,
)
    normalized = lowercase(strip(String(name)))
    isempty(normalized) && throw(ArgumentError("potential name must not be empty"))
    if haskey(_POTENTIAL_BUILDERS, normalized) && !replace
        throw(ArgumentError("potential $normalized is already registered"))
    end
    _POTENTIAL_BUILDERS[normalized] = builder
    return normalized
end

"""Construct the selected code-unit potential using an already resolved `B`."""
function build_potential(config::ModelConfig, B::Real)
    scale = Float64(B)
    isfinite(scale) && scale > 0 || throw(ArgumentError(
        "B must be finite and positive when constructing a potential",
    ))
    name = lowercase(strip(config.potential))
    builder = get(_POTENTIAL_BUILDERS, name, nothing)
    builder === nothing && throw(ArgumentError(
        "unknown potential $(repr(config.potential)); available potentials: " *
        join(available_potentials(), ", "),
    ))
    potential = builder(config, scale)
    potential isa AbstractPotential || throw(ArgumentError(
        "registered builder for $name did not return an AbstractPotential",
    ))
    field_count(potential) == length(config.field_names) || throw(DimensionMismatch(
        "potential field count differs from model.field_names",
    ))
    return potential
end

function build_potential(config::SimulationConfig)
    B = resolve_rescaling_B(config)
    return build_potential(config.model, B)
end

build_potential(config::SimulationConfig, B::Real) =
    build_potential(config.model, B)

"""Build immutable model metadata and the selected code-unit potential."""
function build_model(config::SimulationConfig)
    validate_config(config)
    B = resolve_rescaling_B(config)
    potential = build_potential(config.model, B)
    return SimulationModel(
        potential,
        B,
        config.initial.scale_factor,
        copy(config.model.field_names),
        config.model.reduced_planck_mass,
    )
end
