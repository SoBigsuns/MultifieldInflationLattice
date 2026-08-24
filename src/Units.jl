"""Code-unit scale `B` and conversions to/from reduced-Planck units."""
struct CodeUnits
    B::Float64

    function CodeUnits(B::Real)
        value = Float64(B)
        isfinite(value) && value > 0 || throw(ArgumentError(
            "the code-unit scale B must be finite and positive",
        ))
        return new(value)
    end
end

to_code_time(value::Real, units::CodeUnits) = Float64(value) * units.B
to_planck_time(value::Real, units::CodeUnits) = Float64(value) / units.B
to_code_length(value::Real, units::CodeUnits) = Float64(value) * units.B
to_planck_length(value::Real, units::CodeUnits) = Float64(value) / units.B
to_code_rate(value::Real, units::CodeUnits) = Float64(value) / units.B
to_planck_rate(value::Real, units::CodeUnits) = Float64(value) * units.B
to_code_mass(value::Real, units::CodeUnits) = Float64(value) / units.B
to_planck_mass(value::Real, units::CodeUnits) = Float64(value) * units.B
to_code_density(value::Real, units::CodeUnits) = Float64(value) / units.B^2
to_planck_density(value::Real, units::CodeUnits) = Float64(value) * units.B^2

"""
    resolve_rescaling_B(config)

Resolve `[model].rescale_B`.  For the supported quadratic model, `"auto"`
means `sqrt(V(φ̄(0)))` evaluated using physical masses in reduced-Planck
units, exactly as specified by MFIL-SPEC-001 §5.
"""
function resolve_rescaling_B(config::SimulationConfig)
    return resolve_rescaling_B(config.model, config.initial)
end

function resolve_rescaling_B(model::ModelConfig, initial::InitialConfig)
    requested = model.rescale_B
    requested isa Float64 && return requested
    requested == "auto" || throw(ArgumentError(
        "unsupported rescale_B value $(repr(requested))",
    ))

    model.potential == "quadratic" || throw(ArgumentError(
        "automatic B rescaling is not implemented for potential " *
        repr(model.potential),
    ))
    masses = model.masses
    means = initial.field_means
    length(masses) == length(means) || throw(DimensionMismatch(
        "masses and initial field means must have the same length",
    ))

    potential = 0.0
    @inbounds @simd for i in eachindex(masses, means)
        potential += 0.5 * abs2(masses[i] * means[i])
    end
    B = sqrt(potential)
    isfinite(B) && B > 0 || throw(ArgumentError(
        "automatic B rescaling produced a non-positive or non-finite scale",
    ))
    return B
end
