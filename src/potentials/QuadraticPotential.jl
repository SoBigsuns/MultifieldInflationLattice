"""
    QuadraticPotential(masses)

Separable canonical potential `V = ½ Σᴵ mᴵ² φᴵ²`.  `masses` are the
positive masses in solver code units (`m_physical / B`).
"""
struct QuadraticPotential <: AbstractPotential
    masses::Vector{Float64}
    mass_squared::Vector{Float64}

    function QuadraticPotential(masses::AbstractVector{<:Real})
        isempty(masses) && throw(ArgumentError("at least one mass is required"))
        converted = Float64.(masses)
        all(value -> isfinite(value) && value > 0, converted) || throw(ArgumentError(
            "quadratic masses must be finite and positive",
        ))
        return new(converted, abs2.(converted))
    end
end

field_count(potential::QuadraticPotential) = length(potential.masses)

@inline function _check_potential_vector(potential::QuadraticPotential, values, label)
    length(values) == field_count(potential) || throw(DimensionMismatch(
        "$label length $(length(values)) does not match the potential field count " *
        "$(field_count(potential))",
    ))
    return nothing
end

function potential_value(
    potential::QuadraticPotential,
    fields::AbstractVector{<:Real},
)
    _check_potential_vector(potential, fields, "field vector")
    value = 0.0
    @inbounds @simd for index in eachindex(potential.mass_squared, fields)
        value += 0.5 * potential.mass_squared[index] * fields[index]^2
    end
    return value
end

function potential_gradient!(
    out::AbstractVector,
    potential::QuadraticPotential,
    fields::AbstractVector{<:Real},
)
    _check_potential_vector(potential, fields, "field vector")
    _check_potential_vector(potential, out, "gradient output")
    @inbounds @simd for index in eachindex(potential.mass_squared, fields, out)
        out[index] = potential.mass_squared[index] * fields[index]
    end
    return nothing
end

function potential_hessian!(
    out::AbstractMatrix,
    potential::QuadraticPotential,
    fields::AbstractVector{<:Real},
)
    _check_potential_vector(potential, fields, "field vector")
    nf = field_count(potential)
    size(out) == (nf, nf) || throw(DimensionMismatch(
        "Hessian output must have dimensions ($nf, $nf)",
    ))
    fill!(out, zero(eltype(out)))
    @inbounds for index in 1:nf
        out[index, index] = potential.mass_squared[index]
    end
    return nothing
end

function characteristic_mass(
    potential::QuadraticPotential,
    background_fields::AbstractVector{<:Real},
)
    _check_potential_vector(potential, background_fields, "background field vector")
    return maximum(potential.masses)
end
