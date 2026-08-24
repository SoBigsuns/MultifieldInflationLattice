"""
    AbstractPotential

Supertype for analytic scalar-field potentials.  A concrete potential is stored
in code units and must implement [`potential_value`](@ref),
[`potential_gradient!`](@ref), [`potential_hessian!`](@ref), and
[`characteristic_mass`](@ref).
"""
abstract type AbstractPotential end

"""
    potential_value(potential, fields)

Return the potential energy density at one lattice site.  `fields` contains
the field values in configured field order.  Both the potential and its result
are in code units (`V / B^2`).
"""
function potential_value end

"""
    potential_gradient!(out, potential, fields) -> nothing

Write `∂V/∂φᴵ` at one lattice site into the caller-owned vector `out`.
The method must not allocate in the lattice-site inner loop.
"""
function potential_gradient! end

"""
    potential_hessian!(out, potential, fields) -> nothing

Write `∂²V/(∂φᴵ∂φʲ)` into the caller-owned square matrix `out`.
"""
function potential_hessian! end

"""
    characteristic_mass(potential, background_fields)

Return a conservative characteristic mass in code units.  It is used when
selecting a stable time step.  `background_fields` is accepted so that
field-dependent effective masses can be implemented by future potentials.
"""
function characteristic_mass end

"""
    field_count(potential)

Return the number of real scalar fields expected by `potential`.
"""
function field_count end
