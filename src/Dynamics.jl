"""
Reusable storage for force and energy evaluation.  The first three array axes
are spatial and the final axis labels fields.
"""
mutable struct DynamicsWorkspace{T}
    laplacian::Array{T,4}
    site_fields::Matrix{T}
    site_gradient::Matrix{T}
    kinetic_partial::Matrix{T}
    gradient_partial::Matrix{T}
    potential_partial::Vector{T}
    deterministic_reductions::Bool
end

function DynamicsWorkspace(
    state::SimulationState{T};
    deterministic_reductions::Bool=false,
) where {T}
    nf = size(state.field, 4)
    n = size(state.field, 3)
    nt = Threads.maxthreadid()
    reduction_slots = max(n, nt)
    return DynamicsWorkspace(
        similar(state.field),
        zeros(T, nf, nt),
        zeros(T, nf, nt),
        zeros(T, nf, reduction_slots),
        zeros(T, nf, reduction_slots),
        zeros(T, reduction_slots),
        deterministic_reductions,
    )
end

@inline hubble_code(state::SimulationState) = state.a_prime / state.a^2
@inline conformal_hubble(state::SimulationState) = state.a_prime / state.a

"""
    dynamics_rhs!(dfield, dprime, field, field_prime, a, a_prime,
                  model, lattice, workspace)

Evaluate the code-unit first-order Klein--Gordon/FLRW system.  The returned
named tuple contains `da`, `da_prime`, `rho`, and `p`.  Potential values and
spatial derivatives are code-unit quantities; fields themselves remain in
reduced-Planck units.
"""
function dynamics_rhs!(
    dfield::Array{T,4},
    dprime::Array{T,4},
    field::Array{T,4},
    field_prime::Array{T,4},
    a::T,
    a_prime::T,
    model::SimulationModel,
    lattice::Lattice3D,
    workspace::DynamicsWorkspace{T},
) where {T<:AbstractFloat}
    a > zero(T) || throw(NumericalError("scale factor became non-positive"))
    laplacian!(workspace.laplacian, field, lattice)
    fill!(workspace.kinetic_partial, zero(T))
    fill!(workspace.gradient_partial, zero(T))
    fill!(workspace.potential_partial, zero(T))

    nf = size(field, 4)
    n = lattice.N
    inva2 = inv(a * a)
    friction = 2 * a_prime / a

    Threads.@threads :static for k in 1:n
        tid = Threads.threadid()
        # Validation mode accumulates one independent partial per z-slab.
        # The final merge is therefore ordered by k and independent of the
        # number of Julia threads.  Normal mode retains cheaper thread-local
        # partials.
        reduction_slot = workspace.deterministic_reductions ? k : tid
        local_fields = @view workspace.site_fields[:, tid]
        local_gradient = @view workspace.site_gradient[:, tid]
        @inbounds for j in 1:n, i in 1:n
            for f in 1:nf
                local_fields[f] = field[i, j, k, f]
            end
            potential_gradient!(local_gradient, model.potential, local_fields)
            workspace.potential_partial[reduction_slot] +=
                potential_value(model.potential, local_fields)
            for f in 1:nf
                value = field[i, j, k, f]
                velocity = field_prime[i, j, k, f]
                lap = workspace.laplacian[i, j, k, f]
                dfield[i, j, k, f] = velocity
                dprime[i, j, k, f] = lap - friction * velocity - a^2 * local_gradient[f]
                workspace.kinetic_partial[f, reduction_slot] +=
                    T(0.5) * velocity^2 * inva2
                workspace.gradient_partial[f, reduction_slot] +=
                    -T(0.5) * value * lap * inva2
            end
        end
    end

    nsites = T(n^3)
    kinetic = sum(workspace.kinetic_partial) / nsites
    gradient = sum(workspace.gradient_partial) / nsites
    potential = sum(workspace.potential_partial) / nsites
    rho = kinetic + gradient + potential
    pressure = kinetic - gradient / 3 - potential
    da_prime = a^3 * (rho - 3pressure) / 6

    all(isfinite, (rho, pressure, da_prime)) ||
        throw(NumericalError("non-finite value while evaluating the equations of motion"))
    rho >= zero(T) || throw(NumericalError("negative mean energy density: $rho"))
    return (; da=a_prime, da_prime, rho, p=pressure)
end

"""Evaluate per-field and total code-unit energy components without mutation."""
function energy_components(
    state::SimulationState{T},
    model::SimulationModel,
    lattice::Lattice3D,
    workspace::DynamicsWorkspace{T}=DynamicsWorkspace(state),
) where {T<:AbstractFloat}
    laplacian!(workspace.laplacian, state.field, lattice)
    fill!(workspace.kinetic_partial, zero(T))
    fill!(workspace.gradient_partial, zero(T))
    fill!(workspace.potential_partial, zero(T))
    nf = size(state.field, 4)
    n = lattice.N
    inva2 = inv(state.a^2)

    Threads.@threads :static for k in 1:n
        tid = Threads.threadid()
        reduction_slot = workspace.deterministic_reductions ? k : tid
        local_fields = @view workspace.site_fields[:, tid]
        @inbounds for j in 1:n, i in 1:n
            for f in 1:nf
                local_fields[f] = state.field[i, j, k, f]
            end
            workspace.potential_partial[reduction_slot] +=
                potential_value(model.potential, local_fields)
            for f in 1:nf
                velocity = state.field_prime[i, j, k, f]
                workspace.kinetic_partial[f, reduction_slot] +=
                    T(0.5) * velocity^2 * inva2
                workspace.gradient_partial[f, reduction_slot] +=
                    -T(0.5) * state.field[i, j, k, f] * workspace.laplacian[i, j, k, f] * inva2
            end
        end
    end

    nsites = T(n^3)
    kinetic_by_field = vec(sum(workspace.kinetic_partial; dims=2)) ./ nsites
    gradient_by_field = vec(sum(workspace.gradient_partial; dims=2)) ./ nsites
    potential = sum(workspace.potential_partial) / nsites
    kinetic = sum(kinetic_by_field)
    gradient = sum(gradient_by_field)
    rho = kinetic + gradient + potential
    pressure = kinetic - gradient / 3 - potential
    return (;
        kinetic_by_field,
        gradient_by_field,
        kinetic,
        gradient,
        potential,
        rho,
        p=pressure,
        w=iszero(rho) ? T(NaN) : pressure / rho,
    )
end

"""Relative Friedmann constraint residual in the convention of MFIL-SPEC-001."""
function friedmann_residual(state::SimulationState, rho_code::Real; rho_floor::Real=floatmin(Float64))
    evolved = hubble_code(state)^2
    constrained = rho_code / 3
    return abs(evolved - constrained) / max(evolved, constrained, rho_floor)
end

"""
Choose a conformal-time step from Hubble, ultraviolet lattice and potential
mass scales, then enforce the three-dimensional Courant bound.
"""
function choose_dtau(
    state::SimulationState{T},
    model::SimulationModel,
    lattice::Lattice3D,
    config::SimulationConfig,
) where {T<:AbstractFloat}
    h = abs(hubble_code(state))
    means = vec(mean(state.field; dims=(1, 2, 3)))
    mass = abs(characteristic_mass(model.potential, means))
    hubble_scale = h > zero(T) ? inv(h) : T(Inf)
    uv_scale = lattice.kmax_lat > zero(T) ? state.a / lattice.kmax_lat : T(Inf)
    mass_scale = mass > zero(T) ? inv(mass) : T(Inf)
    dt_cosmic = config.evolution.timestep_factor * min(hubble_scale, uv_scale, mass_scale)
    dtau = dt_cosmic / state.a
    cfl_limit = lattice.dx / sqrt(T(3))
    dtau = min(dtau, cfl_limit)
    (isfinite(dtau) && dtau > zero(T)) ||
        throw(NumericalError("could not determine a finite positive time step"))
    dtau / lattice.dx <= inv(sqrt(T(3))) * (one(T) + 16eps(T)) ||
        throw(NumericalError("Courant condition violated"))
    return dtau
end

"""Return false as soon as any evolved state value is NaN or infinite."""
function finite_state(state::SimulationState)
    return isfinite(state.a) && isfinite(state.a_prime) && state.a > 0 &&
           isfinite(state.tau_code) && isfinite(state.cosmic_time_code) &&
           all(isfinite, state.field) && all(isfinite, state.field_prime)
end
