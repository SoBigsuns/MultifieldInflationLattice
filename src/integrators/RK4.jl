"""Classical four-stage Runge--Kutta integrator with reusable stage storage."""
mutable struct RK4Integrator{T} <: AbstractIntegrator
    k1_field::Array{T,4}
    k2_field::Array{T,4}
    k3_field::Array{T,4}
    k4_field::Array{T,4}
    k1_prime::Array{T,4}
    k2_prime::Array{T,4}
    k3_prime::Array{T,4}
    k4_prime::Array{T,4}
    stage_field::Array{T,4}
    stage_prime::Array{T,4}
    dynamics::DynamicsWorkspace{T}
end

function RK4Integrator(
    state::SimulationState{T};
    deterministic_reductions::Bool=false,
) where {T<:AbstractFloat}
    arrays = ntuple(_ -> similar(state.field), 10)
    return RK4Integrator(
        arrays...,
        DynamicsWorkspace(state; deterministic_reductions),
    )
end

function step!(
    integrator::RK4Integrator{T},
    state::SimulationState{T},
    model::SimulationModel,
    lattice::Lattice3D,
    dtau::Real,
    cache=nothing,
) where {T<:AbstractFloat}
    h = T(dtau)
    h > zero(T) || throw(ArgumentError("dtau must be positive"))
    f0 = state.field
    p0 = state.field_prime
    a0 = state.a
    q0 = state.a_prime

    r1 = dynamics_rhs!(integrator.k1_field, integrator.k1_prime,
        f0, p0, a0, q0, model, lattice, integrator.dynamics)

    a2 = a0 + h * r1.da / 2
    q2 = q0 + h * r1.da_prime / 2
    @. integrator.stage_field = f0 + h * integrator.k1_field / 2
    @. integrator.stage_prime = p0 + h * integrator.k1_prime / 2
    r2 = dynamics_rhs!(integrator.k2_field, integrator.k2_prime,
        integrator.stage_field, integrator.stage_prime, a2, q2,
        model, lattice, integrator.dynamics)

    a3 = a0 + h * r2.da / 2
    q3 = q0 + h * r2.da_prime / 2
    @. integrator.stage_field = f0 + h * integrator.k2_field / 2
    @. integrator.stage_prime = p0 + h * integrator.k2_prime / 2
    r3 = dynamics_rhs!(integrator.k3_field, integrator.k3_prime,
        integrator.stage_field, integrator.stage_prime, a3, q3,
        model, lattice, integrator.dynamics)

    a4 = a0 + h * r3.da
    q4 = q0 + h * r3.da_prime
    @. integrator.stage_field = f0 + h * integrator.k3_field
    @. integrator.stage_prime = p0 + h * integrator.k3_prime
    r4 = dynamics_rhs!(integrator.k4_field, integrator.k4_prime,
        integrator.stage_field, integrator.stage_prime, a4, q4,
        model, lattice, integrator.dynamics)

    @. state.field = f0 + (h / 6) * (
        integrator.k1_field + 2 * integrator.k2_field +
        2 * integrator.k3_field + integrator.k4_field)
    @. state.field_prime = p0 + (h / 6) * (
        integrator.k1_prime + 2 * integrator.k2_prime +
        2 * integrator.k3_prime + integrator.k4_prime)
    state.a = a0 + (h / 6) * (r1.da + 2 * r2.da + 2 * r3.da + r4.da)
    state.a_prime = q0 + (h / 6) *
        (r1.da_prime + 2 * r2.da_prime + 2 * r3.da_prime + r4.da_prime)
    state.cosmic_time_code += (h / 6) * (a0 + 2 * a2 + 2 * a3 + a4)
    state.tau_code += h
    state.efolds = log(state.a / model.a0)
    state.step += 1
    state.last_dtau = h
    finite_state(state) || throw(NumericalError("RK4 produced a non-finite state"))
    return StepReport(h; synchronized=true)
end

"""RK4 stores all variables at the same integer time, so synchronization is a no-op."""
synchronize!(::RK4Integrator, state::SimulationState, model=nothing, lattice=nothing, cache=nothing) =
    SynchronizedView(state)
