"""
Staggered second-order leapfrog.  Integer-time synchronized velocities remain
in `state` for diagnostics; the authoritative half-step values used for the
next drift are held by the integrator and are checkpointed with it.
"""
mutable struct LeapfrogIntegrator{T} <: AbstractIntegrator
    field_prime_half::Array{T,4}
    half_before::Array{T,4}
    force::Array{T,4}
    rhs_prime::Array{T,4}
    dummy_dfield::Array{T,4}
    zero_prime::Array{T,4}
    synchronized_prime::Array{T,4}
    a_prime_half::T
    initialized::Bool
    last_dtau::T
    dynamics::DynamicsWorkspace{T}
end

function LeapfrogIntegrator(
    state::SimulationState{T};
    deterministic_reductions::Bool=false,
) where {T<:AbstractFloat}
    arrays = ntuple(_ -> similar(state.field), 7)
    fill!(arrays[6], zero(T))
    return LeapfrogIntegrator(
        arrays..., zero(T), false, zero(T),
        DynamicsWorkspace(state; deterministic_reductions),
    )
end

function _initialize_half_step!(
    integrator::LeapfrogIntegrator{T},
    state::SimulationState{T},
    model,
    lattice,
    h::T,
) where {T}
    report = dynamics_rhs!(integrator.dummy_dfield, integrator.rhs_prime,
        state.field, state.field_prime, state.a, state.a_prime,
        model, lattice, integrator.dynamics)
    @. integrator.field_prime_half = state.field_prime + (h / 2) * integrator.rhs_prime
    integrator.a_prime_half = state.a_prime + (h / 2) * report.da_prime
    integrator.initialized = true
    integrator.last_dtau = h
    return nothing
end

function step!(
    integrator::LeapfrogIntegrator{T},
    state::SimulationState{T},
    model::SimulationModel,
    lattice::Lattice3D,
    dtau::Real,
    cache=nothing,
) where {T<:AbstractFloat}
    h = T(dtau)
    h > zero(T) || throw(ArgumentError("dtau must be positive"))
    changed_step = integrator.initialized &&
        abs(h - integrator.last_dtau) > 32eps(T) * max(h, integrator.last_dtau)
    if !integrator.initialized || changed_step
        # `state` is always synchronized, so changing the step discards the old
        # half-step representation and creates the new one without phase drift.
        _initialize_half_step!(integrator, state, model, lattice, h)
    end

    copyto!(integrator.half_before, integrator.field_prime_half)
    q_before = integrator.a_prime_half
    a_before = state.a
    @. state.field = state.field + h * integrator.half_before
    state.a += h * q_before
    state.a > zero(T) || throw(NumericalError("leapfrog produced a non-positive scale factor"))

    # Obtain the velocity-independent field force at the new integer position.
    fill!(integrator.zero_prime, zero(T))
    dynamics_rhs!(integrator.dummy_dfield, integrator.force,
        state.field, integrator.zero_prime, state.a, q_before,
        model, lattice, integrator.dynamics)

    # Center the Hubble friction between the incoming and outgoing half steps.
    hconf = q_before / state.a
    denominator = one(T) + h * hconf
    abs(denominator) > sqrt(eps(T)) ||
        throw(NumericalError("singular centered Hubble-friction update"))
    @. integrator.field_prime_half =
        ((one(T) - h * hconf) * integrator.half_before + h * integrator.force) / denominator
    @. integrator.synchronized_prime =
        (integrator.half_before + integrator.field_prime_half) / 2

    # The scale-factor acceleration depends on the centered kinetic energy.
    q_sync = q_before
    scale_report = dynamics_rhs!(integrator.dummy_dfield, integrator.rhs_prime,
        state.field, integrator.synchronized_prime, state.a, q_sync,
        model, lattice, integrator.dynamics)
    q_after = q_before + h * scale_report.da_prime

    # One corrector couples the centered a' to field friction and kinetic energy.
    q_sync = (q_before + q_after) / 2
    hconf = q_sync / state.a
    denominator = one(T) + h * hconf
    abs(denominator) > sqrt(eps(T)) ||
        throw(NumericalError("singular corrected Hubble-friction update"))
    @. integrator.field_prime_half =
        ((one(T) - h * hconf) * integrator.half_before + h * integrator.force) / denominator
    @. integrator.synchronized_prime =
        (integrator.half_before + integrator.field_prime_half) / 2
    scale_report = dynamics_rhs!(integrator.dummy_dfield, integrator.rhs_prime,
        state.field, integrator.synchronized_prime, state.a, q_sync,
        model, lattice, integrator.dynamics)
    q_after = q_before + h * scale_report.da_prime

    copyto!(state.field_prime, integrator.synchronized_prime)
    integrator.a_prime_half = q_after
    state.a_prime = (q_before + q_after) / 2
    state.cosmic_time_code += h * (a_before + state.a) / 2
    state.tau_code += h
    state.efolds = log(state.a / model.a0)
    state.step += 1
    state.last_dtau = h
    integrator.last_dtau = h
    finite_state(state) || throw(NumericalError("leapfrog produced a non-finite state"))
    return StepReport(h; synchronized=true)
end

"""
Return the integer-time synchronized state without modifying the authoritative
half-step arrays.
"""
synchronize!(::LeapfrogIntegrator, state::SimulationState, model=nothing, lattice=nothing, cache=nothing) =
    SynchronizedView(state)
