"""Supertype for fixed-step conformal-time integrators."""
abstract type AbstractIntegrator end

"""
    step!(integrator, state, model, lattice, dtau, cache) -> StepReport

Advance `state` by the positive code-unit conformal-time interval `dtau`.
Concrete integrators own any method-specific stage or staggering state.
"""
function step! end

"""
    synchronize!(integrator, state, model, lattice, cache) -> SynchronizedView

Return an integer-time view suitable for diagnostics and output.  For a
staggered method this must not destroy or permanently alter its half-step
state.
"""
function synchronize! end
