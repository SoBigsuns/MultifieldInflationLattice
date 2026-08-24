"""Return the Hubble parameter in the dimensionless code units used by the solver."""
@inline hubble_code(state) = state.a_prime / (state.a * state.a)

"""Return the physical Hubble parameter (reduced-Planck units)."""
@inline hubble_parameter(state, model) = model.B * hubble_code(state)

@inline _obs_nfields(state) = size(state.field, 4)
@inline _obs_nsites(state) = size(state.field, 1) * size(state.field, 2) * size(state.field, 3)

"""
    field_means(state)

Spatial means of all scalar fields.  Field storage follows `(x,y,z,field)`.
"""
function field_means(state)
    nf = _obs_nfields(state)
    nsites = _obs_nsites(state)
    means = zeros(Float64, nf)
    @inbounds for q in 1:nf
        total = 0.0
        for k in axes(state.field, 3), j in axes(state.field, 2), i in axes(state.field, 1)
            total += state.field[i, j, k, q]
        end
        means[q] = total / nsites
    end
    return means
end

"""
    field_variances(state, means=field_means(state))

Population variances of the fields.  A negative result caused solely by
roundoff in `E[x^2]-E[x]^2` is clipped to zero.
"""
function field_variances(state, means=field_means(state))
    nf = _obs_nfields(state)
    length(means) == nf || throw(DimensionMismatch("one mean is required per field"))
    nsites = _obs_nsites(state)
    variances = zeros(Float64, nf)
    @inbounds for q in 1:nf
        total2 = 0.0
        for k in axes(state.field, 3), j in axes(state.field, 2), i in axes(state.field, 1)
            value = state.field[i, j, k, q]
            total2 += value * value
        end
        raw = total2 / nsites - means[q] * means[q]
        scale = max(total2 / nsites, means[q] * means[q], 1.0)
        variances[q] = raw < 0.0 && abs(raw) <= 32eps(Float64) * scale ? 0.0 : raw
    end
    return variances
end

"""
    mean_field_velocities_code(state)

Spatial means of `dphi/dt_code = field_prime/a`.
"""
function mean_field_velocities_code(state)
    nf = _obs_nfields(state)
    nsites = _obs_nsites(state)
    inva = inv(state.a)
    velocities = zeros(Float64, nf)
    @inbounds for q in 1:nf
        total = 0.0
        for k in axes(state.field_prime, 3), j in axes(state.field_prime, 2), i in axes(state.field_prime, 1)
            total += state.field_prime[i, j, k, q]
        end
        velocities[q] = total * inva / nsites
    end
    return velocities
end

"""Spatially averaged physical cosmic-time velocities `dphi/dt`."""
mean_field_velocities(state, model) = model.B .* mean_field_velocities_code(state)

"""
    background_summary(state, model, lattice=nothing)

Collect one wide-row-ready background observation.  Times, Hubble quantities,
and velocities are converted from solver code units using `model.B`; fields and
e-folds are already dimensionless reduced-Planck quantities.
"""
function background_summary(state, model, lattice=nothing)
    means = field_means(state)
    variances = field_variances(state, means)
    velocities = mean_field_velocities(state, model)
    H = hubble_parameter(state, model)

    # Hdot follows the evolved scale factor only if a'' is stored.  The
    # matter expression, Hdot = -(rho+p)/2, is supplied by the overload below.
    return (
        step=state.step,
        tau=state.tau_code / model.B,
        t=state.cosmic_time_code / model.B,
        efolds=state.efolds,
        a=state.a,
        H=H,
        field_means=means,
        field_velocities=velocities,
        field_variances=variances,
    )
end

"""
    background_summary(state, model, lattice, energy)

As `background_summary`, with physical `Hdot`, `rho`, `p`, and `w`.  `energy`
is the code-unit result returned by `energy_summary`.
"""
function background_summary(state, model, lattice, energy)
    base = background_summary(state, model, lattice)
    B2 = model.B * model.B
    return merge(base, (
        Hdot=-0.5 * B2 * (energy.rho + energy.p),
        rho=B2 * energy.rho,
        p=B2 * energy.p,
        w=energy.w,
    ))
end
