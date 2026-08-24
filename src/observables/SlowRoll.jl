"""
    slowroll_summary(state, model, lattice, energy, cache=nothing;
                     min_speed2=1e-30) -> SlowRollSummary

Evaluate Hubble and potential slow-roll diagnostics and the mean-field
trajectory geometry.  `energy` must be the code-unit result of
`energy_summary`.  `turn_rate` is returned in physical reduced-Planck units;
all other returned rates are dimensionless except as indicated by their name.
Undefined ratios are represented by IEEE `NaN`, never by a fabricated zero.
`min_speed2` is a physical squared cosmic-time velocity in reduced-Planck
units, matching the TOML convention.
"""
function slowroll_summary(
    state,
    model,
    lattice,
    energy,
    cache=nothing;
    min_speed2::Real=1.0e-30,
)
    nf = _obs_nfields(state)
    nsites = _obs_nsites(state)
    Hcode = hubble_code(state)
    H2 = Hcode * Hcode

    epsilonH_by_field = fill(NaN, nf)
    epsilon_H = NaN
    if isfinite(H2) && H2 > 0.0
        @inbounds for q in 1:nf
            epsilonH_by_field[q] =
                (energy.kinetic_by_field[q] + energy.gradient_by_field[q] / 3.0) / H2
        end
        # This form is identical to sum(epsilonH_by_field), but directly
        # records the defining stress-energy expression.
        epsilon_H = (energy.rho + energy.p) / (2.0 * H2)
    end

    means = field_means(state)
    velocities = mean_field_velocities_code(state)
    speed2 = sum(abs2, velocities)
    physical_speed2 = model.B * model.B * speed2
    eta_parallel = NaN
    turn_rate_code = NaN

    # Configuration thresholds are expressed in reduced-Planck physical
    # cosmic-time units, whereas `velocities` uses code time.  Convert the
    # squared speed before deciding whether the trajectory direction exists.
    if isfinite(physical_speed2) && physical_speed2 >= min_speed2 &&
       isfinite(Hcode) && !iszero(Hcode)
        deterministic = cache !== nothing &&
            hasproperty(cache, :deterministic_reductions) &&
            getproperty(cache, :deterministic_reductions)
        mean_gradient = _mean_potential_gradient(
            state, model; deterministic_reductions=deterministic,
        )
        accelerations = similar(velocities)
        @inbounds for q in 1:nf
            # The spatial mean of the periodic Laplacian vanishes exactly.
            accelerations[q] = -3.0 * Hcode * velocities[q] - mean_gradient[q]
        end

        speed = sqrt(speed2)
        esigma = velocities ./ speed
        sigma_acceleration = dot(esigma, accelerations)
        eta_parallel = -sigma_acceleration / (Hcode * speed)
        perpendicular2 = 0.0
        @inbounds for q in 1:nf
            component = accelerations[q] - esigma[q] * sigma_acceleration
            perpendicular2 += component * component
        end
        turn_rate_code = sqrt(max(perpendicular2, 0.0)) / speed
    end

    potential_at_mean = potential_value(model.potential, means)
    potential_gradient = zeros(Float64, nf)
    potential_hessian = zeros(Float64, nf, nf)
    potential_gradient!(potential_gradient, model.potential, means)
    potential_hessian!(potential_hessian, model.potential, means)

    epsilon_V = NaN
    etaV_diag = fill(NaN, nf)
    etaV_eigenvalues = fill(NaN, nf)
    # All potential derivatives carry the same B^-2 scaling.  A relative
    # scale assembled only from those quantities keeps the undefined-potential
    # test invariant when an equivalent explicit rescaling B is selected.
    potential_scale = max(
        abs(potential_at_mean),
        sum(abs, potential_gradient),
        sum(abs, potential_hessian),
        floatmin(Float64),
    )
    potential_floor = eps(Float64) * potential_scale
    if isfinite(potential_at_mean) && abs(potential_at_mean) > potential_floor &&
       all(isfinite, potential_gradient) && all(isfinite, potential_hessian)
        epsilon_V = 0.5 * sum(abs2, potential_gradient) / (potential_at_mean * potential_at_mean)
        eta_matrix = 0.5 .* (potential_hessian .+ transpose(potential_hessian)) ./ potential_at_mean
        etaV_diag .= diag(eta_matrix)
        etaV_eigenvalues .= sort!(collect(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(eta_matrix))))
    end

    turn_rate = model.B * turn_rate_code
    turn_rate_over_H = turn_rate_code / Hcode
    return SlowRollSummary(
        epsilon_H,
        epsilon_V,
        eta_parallel,
        turn_rate,
        turn_rate_over_H,
        epsilonH_by_field,
        etaV_diag,
        etaV_eigenvalues,
    )
end

function _mean_potential_gradient(
    state,
    model;
    deterministic_reductions::Bool=false,
)
    nx, ny, nz, nf = size(state.field)
    nt = Threads.maxthreadid()
    reduction_slots = max(nz, nt)
    partial = zeros(Float64, nf, reduction_slots)
    scratch_gradient = zeros(Float64, nf, nt)
    scratch_fields = zeros(Float64, nf, nt)
    Threads.@threads :static for k in 1:nz
        tid = Threads.threadid()
        reduction_slot = deterministic_reductions ? k : tid
        out = @view scratch_gradient[:, tid]
        fields = @view scratch_fields[:, tid]
        @inbounds for j in 1:ny, i in 1:nx
            for q in 1:nf
                fields[q] = state.field[i, j, k, q]
            end
            potential_gradient!(out, model.potential, fields)
            for q in 1:nf
                partial[q, reduction_slot] += out[q]
            end
        end
    end
    return vec(sum(partial; dims=2)) ./ (nx * ny * nz)
end
