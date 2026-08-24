mutable struct DeltaNPatchWorkspace
    phi::Vector{Float64}
    u::Vector{Float64}
    temp_phi::Vector{Float64}
    temp_u::Vector{Float64}
    k1_phi::Vector{Float64}
    k1_u::Vector{Float64}
    k2_phi::Vector{Float64}
    k2_u::Vector{Float64}
    k3_phi::Vector{Float64}
    k3_u::Vector{Float64}
    k4_phi::Vector{Float64}
    k4_u::Vector{Float64}
    gradient::Vector{Float64}
end

function DeltaNPatchWorkspace(nfields::Integer)
    arrays = ntuple(_ -> zeros(Float64, nfields), 13)
    return DeltaNPatchWorkspace(arrays...)
end

function _deltaN_config(config)
    if hasproperty(config, :deltaN)
        return getproperty(config, :deltaN)
    elseif hasproperty(config, :delta_n)
        return getproperty(config, :delta_n)
    end
    return config
end

const DELTAN_PROGRESS_SCHEMA_VERSION = 1
const _DELTAN_ACTIVE = Int8(0)
const _DELTAN_SUCCESS = Int8(1)
const _DELTAN_INVALID = Int8(2)
const _DELTAN_EXHAUSTED = Int8(3)

"""
Serializable state of an in-progress separate-universe calculation.

`local_field` and `local_u=dphi/dN` retain every patch's current homogeneous
state.  `finished` is the checkpointed terminal mask; `status` distinguishes a
successful density crossing from an invalid or limit-exhausted patch.  A sweep
advances each unfinished site by at most one configured `dN`, making the whole
object a consistent restart boundary after every sweep.
"""
mutable struct DeltaNProgress
    schema_version::Int
    local_field::Array{Float64,4}
    local_u::Array{Float64,4}
    local_efolds::Array{Float64,3}
    rho_previous::Array{Float64,3}
    directions::Array{Int8,3}
    status::Array{Int8,3}
    # Do not use a bit-packed BitArray here: threaded writes to distinct sites
    # can still share a storage word and race through read-modify-write.
    finished::Array{Bool,3}
    rho_reference_code::Float64
    su_ratio::Float64
    su_valid::Bool
    efolds_reference::Float64
    reference_step::Int
    model_B::Float64
    integrator::String
    dN::Float64
    max_extra_efolds::Float64
    max_backward_efolds::Float64
    sweep::Int
end

"""True when every separate-universe site is terminal (successful or failed)."""
deltaN_progress_complete(progress::DeltaNProgress) = all(progress.finished)

"""Number of sites that still require separate-universe integration."""
deltaN_remaining_sites(progress::DeltaNProgress) = count(!, progress.finished)

"""
    separate_universe_diagnostic(state, lattice, config)

Return the dimensionless patch-size diagnostic
`R_SU = a * H_code * dx_code = H_physical * (a*dx_physical)` and its configured
pass/fail result.
"""
function separate_universe_diagnostic(state, lattice, config)
    section = _deltaN_config(config)
    minimum_ratio = hasproperty(section, :separate_universe_min_ratio) ?
        Float64(section.separate_universe_min_ratio) : 1.0
    ratio = abs(state.a * hubble_code(state) * lattice.dx)
    return (ratio=ratio, minimum_ratio=minimum_ratio, valid=isfinite(ratio) && ratio >= minimum_ratio)
end

"""
    interpolate_density_crossing(N0, N1, rho0, rho1, rho_reference)

Linearly interpolate the signed e-fold coordinate of a uniform-density
crossing inside one accepted integration step.
"""
function interpolate_density_crossing(
    N0::Real,
    N1::Real,
    rho0::Real,
    rho1::Real,
    rho_reference::Real,
)
    denominator = rho0 - rho1
    isfinite(denominator) && !iszero(denominator) ||
        throw(ArgumentError("density crossing interpolation needs distinct finite endpoint densities"))
    fraction = (rho0 - rho_reference) / denominator
    tolerance = 64eps(Float64)
    -tolerance <= fraction <= 1.0 + tolerance ||
        throw(ArgumentError("reference density is not bracketed by the step endpoints"))
    return Float64(N0 + clamp(fraction, 0.0, 1.0) * (N1 - N0))
end

@inline function _patch_density(potential, phi, u)
    epsilon = 0.5 * sum(abs2, u)
    denominator = 3.0 - epsilon
    denominator > 0.0 && isfinite(denominator) || return NaN
    value = potential_value(potential, phi)
    H2 = value / denominator
    return isfinite(H2) && H2 > 0.0 ? 3.0 * H2 : NaN
end

function _patch_rhs!(dphi, du, phi, u, potential, gradient)
    epsilon = 0.5 * sum(abs2, u)
    denominator = 3.0 - epsilon
    denominator > 0.0 && isfinite(denominator) || return false
    value = potential_value(potential, phi)
    H2 = value / denominator
    isfinite(H2) && H2 > 0.0 || return false
    potential_gradient!(gradient, potential, phi)
    all(isfinite, gradient) || return false
    @inbounds for q in eachindex(phi)
        dphi[q] = u[q]
        du[q] = -(3.0 - epsilon) * u[q] - gradient[q] / H2
    end
    return all(isfinite, du)
end

"""
Generalized staggered leapfrog for `phi'=u`, `u'=f(phi,u)`.

Hubble friction makes the force velocity dependent, so the incoming half-kick
solves `u_half = u_n + h*f(phi_n,u_half)/2` by fixed-point iteration.  The
position then drifts with `u_half`, followed by the outgoing half-kick at
`(phi_{n+1},u_half)`.  This is a genuine time-centred second-order staggered
scheme rather than explicit midpoint RK2.
"""
function _patch_leapfrog_step!(workspace::DeltaNPatchWorkspace, potential, h::Float64)
    w = workspace
    _patch_rhs!(w.k1_phi, w.k1_u, w.phi, w.u, potential, w.gradient) || return false
    @inbounds for q in eachindex(w.phi)
        w.temp_u[q] = w.u[q] + 0.5 * h * w.k1_u[q]
    end

    converged = false
    for _ in 1:16
        _patch_rhs!(w.k2_phi, w.k2_u, w.phi, w.temp_u, potential, w.gradient) || return false
        difference = 0.0
        scale = 1.0
        @inbounds for q in eachindex(w.phi)
            candidate = w.u[q] + 0.5 * h * w.k2_u[q]
            difference = max(difference, abs(candidate - w.temp_u[q]))
            scale = max(scale, abs(candidate), abs(w.temp_u[q]))
            w.k3_u[q] = candidate
        end
        copyto!(w.temp_u, w.k3_u)
        if difference <= 128eps(Float64) * scale
            converged = true
            break
        end
    end
    converged || return false

    @inbounds for q in eachindex(w.phi)
        w.temp_phi[q] = w.phi[q] + h * w.temp_u[q]
    end
    _patch_rhs!(w.k2_phi, w.k2_u, w.temp_phi, w.temp_u, potential, w.gradient) || return false
    @inbounds for q in eachindex(w.phi)
        w.phi[q] = w.temp_phi[q]
        w.u[q] = w.temp_u[q] + 0.5 * h * w.k2_u[q]
    end
    return all(isfinite, w.phi) && all(isfinite, w.u)
end

_patch_midpoint_step!(workspace::DeltaNPatchWorkspace, potential, h::Float64) =
    _patch_leapfrog_step!(workspace, potential, h)

function _patch_rk4_step!(workspace::DeltaNPatchWorkspace, potential, h::Float64)
    w = workspace
    _patch_rhs!(w.k1_phi, w.k1_u, w.phi, w.u, potential, w.gradient) || return false
    @inbounds for q in eachindex(w.phi)
        w.temp_phi[q] = w.phi[q] + 0.5 * h * w.k1_phi[q]
        w.temp_u[q] = w.u[q] + 0.5 * h * w.k1_u[q]
    end
    _patch_rhs!(w.k2_phi, w.k2_u, w.temp_phi, w.temp_u, potential, w.gradient) || return false
    @inbounds for q in eachindex(w.phi)
        w.temp_phi[q] = w.phi[q] + 0.5 * h * w.k2_phi[q]
        w.temp_u[q] = w.u[q] + 0.5 * h * w.k2_u[q]
    end
    _patch_rhs!(w.k3_phi, w.k3_u, w.temp_phi, w.temp_u, potential, w.gradient) || return false
    @inbounds for q in eachindex(w.phi)
        w.temp_phi[q] = w.phi[q] + h * w.k3_phi[q]
        w.temp_u[q] = w.u[q] + h * w.k3_u[q]
    end
    _patch_rhs!(w.k4_phi, w.k4_u, w.temp_phi, w.temp_u, potential, w.gradient) || return false
    sixth = h / 6.0
    @inbounds for q in eachindex(w.phi)
        w.phi[q] += sixth * (w.k1_phi[q] + 2.0 * w.k2_phi[q] + 2.0 * w.k3_phi[q] + w.k4_phi[q])
        w.u[q] += sixth * (w.k1_u[q] + 2.0 * w.k2_u[q] + 2.0 * w.k3_u[q] + w.k4_u[q])
    end
    return all(isfinite, w.phi) && all(isfinite, w.u)
end

function _load_patch!(workspace, state, model, i, j, k)
    nf = length(workspace.phi)
    inva = inv(state.a)
    kinetic = 0.0
    @inbounds for q in 1:nf
        phi = state.field[i, j, k, q]
        velocity = state.field_prime[i, j, k, q] * inva
        workspace.phi[q] = phi
        workspace.temp_u[q] = velocity
        kinetic += 0.5 * velocity * velocity
    end
    rho = kinetic + potential_value(model.potential, workspace.phi)
    isfinite(rho) && rho > 0.0 || return NaN
    Hlocal = sqrt(rho / 3.0)
    @inbounds for q in 1:nf
        workspace.u[q] = workspace.temp_u[q] / Hlocal
    end
    return _patch_density(model.potential, workspace.phi, workspace.u)
end

function _validate_deltaN_reference(state, lattice, config)
    section = _deltaN_config(config)
    reference_mode = lowercase(String(section.reference_surface))
    reference_mode == "uniform_density_at_end" ||
        throw(ArgumentError("only uniform_density_at_end is supported for delta-N"))
    Float64(section.dN) > 0.0 && isfinite(section.dN) ||
        throw(ArgumentError("delta-N step must be finite and positive"))
    Float64(section.max_extra_efolds) >= 0.0 && isfinite(section.max_extra_efolds) ||
        throw(ArgumentError("delta-N forward limit must be finite and non-negative"))
    Float64(section.max_backward_efolds) >= 0.0 && isfinite(section.max_backward_efolds) ||
        throw(ArgumentError("delta-N backward limit must be finite and non-negative"))
    lowercase(String(section.integrator)) in ("rk4", "leapfrog") ||
        throw(ArgumentError("delta-N integrator must be rk4 or leapfrog"))

    diagnostic = separate_universe_diagnostic(state, lattice, config)
    if !diagnostic.valid
        strict = hasproperty(section, :strict_separate_universe) && Bool(section.strict_separate_universe)
        message = "separate-universe condition failed: R_SU=$(diagnostic.ratio), required >= $(diagnostic.minimum_ratio)"
        strict && throw(ArgumentError(message))
        @warn message
    end
    return section, diagnostic
end

"""
    initialize_deltaN_progress(state, model, lattice, config)

Create a fully serializable separate-universe work set.  The common reference
density is captured once from the synchronized end state.  Initial local
Hubble rates omit gradients, while the common reference retains the full
lattice-averaged energy as required by the uniform-density prescription.
"""
function initialize_deltaN_progress(state, model, lattice, config)
    section, diagnostic = _validate_deltaN_reference(state, lattice, config)
    deterministic = hasproperty(config, :parallel) &&
        hasproperty(config.parallel, :deterministic_reductions) &&
        Bool(config.parallel.deterministic_reductions)
    reference_cache = SimulationCache(
        state; deterministic_reductions=deterministic,
    )
    reference_energy = energy_summary(state, model, lattice, reference_cache)
    rho_reference_code = reference_energy.rho
    isfinite(rho_reference_code) && rho_reference_code > 0.0 ||
        throw(DomainError(rho_reference_code, "delta-N reference density must be finite and positive"))

    dimensions = (lattice.N, lattice.N, lattice.N)
    nf = _obs_nfields(state)
    size(state.field)[1:3] == dimensions || throw(DimensionMismatch(
        "delta-N state and lattice dimensions differ",
    ))
    local_field = Float64.(state.field)
    local_u = fill(NaN, dimensions..., nf)
    local_efolds = zeros(Float64, dimensions)
    rho_previous = fill(NaN, dimensions)
    directions = zeros(Int8, dimensions)
    status = fill(_DELTAN_ACTIVE, dimensions)
    finished = fill(false, dimensions)
    sites = CartesianIndices(dimensions)
    workspaces = [DeltaNPatchWorkspace(nf) for _ in 1:Threads.maxthreadid()]

    # A Ctrl-C is delivered only after this sweep-sized transaction, so a
    # checkpoint never observes a half-copied local field vector.
    Base.disable_sigint() do
        Threads.@threads :static for linear_index in eachindex(local_efolds)
            ci = sites[linear_index]
            i, j, k = Tuple(ci)
            workspace = workspaces[Threads.threadid()]
            density = try
                _load_patch!(workspace, state, model, i, j, k)
            catch
                NaN
            end
            if !isfinite(density)
                status[linear_index] = _DELTAN_INVALID
                finished[linear_index] = true
                continue
            end
            @inbounds for q in 1:nf
                local_u[i, j, k, q] = workspace.u[q]
            end
            rho_previous[linear_index] = density
            # Purely relative to the density surface, hence invariant under
            # the solver's arbitrary B rescaling.
            scale = max(abs(density), abs(rho_reference_code))
            if abs(density - rho_reference_code) <= 64eps(Float64) * scale
                status[linear_index] = _DELTAN_SUCCESS
                finished[linear_index] = true
            else
                directions[linear_index] = density > rho_reference_code ? Int8(1) : Int8(-1)
            end
        end
    end

    return DeltaNProgress(
        DELTAN_PROGRESS_SCHEMA_VERSION,
        local_field,
        local_u,
        local_efolds,
        rho_previous,
        directions,
        status,
        finished,
        rho_reference_code,
        diagnostic.ratio,
        diagnostic.valid,
        state.efolds,
        state.step,
        model.B,
        lowercase(String(section.integrator)),
        Float64(section.dN),
        Float64(section.max_extra_efolds),
        Float64(section.max_backward_efolds),
        0,
    )
end

function _validate_deltaN_progress(progress, state, model, lattice, config)
    progress.schema_version == DELTAN_PROGRESS_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported delta-N progress schema $(progress.schema_version)",
    ))
    dimensions = (lattice.N, lattice.N, lattice.N)
    nf = _obs_nfields(state)
    size(progress.local_field) == (dimensions..., nf) ||
        throw(DimensionMismatch("checkpointed delta-N field dimensions differ"))
    size(progress.local_u) == size(progress.local_field) ||
        throw(DimensionMismatch("checkpointed delta-N field/u dimensions differ"))
    for array in (progress.local_efolds, progress.rho_previous, progress.directions,
                  progress.status, progress.finished)
        size(array) == dimensions || throw(DimensionMismatch(
            "checkpointed delta-N site-map dimensions differ",
        ))
    end
    section = _deltaN_config(config)
    progress.reference_step == state.step || throw(ArgumentError(
        "checkpointed delta-N reference step differs from the restored state",
    ))
    progress.efolds_reference == state.efolds || throw(ArgumentError(
        "checkpointed delta-N reference e-fold differs from the restored state",
    ))
    progress.model_B == model.B || throw(ArgumentError(
        "checkpointed delta-N code-unit scale differs from the restored model",
    ))
    progress.integrator == lowercase(String(section.integrator)) || throw(ArgumentError(
        "checkpointed delta-N integrator differs from the restored configuration",
    ))
    progress.dN == Float64(section.dN) || throw(ArgumentError(
        "checkpointed delta-N step differs from the restored configuration",
    ))
    progress.max_extra_efolds == Float64(section.max_extra_efolds) || throw(ArgumentError(
        "checkpointed delta-N forward limit differs from the restored configuration",
    ))
    progress.max_backward_efolds == Float64(section.max_backward_efolds) || throw(ArgumentError(
        "checkpointed delta-N backward limit differs from the restored configuration",
    ))
    return progress
end

function _load_progress_patch!(workspace, progress, i, j, k)
    @inbounds for q in eachindex(workspace.phi)
        workspace.phi[q] = progress.local_field[i, j, k, q]
        workspace.u[q] = progress.local_u[i, j, k, q]
    end
    return workspace
end

function _store_progress_patch!(progress, workspace, i, j, k)
    @inbounds for q in eachindex(workspace.phi)
        progress.local_field[i, j, k, q] = workspace.phi[q]
        progress.local_u[i, j, k, q] = workspace.u[q]
    end
    return progress
end

function _advance_deltaN_site!(progress, workspace, model, ci, linear_index)
    progress.finished[linear_index] && return nothing
    i, j, k = Tuple(ci)
    direction = progress.directions[linear_index]
    direction in (Int8(-1), Int8(1)) || begin
        progress.status[linear_index] = _DELTAN_INVALID
        progress.finished[linear_index] = true
        return nothing
    end
    Nprevious = progress.local_efolds[linear_index]
    rho_previous = progress.rho_previous[linear_index]
    limit = direction > 0 ? progress.max_extra_efolds : progress.max_backward_efolds
    remaining = limit - abs(Nprevious)
    if remaining <= 8eps(Float64) * max(limit, 1.0)
        progress.status[linear_index] = _DELTAN_EXHAUSTED
        progress.finished[linear_index] = true
        return nothing
    end

    _load_progress_patch!(workspace, progress, i, j, k)
    h = Float64(direction) * min(progress.dN, remaining)
    step_ok = try
        progress.integrator == "rk4" ?
            _patch_rk4_step!(workspace, model.potential, h) :
            _patch_leapfrog_step!(workspace, model.potential, h)
    catch
        false
    end
    if !step_ok
        progress.status[linear_index] = _DELTAN_INVALID
        progress.finished[linear_index] = true
        return nothing
    end
    rho_next = try
        _patch_density(model.potential, workspace.phi, workspace.u)
    catch
        NaN
    end
    if !isfinite(rho_next)
        progress.status[linear_index] = _DELTAN_INVALID
        progress.finished[linear_index] = true
        return nothing
    end

    _store_progress_patch!(progress, workspace, i, j, k)
    Nnext = Nprevious + h
    progress.rho_previous[linear_index] = rho_next
    crossed = direction > 0 ?
        rho_next <= progress.rho_reference_code :
        rho_next >= progress.rho_reference_code
    if crossed
        progress.local_efolds[linear_index] = interpolate_density_crossing(
            Nprevious, Nnext, rho_previous, rho_next, progress.rho_reference_code,
        )
        progress.status[linear_index] = _DELTAN_SUCCESS
        progress.finished[linear_index] = true
    else
        progress.local_efolds[linear_index] = Nnext
        new_remaining = limit - abs(Nnext)
        if new_remaining <= 8eps(Float64) * max(limit, 1.0)
            progress.status[linear_index] = _DELTAN_EXHAUSTED
            progress.finished[linear_index] = true
        end
    end
    return nothing
end

function _advance_deltaN_sweep!(progress, model, workspaces)
    dimensions = size(progress.local_efolds)
    sites = CartesianIndices(dimensions)
    # Deferring SIGINT across a sweep makes every serialized progress object a
    # set of complete per-site commits.  The interrupt is delivered before the
    # next sweep or callback.
    Base.disable_sigint() do
        Threads.@threads :static for linear_index in eachindex(progress.local_efolds)
            progress.finished[linear_index] && continue
            _advance_deltaN_site!(
                progress,
                workspaces[Threads.threadid()],
                model,
                sites[linear_index],
                linear_index,
            )
        end
        progress.sweep += 1
    end
    return progress
end

"""
    advance_deltaN!(progress, state, model, lattice, config;
                    max_sweeps=typemax(Int), checkpoint_callback=nothing,
                    checkpoint_every_sweeps=1)

Advance only unfinished patches.  A callback receives the same serializable
`DeltaNProgress` at consistent sweep boundaries and may save it or deliberately
throw `InterruptException`; resuming with that object never recomputes finished
sites.
"""
function advance_deltaN!(
    progress::DeltaNProgress,
    state,
    model,
    lattice,
    config;
    max_sweeps::Integer=typemax(Int),
    checkpoint_callback=nothing,
    checkpoint_every_sweeps::Integer=1,
)
    _validate_deltaN_progress(progress, state, model, lattice, config)
    max_sweeps >= 0 || throw(ArgumentError("max_sweeps must be non-negative"))
    checkpoint_every_sweeps > 0 || throw(ArgumentError(
        "checkpoint_every_sweeps must be positive",
    ))
    completed_here = 0
    workspaces = [DeltaNPatchWorkspace(size(progress.local_field, 4))
                  for _ in 1:Threads.maxthreadid()]
    while !deltaN_progress_complete(progress) && completed_here < max_sweeps
        _advance_deltaN_sweep!(progress, model, workspaces)
        completed_here += 1
        if checkpoint_callback !== nothing &&
           (progress.sweep % checkpoint_every_sweeps == 0 || deltaN_progress_complete(progress))
            checkpoint_callback(progress)
        end
    end
    return progress
end

function _integrate_patch!(workspace, state, model, ci, rho_reference, section)
    i, j, k = Tuple(ci)
    rho_previous = _load_patch!(workspace, state, model, i, j, k)
    isfinite(rho_previous) || return (NaN, Int8(1), Int8(0))

    scale = max(abs(rho_previous), abs(rho_reference))
    if abs(rho_previous - rho_reference) <= 64eps(Float64) * scale
        return (0.0, Int8(0), Int8(0))
    end

    direction = rho_previous > rho_reference ? Int8(1) : Int8(-1)
    spacing = Float64(section.dN)
    spacing > 0.0 && isfinite(spacing) || return (NaN, Int8(1), direction)
    limit = direction > 0 ? Float64(section.max_extra_efolds) : Float64(section.max_backward_efolds)
    limit >= 0.0 && isfinite(limit) || return (NaN, Int8(1), direction)
    integrator_name = lowercase(String(section.integrator))
    stepper! = integrator_name == "rk4" ? _patch_rk4_step! :
               integrator_name == "leapfrog" ? _patch_leapfrog_step! : nothing
    stepper! === nothing && return (NaN, Int8(1), direction)

    Nprevious = 0.0
    max_steps = ceil(Int, limit / spacing) + 1
    @inbounds for _ in 1:max_steps
        remaining = limit - abs(Nprevious)
        remaining <= 8eps(Float64) * max(limit, 1.0) && break
        h = Float64(direction) * min(spacing, remaining)
        stepper!(workspace, model.potential, h) || return (NaN, Int8(1), direction)
        Nnext = Nprevious + h
        rho_next = _patch_density(model.potential, workspace.phi, workspace.u)
        isfinite(rho_next) || return (NaN, Int8(1), direction)
        crossed = direction > 0 ? rho_next <= rho_reference : rho_next >= rho_reference
        if crossed
            crossing = interpolate_density_crossing(
                Nprevious, Nnext, rho_previous, rho_next, rho_reference,
            )
            return (crossing, Int8(0), direction)
        end
        Nprevious = Nnext
        rho_previous = rho_next
    end
    return (NaN, Int8(2), direction)
end

function _deltaN_statistics(zeta, successful)
    sample_count = count(identity, successful)
    sample_count > 0 || return (NaN, NaN, NaN, NaN)
    mean_value = sum(zeta[i] for i in eachindex(zeta) if successful[i]) / sample_count
    second = sum((zeta[i] - mean_value)^2 for i in eachindex(zeta) if successful[i]) / sample_count
    if !(isfinite(second) && second > 0.0)
        return (mean_value, second, NaN, NaN)
    end
    sigma = sqrt(second)
    skewness = sum(((zeta[i] - mean_value) / sigma)^3 for i in eachindex(zeta) if successful[i]) / sample_count
    excess = sum(((zeta[i] - mean_value) / sigma)^4 for i in eachindex(zeta) if successful[i]) / sample_count - 3.0
    return (mean_value, second, skewness, excess)
end

"""Finalize a terminal progress object into the stable public `DeltaNResult`."""
function finalize_deltaN(progress::DeltaNProgress, state, model, lattice, config)
    _validate_deltaN_progress(progress, state, model, lattice, config)
    deltaN_progress_complete(progress) || throw(ArgumentError(
        "cannot finalize delta-N while $(deltaN_remaining_sites(progress)) sites remain",
    ))
    dimensions = size(progress.local_efolds)
    sites = CartesianIndices(dimensions)
    successful = progress.status .== _DELTAN_SUCCESS
    local_efolds = copy(progress.local_efolds)
    failed_indices = CartesianIndex{3}[]
    @inbounds for linear_index in eachindex(progress.status)
        if !successful[linear_index]
            local_efolds[linear_index] = NaN
            push!(failed_indices, sites[linear_index])
        end
    end

    successful_count = count(identity, successful)
    local_mean = successful_count > 0 ?
        sum(local_efolds[i] for i in eachindex(local_efolds) if successful[i]) /
        successful_count : NaN
    zeta = fill(NaN, dimensions)
    @inbounds for i in eachindex(zeta)
        successful[i] && (zeta[i] = local_efolds[i] - local_mean)
    end
    mean_value, variance, skewness, excess = _deltaN_statistics(zeta, successful)

    rho_reference_physical = model.B^2 * progress.rho_reference_code
    spectrum = NamedTuple[]
    if isempty(failed_indices)
        for shell in scalar_spectrum(zeta, state, model, lattice)
            push!(spectrum, (
                efolds_reference=progress.efolds_reference,
                rho_reference=rho_reference_physical,
                k_bin=shell.k_bin,
                k_comoving=shell.k_comoving,
                k_physical=shell.k_physical,
                n_modes=shell.n_modes,
                power_dimensionless=shell.power_dimensionless,
            ))
        end
    else
        @warn "delta-N patches failed to reach the reference density" count=length(failed_indices) first_indices=first(failed_indices, min(8, length(failed_indices)))
    end

    return DeltaNResult(
        local_efolds,
        zeta,
        rho_reference_physical,
        progress.su_ratio,
        progress.su_valid,
        count(==(Int8(1)), progress.directions),
        count(==(Int8(-1)), progress.directions),
        failed_indices,
        mean_value,
        variance,
        skewness,
        excess,
        spectrum,
    )
end

"""
    compute_deltaN(state, model, lattice, config; progress=nothing,
                   checkpoint_callback=nothing,
                   checkpoint_every_sweeps=1) -> DeltaNResult

Evolve every lattice site as an independent homogeneous patch in e-fold time,
dropping all spatial gradients.  Supplying a checkpoint-restored `progress`
continues only unfinished sites.  `checkpoint_callback` runs at consistent
sweep boundaries and receives the serializable local fields, local `u`, e-fold
positions, densities, directions, status codes, and finished mask.
"""
function compute_deltaN(
    state,
    model,
    lattice,
    config;
    progress=nothing,
    checkpoint_callback=nothing,
    checkpoint_every_sweeps::Integer=1,
)
    work = progress === nothing ?
        initialize_deltaN_progress(state, model, lattice, config) : progress
    work isa DeltaNProgress || throw(ArgumentError(
        "progress must be a DeltaNProgress or nothing",
    ))
    advance_deltaN!(
        work,
        state,
        model,
        lattice,
        config;
        checkpoint_callback=checkpoint_callback,
        checkpoint_every_sweeps=checkpoint_every_sweeps,
    )
    return finalize_deltaN(work, state, model, lattice, config)
end
