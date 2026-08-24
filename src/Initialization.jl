@inline function _splitmix64(value::UInt64)
    z = value + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

"""Deterministically derive a field-local seed from the configured root seed."""
field_seed(root::Integer, field_index::Integer) =
    _splitmix64(UInt64(root) ⊻ (UInt64(field_index) * 0x9e3779b97f4a7c15))

@inline _negative_fft_index(index::Int, n::Int) = mod(-(index - 1), n) + 1

@inline function _signed_fft_mode(index::Int, n::Int)
    raw = index - 1
    return raw <= n ÷ 2 ? raw : raw - n
end

@inline function _mode_k_lat(i::Int, j::Int, k::Int, lattice::Lattice3D)
    nx = _signed_fft_mode(i, lattice.N)
    ny = _signed_fft_mode(j, lattice.N)
    nz = _signed_fft_mode(k, lattice.N)
    return (2 / lattice.dx) * sqrt(
        sinpi(nx / lattice.N)^2 +
        sinpi(ny / lattice.N)^2 +
        sinpi(nz / lattice.N)^2,
    )
end

function _mode_allowed(i::Int, j::Int, k::Int, lattice, initial)
    nx = _signed_fft_mode(i, lattice.N)
    ny = _signed_fft_mode(j, lattice.N)
    nz = _signed_fft_mode(k, lattice.N)
    index_norm = sqrt(nx^2 + ny^2 + nz^2)
    index_norm == 0 && return false
    initial.low_cutoff_index > 0 && index_norm < initial.low_cutoff_index && return false
    initial.high_cutoff_index > 0 && index_norm > initial.high_cutoff_index && return false
    return true
end

"""
Create one real Bunch--Davies field realization and its conformal code-time
derivative.  FFTW coefficients are normalized so that the physical DFT
coefficient has variance `V_phys/(2a^2 omega_phys)`.  A conjugate pair is
sampled once and explicitly mirrored, which makes the inverse transform real.
"""
function _bunch_davies_field(
    rng::AbstractRNG,
    lattice::Lattice3D,
    a0::T,
    hconf0::T,
    B::T,
    effective_mass::T,
    initial,
) where {T<:AbstractFloat}
    n = lattice.N
    coefficients = zeros(Complex{T}, n, n, n)
    prime_coefficients = similar(coefficients)
    linear = LinearIndices(coefficients)
    normalization = B * sqrt(lattice.volume / 2) / (a0 * lattice.dx^3)
    removed = 0

    @inbounds for k in 1:n, j in 1:n, i in 1:n
        ci = _negative_fft_index(i, n)
        cj = _negative_fft_index(j, n)
        ck = _negative_fft_index(k, n)
        linear[i, j, k] > linear[ci, cj, ck] && continue
        if !_mode_allowed(i, j, k, lattice, initial)
            removed += linear[i, j, k] == linear[ci, cj, ck] ? 1 : 2
            continue
        end
        klat = T(_mode_k_lat(i, j, k, lattice))
        omega = sqrt(klat^2 + (a0 * effective_mass)^2)
        omega > zero(T) || continue
        u1 = clamp(rand(rng, T), eps(T), one(T) - eps(T))
        theta = T(2pi) * rand(rng, T)
        rayleigh = sqrt(-log(u1))
        amplitude = normalization * rayleigh / sqrt(omega)

        if linear[i, j, k] == linear[ci, cj, ck]
            # Zero/Nyquist self-conjugate modes have one real degree of
            # freedom.  The quadrature below preserves the ensemble variance.
            value = sqrt(T(2)) * amplitude * cos(theta)
            prime = -hconf0 * value + sqrt(T(2)) * omega * amplitude * sin(theta)
            coefficients[i, j, k] = complex(value)
            prime_coefficients[i, j, k] = complex(prime)
        else
            value = amplitude * cis(theta)
            prime = (-im * omega - hconf0) * value
            coefficients[i, j, k] = value
            coefficients[ci, cj, ck] = conj(value)
            prime_coefficients[i, j, k] = prime
            prime_coefficients[ci, cj, ck] = conj(prime)
        end
    end
    coefficients[1, 1, 1] = zero(Complex{T})
    prime_coefficients[1, 1, 1] = zero(Complex{T})
    field = real.(ifft(coefficients))
    field_prime = real.(ifft(prime_coefficients))
    return field, field_prime, removed
end

"""
    initialize_simulation(config)

Build the lattice/model/state, add independent field-local Bunch--Davies
realizations, and finally recompute the Friedmann-constrained initial Hubble
rate from the complete lattice energy.
"""
function initialize_simulation(config::SimulationConfig)
    validate_config(config)
    # Initialization itself performs the first FFTs when vacuum fluctuations
    # are enabled, so configure FFTW before constructing any modes.
    FFTW.set_num_threads(config.parallel.fft_threads)
    model = build_model(config)
    lattice = build_lattice(config.lattice)
    n = lattice.N
    nf = length(config.model.field_names)
    a0 = Float64(config.initial.scale_factor)
    fields = Array{Float64}(undef, n, n, n, nf)
    primes = Array{Float64}(undef, n, n, n, nf)
    for f in 1:nf
        @views fill!(fields[:, :, :, f], config.initial.field_means[f])
        @views fill!(primes[:, :, :, f],
            a0 * config.initial.cosmic_velocities[f] / model.B)
    end

    # The provisional rate is used only in the vacuum mode derivative.
    background = Float64.(config.initial.field_means)
    v0 = potential_value(model.potential, background)
    kinetic0 = sum((Float64.(config.initial.cosmic_velocities) ./ model.B) .^ 2) / 2
    h0_code = sqrt(max((v0 + kinetic0) / 3, 0.0))
    state = SimulationState(fields, primes, a0, a0^2 * h0_code,
        0.0, 0.0, 0.0, 0, 0.0)

    removed_modes = zeros(Int, nf)
    seeds = [field_seed(config.initial.seed, f) for f in 1:nf]
    if config.initial.vacuum_fluctuations
        hessian = zeros(Float64, nf, nf)
        potential_hessian!(hessian, model.potential, background)
        for f in 1:nf
            effective_mass = config.initial.include_effective_mass ?
                sqrt(max(hessian[f, f], 0.0)) : 0.0
            field_rng = MersenneTwister(seeds[f])
            fluctuation, fluctuation_prime, removed = _bunch_davies_field(
                field_rng, lattice, a0, a0 * h0_code, model.B,
                effective_mass, config.initial)
            @views fields[:, :, :, f] .+= fluctuation
            @views primes[:, :, :, f] .+= fluctuation_prime
            removed_modes[f] = removed
        end
    end

    workspace = DynamicsWorkspace(
        state;
        deterministic_reductions=config.parallel.deterministic_reductions,
    )
    energy = energy_components(state, model, lattice, workspace)
    (isfinite(energy.rho) && energy.rho >= 0) ||
        throw(NumericalError("invalid initial mean energy density $(energy.rho)"))
    h0_code = sqrt(energy.rho / 3)
    state.a_prime = state.a^2 * h0_code
    root_rng = MersenneTwister(UInt64(config.initial.seed))
    initial_diagnostics = (
        B=model.B,
        H_code=h0_code,
        H=model.B * h0_code,
        rho_code=energy.rho,
        rho=energy.rho * model.B^2,
        kmin_over_aH=lattice.kmin_lat / (a0 * h0_code),
        kmax_over_aH=lattice.kmax_lat / (a0 * h0_code),
        cfl_limit=lattice.dx / sqrt(3),
        field_seeds=seeds,
        removed_modes=removed_modes,
    )
    return (; state, model, lattice, rng=root_rng, workspace, initial_diagnostics)
end
