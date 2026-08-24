const _TWO_PI_SQUARED = 2.0 * pi * pi

@inline function _fft_integer(index::Int, n::Int)
    raw = index - 1
    return raw <= fld(n, 2) ? raw : raw - n
end

function _spectra_use_lattice_momentum(config)
    section = hasproperty(config, :spectra) ? getproperty(config, :spectra) : config
    return hasproperty(section, :use_lattice_momentum) ?
        Bool(getproperty(section, :use_lattice_momentum)) : true
end

"""
    lattice_dft(map3, lattice)

Forward DFT with the physical Riemann-sum normalization
`X_k = dx^3 * sum_x exp(-ikx) X(x)` in *code spatial units*.  Together with
the code-unit `k` and `L` used below, all powers of the rescaling `B` cancel,
giving the same dimensionless spectrum as an explicit Planck-unit transform.
Julia/FFTW's forward transform is unnormalised, hence the explicit `dx^3`.
"""
function lattice_dft(map3::AbstractArray{<:Real,3}, lattice)
    size(map3) == (lattice.N, lattice.N, lattice.N) ||
        throw(DimensionMismatch("map and lattice sizes differ"))
    all(isfinite, map3) || throw(ArgumentError("a spectrum cannot be formed from non-finite data"))
    centered = Float64.(map3)
    centered .-= sum(centered) / length(centered)
    return (lattice.dx^3) .* FFTW.fft(centered)
end

"""
Internal shell lookup.  Shell membership uses the rounded norm of the integer
FFT wave vector.  Labels and power prefactors use the seven-point-Laplacian
momentum when requested:

`k_lat = (2/dx) sqrt(sum_i sin(pi*n_i/N)^2)`.
"""
function _spectral_shell_plan(lattice; use_lattice_momentum::Bool=true)
    n = lattice.N
    nmodes = n * n * n
    bins = fill(-1, nmodes)
    kmode = zeros(Float64, nmodes)
    maximum_bin = round(Int, sqrt(3.0) * fld(n, 2))
    counts = zeros(Int, maximum_bin + 1)
    ksum = zeros(Float64, maximum_bin + 1)
    linear = LinearIndices((n, n, n))

    @inbounds for iz in 1:n, iy in 1:n, ix in 1:n
        nx = _fft_integer(ix, n)
        ny = _fft_integer(iy, n)
        nz = _fft_integer(iz, n)
        nsquared = nx * nx + ny * ny + nz * nz
        nsquared == 0 && continue
        bin = round(Int, sqrt(Float64(nsquared)))
        k_lattice = (2.0 / lattice.dx) * sqrt(
            sinpi(nx / n)^2 + sinpi(ny / n)^2 + sinpi(nz / n)^2,
        )
        k_discrete = (2.0 * pi / lattice.L) * sqrt(Float64(nsquared))
        k = use_lattice_momentum ? k_lattice : k_discrete
        idx = linear[ix, iy, iz]
        bins[idx] = bin
        kmode[idx] = k
        counts[bin + 1] += 1
        ksum[bin + 1] += k
    end

    kmean = zeros(Float64, length(counts))
    @inbounds for slot in eachindex(counts)
        counts[slot] > 0 && (kmean[slot] = ksum[slot] / counts[slot])
    end
    return (bins=bins, kmode=kmode, counts=counts, kmean=kmean)
end

function _cross_power_shells(
    transform_x::AbstractArray{<:Complex,3},
    transform_y::AbstractArray{<:Complex,3},
    state,
    model,
    lattice;
    use_lattice_momentum::Bool=true,
)
    size(transform_x) == size(transform_y) == (lattice.N, lattice.N, lattice.N) ||
        throw(DimensionMismatch("Fourier arrays and lattice sizes differ"))
    plan = _spectral_shell_plan(lattice; use_lattice_momentum=use_lattice_momentum)
    sums = zeros(Float64, length(plan.counts))
    @inbounds for idx in eachindex(transform_x)
        bin = plan.bins[idx]
        bin < 0 && continue
        sums[bin + 1] += real(transform_x[idx] * conj(transform_y[idx]))
    end

    rows = NamedTuple[]
    volume = lattice.L^3
    @inbounds for slot in eachindex(plan.counts)
        count = plan.counts[slot]
        count == 0 && continue
        kcode = plan.kmean[slot]
        averaged_covariance = sums[slot] / count
        power = kcode^3 * averaged_covariance / (_TWO_PI_SQUARED * volume)
        push!(rows, (
            k_bin=slot - 1,
            k_comoving=model.B * kcode,
            k_physical=model.B * kcode / state.a,
            n_modes=count,
            power_dimensionless=power,
        ))
    end
    return rows
end

"""
    scalar_cross_spectrum(map_x, map_y, state, model, lattice;
                          use_lattice_momentum=true)

Dimensionless isotropic cross spectrum using the documented DFT convention.
The zero mode is excluded and every nonzero FFT mode, including the Hermitian
partner of a real-field mode, contributes once to its shell count.
"""
function scalar_cross_spectrum(
    map_x::AbstractArray{<:Real,3},
    map_y::AbstractArray{<:Real,3},
    state,
    model,
    lattice;
    use_lattice_momentum::Bool=true,
)
    return _cross_power_shells(
        lattice_dft(map_x, lattice),
        lattice_dft(map_y, lattice),
        state,
        model,
        lattice;
        use_lattice_momentum=use_lattice_momentum,
    )
end

"""Dimensionless shell-averaged auto spectrum of a real scalar lattice map."""
function scalar_spectrum(map3, state, model, lattice)
    transform = lattice_dft(map3, lattice)
    return _cross_power_shells(transform, transform, state, model, lattice)
end

"""
    field_spectra(state, model, lattice, config, output_id)

Return long-form rows for every field auto spectrum and every `I < J` cross
spectrum.  The maps are `delta_phi_I = phi_I - mean(phi_I)`.
"""
function field_spectra(state, model, lattice, config, output_id)
    nf = _obs_nfields(state)
    length(model.field_names) == nf || throw(DimensionMismatch("field name count differs from state"))
    use_lattice = _spectra_use_lattice_momentum(config)
    transforms = Vector{Array{ComplexF64,3}}(undef, nf)
    @inbounds for q in 1:nf
        transforms[q] = lattice_dft(@view(state.field[:, :, :, q]), lattice)
    end

    rows = NamedTuple[]
    @inbounds for field_i in 1:nf
        for field_j in field_i:nf
            shells = _cross_power_shells(
                transforms[field_i],
                transforms[field_j],
                state,
                model,
                lattice;
                use_lattice_momentum=use_lattice,
            )
            for shell in shells
                push!(rows, (
                    output_id=output_id,
                    step=state.step,
                    efolds=state.efolds,
                    a=state.a,
                    k_bin=shell.k_bin,
                    k_comoving=shell.k_comoving,
                    k_physical=shell.k_physical,
                    n_modes=shell.n_modes,
                    field_i=String(model.field_names[field_i]),
                    field_j=String(model.field_names[field_j]),
                    power_dimensionless=shell.power_dimensionless,
                ))
            end
        end
    end
    return rows
end
