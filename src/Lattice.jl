"""Construct a periodic cubic lattice in code spatial units."""
function Lattice3D(N::Integer, L::Real)
    n = Int(N)
    n >= 2 || throw(ArgumentError("lattice size must be at least 2"))
    length_value = Float64(L)
    isfinite(length_value) && length_value > 0 || throw(ArgumentError(
        "lattice box length must be finite and positive",
    ))
    dx = length_value / n
    plus_index = [index == n ? 1 : index + 1 for index in 1:n]
    minus_index = [index == 1 ? n : index - 1 for index in 1:n]
    kmin_lat = 2.0 * sinpi(1 / n) / dx
    kmax_lat = 2.0 * sqrt(3.0) / dx
    return Lattice3D(
        n,
        length_value,
        dx,
        inv(dx * dx),
        length_value^3,
        kmin_lat,
        kmax_lat,
        plus_index,
        minus_index,
    )
end

"""Build and validate the lattice selected by a typed lattice configuration."""
function build_lattice(config::LatticeConfig)
    config.size >= 8 && ispow2(config.size) || throw(ArgumentError(
        "configured lattice size must be a power of two and at least 8",
    ))
    config.boundary == "periodic" || throw(ArgumentError(
        "only periodic boundaries are implemented",
    ))
    config.laplacian == "second_order_7point" || throw(ArgumentError(
        "only the second-order seven-point Laplacian is implemented",
    ))
    return Lattice3D(config.size, config.box_size)
end

build_lattice(config::SimulationConfig) = build_lattice(config.lattice)

"""
    signed_mode_index(index, N)

Map a one-based FFT array index to its signed integer wave number.  The even
grid's Nyquist entry is represented by `+N/2`; its sign is immaterial to all
quadratic momentum quantities.
"""
@inline function signed_mode_index(index::Integer, N::Integer)
    1 <= index <= N || throw(BoundsError(1:N, index))
    raw = Int(index) - 1
    return raw <= fld(N, 2) ? raw : raw - N
end

"""Periodic one-based lattice index."""
@inline periodic_index(index::Integer, N::Integer) = mod1(index, N)

"""Coordinate `(index-1)dx` in code units for a one-based lattice index."""
@inline function lattice_coordinate(lattice::Lattice3D, index::Integer)
    1 <= index <= lattice.N || throw(BoundsError(1:lattice.N, index))
    return (index - 1) * lattice.dx
end

"""
    lattice_k2(lattice, nx, ny, nz)

Seven-point-Laplacian effective wave number squared for signed integer Fourier
modes.  It satisfies `laplacian!(exp(i k_disc x)) = -k_lat^2 exp(i k_disc x)`.
"""
@inline function lattice_k2(
    lattice::Lattice3D,
    nx::Integer,
    ny::Integer,
    nz::Integer,
)
    return 4.0 * lattice.inv_dx2 * (
        sinpi(nx / lattice.N)^2 +
        sinpi(ny / lattice.N)^2 +
        sinpi(nz / lattice.N)^2
    )
end

lattice_k2(lattice::Lattice3D, mode::NTuple{3,<:Integer}) =
    lattice_k2(lattice, mode...)

"""Effective wave number squared at one-based FFT array indices."""
@inline function lattice_k2_at_index(
    lattice::Lattice3D,
    ix::Integer,
    iy::Integer,
    iz::Integer,
)
    return lattice_k2(
        lattice,
        signed_mode_index(ix, lattice.N),
        signed_mode_index(iy, lattice.N),
        signed_mode_index(iz, lattice.N),
    )
end

@inline lattice_k(lattice::Lattice3D, nx::Integer, ny::Integer, nz::Integer) =
    sqrt(lattice_k2(lattice, nx, ny, nz))

@inline lattice_k_at_index(lattice::Lattice3D, ix::Integer, iy::Integer, iz::Integer) =
    sqrt(lattice_k2_at_index(lattice, ix, iy, iz))

"""Continuum/discrete Fourier wave number squared for signed integer modes."""
@inline function discrete_k2(
    lattice::Lattice3D,
    nx::Integer,
    ny::Integer,
    nz::Integer,
)
    factor = 2.0 * pi / lattice.L
    return factor^2 * (nx^2 + ny^2 + nz^2)
end

discrete_k2(lattice::Lattice3D, mode::NTuple{3,<:Integer}) =
    discrete_k2(lattice, mode...)

@inline discrete_k(lattice::Lattice3D, nx::Integer, ny::Integer, nz::Integer) =
    sqrt(discrete_k2(lattice, nx, ny, nz))

"""Return lattice and continuum wave-number metadata for one FFT entry."""
function mode_wavenumbers(
    lattice::Lattice3D,
    ix::Integer,
    iy::Integer,
    iz::Integer,
)
    mode = (
        signed_mode_index(ix, lattice.N),
        signed_mode_index(iy, lattice.N),
        signed_mode_index(iz, lattice.N),
    )
    k2_lat = lattice_k2(lattice, mode)
    k2_disc = discrete_k2(lattice, mode)
    return (; mode, k2_lat, k_lat=sqrt(k2_lat), k2_disc, k_disc=sqrt(k2_disc))
end

"""Allocate the complete `k_lat^2` FFT-layout grid."""
function lattice_k2_array(lattice::Lattice3D)
    result = Array{Float64}(undef, lattice.N, lattice.N, lattice.N)
    @inbounds for iz in 1:lattice.N, iy in 1:lattice.N, ix in 1:lattice.N
        result[ix, iy, iz] = lattice_k2_at_index(lattice, ix, iy, iz)
    end
    return result
end

function _check_laplacian_arrays(out, field, lattice::Lattice3D, dimensions::Int)
    ndims(out) == dimensions || throw(DimensionMismatch(
        "Laplacian output must be $dimensions-dimensional",
    ))
    ndims(field) == dimensions || throw(DimensionMismatch(
        "Laplacian input must be $dimensions-dimensional",
    ))
    size(out) == size(field) || throw(DimensionMismatch(
        "Laplacian output and input must have identical dimensions",
    ))
    size(field, 1) == lattice.N && size(field, 2) == lattice.N &&
        size(field, 3) == lattice.N || throw(DimensionMismatch(
        "the first three array dimensions must match the lattice",
    ))
    Base.mightalias(out, field) && throw(ArgumentError(
        "in-place aliasing of Laplacian input and output is not supported",
    ))
    return nothing
end

"""
    laplacian!(out, field, lattice) -> out

Apply the periodic second-order seven-point Laplacian.  Three-dimensional
maps and four-dimensional `(x,y,z,field)` arrays are supported.  Spatial
planes are statically distributed over Julia CPU threads.
"""
function laplacian!(
    out::AbstractArray{T,3},
    field::AbstractArray{T,3},
    lattice::Lattice3D,
) where {T<:AbstractFloat}
    _check_laplacian_arrays(out, field, lattice, 3)
    n = lattice.N
    plus = lattice.plus_index
    minus = lattice.minus_index
    invdx2 = T(lattice.inv_dx2)
    Base.Threads.@threads :static for k in 1:n
        kp = plus[k]
        km = minus[k]
        @inbounds for j in 1:n
            jp = plus[j]
            jm = minus[j]
            for i in 1:n
                ip = plus[i]
                im = minus[i]
                out[i, j, k] = invdx2 * (
                    field[ip, j, k] + field[im, j, k] +
                    field[i, jp, k] + field[i, jm, k] +
                    field[i, j, kp] + field[i, j, km] -
                    T(6) * field[i, j, k]
                )
            end
        end
    end
    return out
end

function laplacian!(
    out::AbstractArray{T,4},
    field::AbstractArray{T,4},
    lattice::Lattice3D,
) where {T<:AbstractFloat}
    _check_laplacian_arrays(out, field, lattice, 4)
    n = lattice.N
    nf = size(field, 4)
    plus = lattice.plus_index
    minus = lattice.minus_index
    invdx2 = T(lattice.inv_dx2)
    Base.Threads.@threads :static for slab in 1:(n * nf)
        k = mod1(slab, n)
        field_index = fld(slab - 1, n) + 1
        kp = plus[k]
        km = minus[k]
        @inbounds for j in 1:n
            jp = plus[j]
            jm = minus[j]
            for i in 1:n
                ip = plus[i]
                im = minus[i]
                out[i, j, k, field_index] = invdx2 * (
                    field[ip, j, k, field_index] + field[im, j, k, field_index] +
                    field[i, jp, k, field_index] + field[i, jm, k, field_index] +
                    field[i, j, kp, field_index] + field[i, j, km, field_index] -
                    T(6) * field[i, j, k, field_index]
                )
            end
        end
    end
    return out
end
