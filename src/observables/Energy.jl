"""
    energy_summary(state, model, lattice, cache=nothing) -> EnergySummary

Compute volume-averaged matter energy in solver code units.  The gradient
term uses forward periodic edge differences; after summation this is exactly
the discrete integration-by-parts form `-mean(phi*laplacian(phi))` for the
seven-point Laplacian.  Potential energy is intentionally not split among
fields, since that split is not unique for an interacting potential.
"""
function energy_summary(state, model, lattice, cache=nothing)
    nx, ny, nz, nf = size(state.field)
    (nx == ny == nz == lattice.N) || throw(DimensionMismatch("state and lattice sizes differ"))
    size(state.field_prime) == size(state.field) ||
        throw(DimensionMismatch("field and field_prime sizes differ"))
    state.a > 0.0 || throw(DomainError(state.a, "scale factor must be positive"))

    # `threadid()` is a global id and may exceed `nthreads(:default)` when an
    # interactive pool exists, so partial arrays are sized by `maxthreadid()`.
    nt = Threads.maxthreadid()
    deterministic = cache !== nothing &&
        hasproperty(cache, :deterministic_reductions) &&
        getproperty(cache, :deterministic_reductions)
    reduction_slots = max(nz, nt)
    kinetic_partial = zeros(Float64, nf, reduction_slots)
    gradient_partial = zeros(Float64, nf, reduction_slots)
    potential_partial = zeros(Float64, reduction_slots)
    site_fields = zeros(Float64, nf, nt)
    inva2 = inv(state.a * state.a)
    invdx2 = inv(lattice.dx * lattice.dx)
    Threads.@threads :static for k in axes(state.field, 3)
        tid = Threads.threadid()
        reduction_slot = deterministic ? k : tid
        local_fields = @view site_fields[:, tid]
        kp = k == nz ? 1 : k + 1
        @inbounds for j in axes(state.field, 2), i in axes(state.field, 1)
            ip = i == nx ? 1 : i + 1
            jp = j == ny ? 1 : j + 1
            for q in 1:nf
                prime = state.field_prime[i, j, k, q]
                kinetic_partial[q, reduction_slot] += 0.5 * prime * prime * inva2

                center = state.field[i, j, k, q]
                local_fields[q] = center
                dxphi = state.field[ip, j, k, q] - center
                dyphi = state.field[i, jp, k, q] - center
                dzphi = state.field[i, j, kp, q] - center
                gradient_partial[q, reduction_slot] +=
                    0.5 * inva2 * invdx2 *
                    (dxphi * dxphi + dyphi * dyphi + dzphi * dzphi)
            end

            potential_partial[reduction_slot] +=
                potential_value(model.potential, local_fields)
        end
    end

    norm = inv(Float64(nx * ny * nz))
    kinetic_by_field = vec(sum(kinetic_partial; dims=2)) .* norm
    gradient_by_field = vec(sum(gradient_partial; dims=2)) .* norm
    kinetic_total = sum(kinetic_by_field)
    gradient_total = sum(gradient_by_field)
    potential_total = sum(potential_partial) * norm
    rho = kinetic_total + gradient_total + potential_total
    pressure = kinetic_total - gradient_total / 3.0 - potential_total
    w = iszero(rho) ? NaN : pressure / rho

    all(isfinite, (kinetic_total, gradient_total, potential_total, rho, pressure)) ||
        throw(DomainError(rho, "non-finite energy encountered"))
    return EnergySummary(
        kinetic_total,
        gradient_total,
        potential_total,
        rho,
        pressure,
        w,
        kinetic_by_field,
        gradient_by_field,
    )
end

"""
    physical_energy_summary(energy, model)

Convert a code-unit energy summary to reduced-Planck units.  Since
`rho_tilde = rho/B^2`, every dimensionful entry is multiplied by `B^2`.
"""
function physical_energy_summary(energy, model)
    B2 = model.B * model.B
    return EnergySummary(
        B2 * energy.kinetic_total,
        B2 * energy.gradient_total,
        B2 * energy.potential_total,
        B2 * energy.rho,
        B2 * energy.p,
        energy.w,
        B2 .* energy.kinetic_by_field,
        B2 .* energy.gradient_by_field,
    )
end

"""
    gradient_energy_direct(field3, a, lattice)

Direct periodic finite-difference gradient energy for one field, in code
units.  This helper is useful in discretization and Parseval regression tests.
"""
function gradient_energy_direct(field3::AbstractArray{<:Real,3}, a::Real, lattice)
    nx, ny, nz = size(field3)
    (nx == ny == nz == lattice.N) || throw(DimensionMismatch("field and lattice sizes differ"))
    total = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ip = i == nx ? 1 : i + 1
        jp = j == ny ? 1 : j + 1
        kp = k == nz ? 1 : k + 1
        center = field3[i, j, k]
        dxphi = field3[ip, j, k] - center
        dyphi = field3[i, jp, k] - center
        dzphi = field3[i, j, kp] - center
        total += dxphi * dxphi + dyphi * dyphi + dzphi * dzphi
    end
    return total / (2.0 * a * a * lattice.dx * lattice.dx * length(field3))
end
