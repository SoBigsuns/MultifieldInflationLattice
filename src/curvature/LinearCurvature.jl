function _linear_zeta_min_speed2(config)
    section = hasproperty(config, :spectra) ? getproperty(config, :spectra) : config
    return hasproperty(section, :linear_zeta_min_speed2) ?
        Float64(getproperty(section, :linear_zeta_min_speed2)) : 1.0e-30
end

"""
    linear_curvature_coefficients(state; min_speed2=1e-30)

Coefficients `c_I` in `zeta_I = c_I delta_phi_I`, where

`c_I = -H * mean(dot(phi_I)) / sum_J mean(dot(phi_J))^2`.

Code-unit `H` and velocities may be used because all factors of `model.B`
cancel.  The returned named tuple has `defined=false` and NaN coefficients
when the mean code-speed squared is below `min_speed2`.  Callers accepting a
physical configuration threshold must divide it by `model.B^2` first.
"""
function linear_curvature_coefficients(state; min_speed2::Real=1.0e-30)
    velocities = mean_field_velocities_code(state)
    speed2 = sum(abs2, velocities)
    Hcode = hubble_code(state)
    defined = isfinite(speed2) && speed2 >= min_speed2 && isfinite(Hcode)
    coefficients = defined ? (-Hcode / speed2) .* velocities : fill(NaN, length(velocities))
    return (
        defined=defined,
        speed2=speed2,
        H_code=Hcode,
        velocities_code=velocities,
        coefficients=coefficients,
    )
end

"""
    linear_curvature_maps(state; min_speed2=1e-30)

Build each field contribution and their sum directly from the nonlinear
lattice configuration.  No independent linear mode equation is solved.
"""
function linear_curvature_maps(state; min_speed2::Real=1.0e-30)
    info = linear_curvature_coefficients(state; min_speed2=min_speed2)
    nx, ny, nz, nf = size(state.field)
    components = Vector{Array{Float64,3}}(undef, nf)
    total = zeros(Float64, nx, ny, nz)
    means = field_means(state)

    if !info.defined
        fill!(total, NaN)
        @inbounds for q in 1:nf
            components[q] = fill(NaN, nx, ny, nz)
        end
        return merge(info, (components=components, total=total))
    end

    @inbounds for q in 1:nf
        contribution = Array{Float64}(undef, nx, ny, nz)
        coefficient = info.coefficients[q]
        mean_field = means[q]
        for k in 1:nz, j in 1:ny, i in 1:nx
            value = coefficient * (state.field[i, j, k, q] - mean_field)
            contribution[i, j, k] = value
            total[i, j, k] += value
        end
        components[q] = contribution
    end
    return merge(info, (components=components, total=total))
end

"""Build curvature maps using the physical speed threshold from `config`."""
function linear_curvature_maps(state, model, lattice, config)
    threshold_physical = _linear_zeta_min_speed2(config)
    threshold_code = threshold_physical / (model.B * model.B)
    return linear_curvature_maps(state; min_speed2=threshold_code)
end

"""Return only the total linear-curvature map for the configured threshold."""
function linear_curvature_map(state; min_speed2::Real=1.0e-30)
    return linear_curvature_maps(state; min_speed2=min_speed2).total
end

function linear_curvature_map(state, config::SimulationConfig)
    threshold_physical = _linear_zeta_min_speed2(config)
    B = resolve_rescaling_B(config)
    threshold_code = threshold_physical / (B * B)
    return linear_curvature_map(state; min_speed2=threshold_code)
end

function linear_curvature_map(state, model, lattice, config)
    threshold_physical = _linear_zeta_min_speed2(config)
    threshold_code = threshold_physical / (model.B * model.B)
    return linear_curvature_map(state; min_speed2=threshold_code)
end

function _undefined_linear_curvature_rows(state, model, lattice, output_id, use_lattice)
    plan = _spectral_shell_plan(lattice; use_lattice_momentum=use_lattice)
    rows = NamedTuple[]
    nf = _obs_nfields(state)
    function append_component!(component, field_i, field_j)
        @inbounds for slot in eachindex(plan.counts)
            count = plan.counts[slot]
            count == 0 && continue
            kcode = plan.kmean[slot]
            push!(rows, (
                output_id=output_id,
                step=state.step,
                efolds=state.efolds,
                a=state.a,
                k_bin=slot - 1,
                k_comoving=model.B * kcode,
                k_physical=model.B * kcode / state.a,
                n_modes=count,
                component=component,
                field_i=field_i,
                field_j=field_j,
                power_dimensionless=NaN,
            ))
        end
    end
    @inbounds for q in 1:nf
        name = String(model.field_names[q])
        append_component!("self", name, name)
    end
    if nf > 1
        @inbounds for q in 1:(nf - 1), r in (q + 1):nf
            append_component!("cross", String(model.field_names[q]), String(model.field_names[r]))
        end
    end
    append_component!("total", "", "")
    return rows
end

"""
    linear_curvature_spectra(state, model, lattice, config, output_id)

Return long-form spectra for all individual `zeta_I` auto terms, all `I<J`
cross terms, and `zeta_lin=sum_I zeta_I`.  Cross rows contain `P_IJ`; the total
is checked against `sum(P_II) + 2sum(P_IJ)` shell by shell.
"""
function linear_curvature_spectra(state, model, lattice, config, output_id)
    nf = _obs_nfields(state)
    length(model.field_names) == nf || throw(DimensionMismatch("field name count differs from state"))
    threshold = _linear_zeta_min_speed2(config)
    threshold_code = threshold / (model.B * model.B)
    info = linear_curvature_coefficients(state; min_speed2=threshold_code)
    use_lattice = _spectra_use_lattice_momentum(config)

    if !info.defined
        @warn "linear curvature is undefined: physical mean field speed squared is below threshold" speed2=model.B^2 * info.speed2 threshold=threshold step=state.step
        return _undefined_linear_curvature_rows(state, model, lattice, output_id, use_lattice)
    end

    transforms = Vector{Array{ComplexF64,3}}(undef, nf)
    total_transform = zeros(ComplexF64, lattice.N, lattice.N, lattice.N)
    @inbounds for q in 1:nf
        field_transform = lattice_dft(@view(state.field[:, :, :, q]), lattice)
        transforms[q] = info.coefficients[q] .* field_transform
        total_transform .+= transforms[q]
    end

    rows = NamedTuple[]
    component_shells = Vector{Vector{NamedTuple}}()
    @inbounds for q in 1:nf
        shells = _cross_power_shells(
            transforms[q], transforms[q], state, model, lattice;
            use_lattice_momentum=use_lattice,
        )
        push!(component_shells, shells)
        for shell in shells
            name = String(model.field_names[q])
            push!(rows, (
                output_id=output_id,
                step=state.step,
                efolds=state.efolds,
                a=state.a,
                k_bin=shell.k_bin,
                k_comoving=shell.k_comoving,
                k_physical=shell.k_physical,
                n_modes=shell.n_modes,
                component="self",
                field_i=name,
                field_j=name,
                power_dimensionless=shell.power_dimensionless,
            ))
        end
    end

    cross_shells = Vector{Vector{NamedTuple}}()
    if nf > 1
        @inbounds for q in 1:(nf - 1), r in (q + 1):nf
            shells = _cross_power_shells(
                transforms[q], transforms[r], state, model, lattice;
                use_lattice_momentum=use_lattice,
            )
            push!(cross_shells, shells)
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
                    component="cross",
                    field_i=String(model.field_names[q]),
                    field_j=String(model.field_names[r]),
                    power_dimensionless=shell.power_dimensionless,
                ))
            end
        end
    end

    total_shells = _cross_power_shells(
        total_transform, total_transform, state, model, lattice;
        use_lattice_momentum=use_lattice,
    )
    @inbounds for shell_index in eachindex(total_shells)
        shell = total_shells[shell_index]
        decomposed = 0.0
        for component in component_shells
            decomposed += component[shell_index].power_dimensionless
        end
        for cross in cross_shells
            decomposed += 2.0 * cross[shell_index].power_dimensionless
        end
        if !isapprox(shell.power_dimensionless, decomposed; rtol=2.0e-11, atol=1.0e-14)
            @warn "linear-curvature spectral decomposition lost consistency" k_bin=shell.k_bin direct=shell.power_dimensionless decomposed=decomposed
        end
        push!(rows, (
            output_id=output_id,
            step=state.step,
            efolds=state.efolds,
            a=state.a,
            k_bin=shell.k_bin,
            k_comoving=shell.k_comoving,
            k_physical=shell.k_physical,
            n_modes=shell.n_modes,
            component="total",
            field_i="",
            field_j="",
            power_dimensionless=shell.power_dimensionless,
        ))
    end
    return rows
end
