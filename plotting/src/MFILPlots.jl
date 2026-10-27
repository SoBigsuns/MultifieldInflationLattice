"""
    MFILPlots

Figures for a MultifieldInflationLattice run directory, read only from the CSV
products documented in each run's `data_dictionary.md` (schema `MFIL-CSV-1`).
The solver package is not loaded, so plotting works on copied run directories.

Entry point: [`plot_run`](@ref).  Figures are written to
`<output>/background/` and `<output>/fluctuations/`.
"""
module MFILPlots

using DelimitedFiles, Plots, Printf, Statistics, TOML

export plot_run, read_table, background_figures, fluctuation_figures

# ---------------------------------------------------------------------------
# CSV access
# ---------------------------------------------------------------------------

"""
    read_table(path) -> Dict{String,Vector}

Read one MFIL CSV file into column vectors.  Columns whose cells are all
numeric become `Vector{Float64}` (`NaN` preserved); other columns keep strings.
"""
function read_table(path::AbstractString)
    data, header = readdlm(path, ',', Any; header=true, quotes=true)
    table = Dict{String,Vector}()
    for (index, name) in enumerate(vec(header))
        column = data[:, index]
        if all(x -> x isa Real && !(x isa Bool), column)
            table[String(name)] = Float64.(column)
        else
            table[String(name)] = [x isa AbstractString ? String(x) : string(x) for x in column]
        end
    end
    return table
end

_table(run, name) = (path = joinpath(run, name); isfile(path) ? read_table(path) : nothing)

"""Lattice points per side from `config.effective.toml` (nothing if absent)."""
function lattice_size(run)
    path = joinpath(run, "config.effective.toml")
    isfile(path) || return nothing
    return get(get(TOML.parsefile(path), "lattice", Dict()), "size", nothing)
end

"""Keep only complete spherical shells, k_bin ≤ N_L/2; corner shells hold few modes."""
complete_shells(rows, NL) = NL === nothing ? rows : _rows(rows, rows["k_bin"] .<= NL ÷ 2)

"""Select rows of a column table by a boolean mask."""
_rows(table, mask) = Dict(name => column[mask] for (name, column) in table)

"""User-facing field names and their normalized CSV column names, from `data_dictionary.md`."""
function field_names(run, background)
    mapping = Pair{String,String}[]
    path = joinpath(run, "data_dictionary.md")
    if isfile(path)
        for line in eachline(path)
            m = match(r"^- `(.+)` -> `(.+)`$", line)
            m === nothing || push!(mapping, String(m.captures[1]) => String(m.captures[2]))
        end
    end
    if isempty(mapping) && background !== nothing
        for name in keys(background)
            endswith(name, "_mean") && push!(mapping, name[1:end-5] => name[1:end-5])
        end
        sort!(mapping; by=first)
    end
    return mapping
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Linear interpolation on increasing `xs`, clamped at both ends."""
function interpolate(x, xs, ys)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    return (1 - t) * ys[i] + t * ys[i+1]
end

"""
    select_outputs(ids, efolds; times=nothing, max_curves=6)

Choose output ids to draw: the outputs nearest to the requested e-folds, or up
to `max_curves` outputs evenly spaced in e-folds including the first and last.
"""
function select_outputs(ids, efolds; times=nothing, max_curves=6)
    order = sortperm(efolds)
    ids, efolds = ids[order], efolds[order]
    targets = times === nothing ?
        range(efolds[1], efolds[end]; length=min(max_curves, length(ids))) : times
    chosen = Int[]
    for target in targets
        id = ids[argmin(abs.(efolds .- target))]
        id in chosen || push!(chosen, id)
    end
    return chosen
end

"""Unique (output_id, efolds) pairs of a long-form table, sorted by e-folds."""
function outputs_of(table)
    pairs_ = unique(zip(Int.(table["output_id"]), table["efolds"]))
    sort!(pairs_; by=last)
    return first.(pairs_), last.(pairs_)
end

_colors(n) = n == 1 ? [palette(:viridis)[1]] : [get(cgrad(:viridis), t) for t in range(0, 0.9; length=n)]
_positive(x) = isfinite(x) && x > 0
_logsafe(v) = [_positive(x) ? x : NaN for x in v]

const DEFAULTS = (; framestyle=:box, grid=false, legendfontsize=7, guidefontsize=11,
    titlefontsize=11, tickfontsize=9, fontfamily="DejaVu Sans")

function _save(figure, directory, name, format)
    mkpath(directory)
    path = joinpath(directory, "$(name).$(format)")
    savefig(figure, path)
    return path
end

# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

"""
    background_figures(run, output; format="png") -> Vector{String}

Background evolution: field means and trajectory, Hubble rate, slow-roll
parameters, equation of state, energy budget, field variances and numerical
diagnostics (Friedmann residual, CFL ratio, step size).
"""
function background_figures(run, output; format="png")
    bg = _table(run, "background.csv")
    bg === nothing && (@warn "background.csv not found; skipping background figures"; return String[])
    sr = _table(run, "slowroll.csv"); en = _table(run, "energies.csv"); dg = _table(run, "diagnostics.csv")
    fields = field_names(run, bg)
    N = bg["efolds"]; dir = joinpath(output, "background"); written = String[]

    # (1) overview
    pmean = plot(; DEFAULTS..., xlabel="N", ylabel="⟨φ_I⟩  [M_Pl]", title="field means")
    for (name, column) in fields
        plot!(pmean, N, bg["$(column)_mean"]; label=name, lw=1.5)
    end
    pH = plot(N, _logsafe(bg["H"]); DEFAULTS..., yscale=:log10, xlabel="N", ylabel="H  [M_Pl]",
        label="H", title="Hubble rate", lw=1.5)
    peps = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel="slow-roll parameter", title="ε")
    plot!(peps, N, _logsafe(-bg["Hdot"] ./ bg["H"] .^ 2); label="ε_H = −Ḣ/H²", lw=1.5)
    if sr !== nothing
        plot!(peps, sr["efolds"], _logsafe(sr["epsilon_V"]); label="ε_V", lw=1.5, ls=:dash)
    end
    hline!(peps, [1.0]; c=:gray, ls=:dot, label="ε = 1")
    pw = plot(N, bg["w"]; DEFAULTS..., xlabel="N", ylabel="w = p/ρ", label="w", title="equation of state", lw=1.5)
    hline!(pw, [-1 / 3]; c=:gray, ls=:dot, label="w = −1/3")
    pen = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel="fraction of ρ", title="energy budget")
    if en !== nothing
        for (part, label) in (("potential_total", "potential"), ("kinetic_total", "kinetic"), ("gradient_total", "gradient"))
            plot!(pen, en["efolds"], _logsafe(en[part] ./ en["rho_total"]); label, lw=1.5)
        end
    end
    pdiag = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel="value", title="numerical diagnostics")
    if dg !== nothing
        plot!(pdiag, dg["efolds"], _logsafe(dg["friedmann_residual"]); label="Friedmann residual", lw=1.5)
        plot!(pdiag, dg["efolds"], _logsafe(dg["cfl_ratio"]); label="CFL ratio", lw=1.5)
    end
    figure = plot(pmean, pH, peps, pw, pen, pdiag; layout=(2, 3), size=(1500, 850),
        left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    push!(written, _save(figure, dir, "background_overview", format))

    # (2) field-space trajectory or phase portrait
    if length(fields) >= 2
        panels = Plots.Plot[]
        for a in 1:length(fields)-1, b in a+1:length(fields)
            (na, ca), (nb, cb) = fields[a], fields[b]
            p = plot(bg["$(ca)_mean"], bg["$(cb)_mean"]; DEFAULTS..., line_z=N, c=:viridis, lw=2,
                xlabel="⟨$(na)⟩", ylabel="⟨$(nb)⟩", label="", colorbar_title="N", title="field-space trajectory")
            scatter!(p, [bg["$(ca)_mean"][1]], [bg["$(cb)_mean"][1]]; m=:circle, c=:black, label="start")
            push!(panels, p)
        end
        figure = plot(panels...; layout=length(panels), size=(650 * min(length(panels), 3), 520 * cld(length(panels), 3)),
            left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    else
        name, column = fields[1]
        figure = plot(bg["$(column)_mean"], bg["$(column)_velocity"]; DEFAULTS..., line_z=N, c=:viridis, lw=2,
            xlabel="⟨$(name)⟩", ylabel="⟨d$(name)/dt⟩", label="", colorbar_title="N", title="phase portrait",
            size=(650, 520), left_margin=6Plots.mm)
    end
    push!(written, _save(figure, dir, "field_trajectory", format))

    # (3) slow-roll and multi-field geometry
    if sr !== nothing
        Ns = sr["efolds"]
        p1 = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel="ε_H", title="ε_H and field contributions")
        plot!(p1, Ns, _logsafe(sr["epsilon_H"]); label="ε_H", lw=2, c=:black)
        for (name, column) in fields
            haskey(sr, "epsilonH_$(column)") && plot!(p1, Ns, _logsafe(sr["epsilonH_$(column)"]); label="ε_H from $(name)", lw=1.5)
        end
        p2 = plot(Ns, sr["eta_parallel"]; DEFAULTS..., xlabel="N", ylabel="η_∥", label="η_∥", title="η_∥", lw=1.5,
            ylims=_robust_limits(sr["eta_parallel"]))
        p3 = plot(Ns, _logsafe(sr["turn_rate_over_H"]); DEFAULTS..., yscale=:log10, xlabel="N", ylabel="Ω/H",
            label="turn rate / H", title="trajectory turn rate", lw=1.5)
        p4 = plot(; DEFAULTS..., xlabel="N", ylabel="eigenvalues of V_IJ/V", title="η_V eigenvalues")
        for name in sort(filter(k -> startswith(k, "etaV_eig_"), collect(keys(sr))))
            plot!(p4, Ns, sr[name]; label=replace(name, "etaV_eig_" => "η_V,"), lw=1.5)
        end
        figure = plot(p1, p2, p3, p4; layout=(2, 2), size=(1150, 850), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
        push!(written, _save(figure, dir, "slowroll", format))
    end

    # (4) fluctuation variance and field-wise energy
    p1 = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel="Var(φ_I)  [M_Pl²]", title="field variances")
    for (name, column) in fields
        plot!(p1, N, _logsafe(bg["$(column)_variance"]); label=name, lw=1.5)
    end
    p2 = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel="energy density  [M_Pl⁴]", title="field-wise kinetic / gradient")
    if en !== nothing
        for (name, column) in fields
            plot!(p2, en["efolds"], _logsafe(en["kinetic_$(column)"]); label="kinetic $(name)", lw=1.5)
            plot!(p2, en["efolds"], _logsafe(en["gradient_$(column)"]); label="gradient $(name)", lw=1.5, ls=:dash)
        end
    end
    figure = plot(p1, p2; layout=(1, 2), size=(1150, 450), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    push!(written, _save(figure, dir, "field_variances", format))
    return written
end

"""Axis limits from the 1st–99th percentiles, ignoring the spikes of undefined ratios."""
function _robust_limits(values)
    finite = sort(filter(isfinite, values))
    isempty(finite) && return :auto
    lo = finite[max(1, round(Int, 0.01 * length(finite)))]
    hi = finite[max(1, round(Int, 0.99 * length(finite)))]
    pad = 0.1 * max(hi - lo, eps(abs(hi) + 1))
    return (lo - pad, hi + pad)
end

# ---------------------------------------------------------------------------
# Fluctuations
# ---------------------------------------------------------------------------

"""Comoving horizon scale aH at e-fold `N`, from background.csv."""
_aH(bg, N) = interpolate(N, bg["efolds"], bg["a"] .* bg["H"])

"""E-fold at which comoving `k` equals aH (NaN if never crossed in the run)."""
function _horizon_crossing(bg, k)
    aH = bg["a"] .* bg["H"]
    i = findfirst(>=(k), aH)
    (i === nothing || i == 1) && return NaN
    return interpolate(log(k), log.(aH[i-1:i]), bg["efolds"][i-1:i])
end

"""Spectrum curves P(k) at selected outputs, against comoving k and k/(aH)."""
function _spectrum_panels(rows, bg, chosen; ylabel, title, logy=true)
    colors = _colors(length(chosen))
    pk = plot(; DEFAULTS..., xscale=:log10, yscale=logy ? :log10 : :identity, xlabel="k  (comoving, M_Pl)", ylabel, title)
    pkh = plot(; DEFAULTS..., xscale=:log10, yscale=logy ? :log10 : :identity, xlabel="k / (aH)", ylabel, title="$(title) vs k/(aH)")
    for (c, id) in zip(colors, chosen)
        sel = Int.(rows["output_id"]) .== id
        k = rows["k_comoving"][sel]; P = rows["power_dimensionless"][sel]
        N = rows["efolds"][sel][1]
        y = logy ? _logsafe(P) : P
        lab = @sprintf("N = %.2f", N)
        plot!(pk, k, y; m=:circle, ms=2, lw=1.2, c, label=lab)
        plot!(pkh, k ./ _aH(bg, N), y; m=:circle, ms=2, lw=1.2, c, label=lab)
    end
    vline!(pkh, [1.0]; c=:gray, ls=:dot, label="k = aH")
    return pk, pkh
end

"""Power in a few k shells as a function of N, with their horizon crossings."""
function _evolution_panel(rows, bg; ylabel, title)
    bins = sort(unique(Int.(rows["k_bin"])))
    picks = unique([bins[1], bins[max(1, length(bins) ÷ 4)], bins[max(1, length(bins) ÷ 2)], bins[end]])
    p = plot(; DEFAULTS..., yscale=:log10, xlabel="N", ylabel, title)
    for (c, b) in zip(_colors(length(picks)), picks)
        sel = Int.(rows["k_bin"]) .== b
        order = sortperm(rows["efolds"][sel])
        N = rows["efolds"][sel][order]; P = rows["power_dimensionless"][sel][order]
        k = rows["k_comoving"][sel][1]
        plot!(p, N, _logsafe(P); c, lw=1.5, label=@sprintf("k_bin %d (k = %.3g)", b, k))
        Nx = _horizon_crossing(bg, k)
        isfinite(Nx) && vline!(p, [Nx]; c, ls=:dot, label="")
    end
    return p
end

"""
    fluctuation_figures(run, output; format="png", times=nothing, max_curves=6) -> Vector{String}

Fluctuation products: field auto spectra and cross correlations, their time
evolution, linear-curvature spectra, the nonlinear delta-N spectrum and
one-point PDF, field/curvature histograms, and 2D slices.
"""
function fluctuation_figures(run, output; format="png", times=nothing, max_curves=6)
    bg = _table(run, "background.csv")
    bg === nothing && (@warn "background.csv not found; skipping fluctuation figures"; return String[])
    dir = joinpath(output, "fluctuations"); written = String[]
    fields = first.(field_names(run, bg))
    NL = lattice_size(run)
    # Histograms and slices carry only output_id; background.csv maps it to e-folds.
    output_efolds = Dict(zip(Int.(bg["output_id"]), bg["efolds"]))

    # (1) field spectra
    fs = _table(run, "field_spectra.csv")
    fs === nothing || (fs = complete_shells(fs, NL))
    if fs !== nothing
        ids, efolds = outputs_of(fs)
        chosen = select_outputs(ids, efolds; times, max_curves)
        panels = Plots.Plot[]; evolution = Plots.Plot[]
        for f in fields
            rows = _rows(fs, (fs["field_i"] .== f) .& (fs["field_j"] .== f))
            isempty(rows["k_bin"]) && continue
            pk, pkh = _spectrum_panels(rows, bg, chosen; ylabel="P_$(f)(k)", title="field spectrum $(f)")
            push!(panels, pk, pkh)
            push!(evolution, _evolution_panel(rows, bg; ylabel="P_$(f)", title="growth of $(f) modes (dotted: k = aH)"))
        end
        if !isempty(panels)
            figure = plot(panels...; layout=(length(panels) ÷ 2, 2), size=(1250, 430 * (length(panels) ÷ 2)),
                left_margin=7Plots.mm, bottom_margin=7Plots.mm)
            push!(written, _save(figure, dir, "field_spectra", format))
            figure = plot(evolution...; layout=(length(evolution), 1), size=(800, 380 * length(evolution)),
                left_margin=7Plots.mm, bottom_margin=7Plots.mm)
            push!(written, _save(figure, dir, "field_spectra_evolution", format))
        end
        # cross correlation coefficient r_ij = P_ij / sqrt(P_ii P_jj)
        cross = Plots.Plot[]
        for a in 1:length(fields)-1, b in a+1:length(fields)
            fa, fb = fields[a], fields[b]
            p = plot(; DEFAULTS..., xscale=:log10, xlabel="k  (comoving, M_Pl)", ylabel="r = P_ij / √(P_ii P_jj)",
                title="cross correlation $(fa)–$(fb)", ylims=(-1.05, 1.05))
            for (c, id) in zip(_colors(length(chosen)), chosen)
                at(i, j) = _rows(fs, (Int.(fs["output_id"]) .== id) .& (fs["field_i"] .== i) .& (fs["field_j"] .== j))
                pab = at(fa, fb); paa = at(fa, fa); pbb = at(fb, fb)
                isempty(pab["k_bin"]) && continue
                r = pab["power_dimensionless"] ./ sqrt.(paa["power_dimensionless"] .* pbb["power_dimensionless"])
                plot!(p, pab["k_comoving"], r; c, m=:circle, ms=2, label=@sprintf("N = %.2f", pab["efolds"][1]))
            end
            hline!(p, [0.0]; c=:gray, ls=:dot, label="")
            push!(cross, p)
        end
        if !isempty(cross)
            figure = plot(cross...; layout=length(cross), size=(700 * min(length(cross), 2), 450 * cld(length(cross), 2)),
                left_margin=7Plots.mm, bottom_margin=7Plots.mm)
            push!(written, _save(figure, dir, "field_cross_correlation", format))
        end
    end

    # (2) linear curvature spectra
    lc = _table(run, "linear_curvature_spectra.csv")
    if lc !== nothing
        rows = complete_shells(_rows(lc, lc["component"] .== "total"), NL)
        rows = _rows(rows, isfinite.(rows["power_dimensionless"]))
        # ζ_lin divides by the mean trajectory speed, which is set by hand at the
        # initial time; start the curves at the first evolved output.
        rows = _rows(rows, rows["efolds"] .> minimum(rows["efolds"]))
        if !isempty(rows["k_bin"])
            ids, efolds = outputs_of(rows)
            chosen = select_outputs(ids, efolds; times, max_curves)
            pk, pkh = _spectrum_panels(rows, bg, chosen; ylabel="P_ζlin(k)", title="linear curvature ζ_lin")
            pe = _evolution_panel(rows, bg; ylabel="P_ζlin", title="ζ_lin in fixed k shells (dotted: k = aH)")
            plot!(pe; ylims=exp10.(_robust_limits(log10.(_logsafe(rows["power_dimensionless"])))))
            figure = plot(pk, pkh, pe; layout=@layout([a b; c]), size=(1250, 850), left_margin=7Plots.mm, bottom_margin=7Plots.mm)
            push!(written, _save(figure, dir, "linear_curvature_spectra", format))
        end
    end

    # (3) nonlinear delta N
    ds = _table(run, "deltaN_spectrum.csv"); dsum = _table(run, "deltaN_summary.csv")
    ds === nothing || (ds = complete_shells(ds, NL))
    hz = _table(run, joinpath("histograms", "zeta_deltan.csv"))
    if ds !== nothing || hz !== nothing
        panels = Plots.Plot[]
        if ds !== nothing
            p = plot(ds["k_comoving"], _logsafe(ds["power_dimensionless"]); DEFAULTS..., xscale=:log10, yscale=:log10,
                m=:circle, ms=3, lw=1.2, xlabel="k  (comoving, M_Pl)", ylabel="P_ζ(k)", label="δN",
                title="nonlinear δN curvature spectrum")
            if lc !== nothing
                last_lin = complete_shells(_rows(lc, (lc["component"] .== "total") .& isfinite.(lc["power_dimensionless"])), NL)
                if !isempty(last_lin["k_bin"])
                    ids, _ = outputs_of(last_lin)
                    final = _rows(last_lin, Int.(last_lin["output_id"]) .== ids[end])
                    plot!(p, final["k_comoving"], _logsafe(final["power_dimensionless"]); ls=:dash, lw=1.2,
                        label=@sprintf("ζ_lin at N = %.2f", final["efolds"][1]))
                end
            end
            push!(panels, p)
        end
        if hz !== nothing
            push!(panels, _pdf_panel(hz, "ζ_δN"; efolds=output_efolds, title="one-point PDF of ζ_δN (standardized)"))
        end
        if dsum !== nothing
            text = join([@sprintf("%s = %.4g", name, dsum[name][1]) for name in
                ("efolds_reference", "variance", "stddev", "skewness", "excess_kurtosis", "n_failed", "separate_universe_ratio")
                if haskey(dsum, name)], "\n")
            push!(panels, plot(; DEFAULTS..., framestyle=:none, axis=false, ticks=false, legend=false,
                annotations=(0.02, 0.5, Plots.text(text, 10, :left, "DejaVu Sans")), xlims=(0, 1), ylims=(0, 1),
                title="deltaN_summary.csv"))
        end
        figure = plot(panels...; layout=(1, length(panels)), size=(540 * length(panels), 480),
            left_margin=8Plots.mm, bottom_margin=9Plots.mm)
        push!(written, _save(figure, dir, "deltaN", format))
    end

    # (4) histograms of fields and linear curvature
    hdir = joinpath(run, "histograms")
    if isdir(hdir)
        for file in sort(readdir(hdir))
            endswith(file, ".csv") && file != "zeta_deltan.csv" || continue
            h = read_table(joinpath(hdir, file))
            quantity = h["quantity"][1]
            ids = unique(Int.(h["output_id"]))
            efs = [get(output_efolds, id, NaN) for id in ids]
            chosen = select_outputs(ids, all(isfinite, efs) ? efs : Float64.(ids); times, max_curves)
            p = _pdf_panel(h, quantity; ids=chosen, efolds=output_efolds, title="standardized PDF of $(quantity)")
            push!(written, _save(plot(p; size=(700, 480), left_margin=6Plots.mm), dir, "histogram_$(splitext(file)[1])", format))
        end
    end

    # (5) 2D slices
    sdir = joinpath(run, "slices")
    if isdir(sdir)
        groups = Dict{String,Vector{Tuple{Int,String}}}()
        for file in readdir(sdir)
            m = match(r"^(.+)_(\d+)_([xyz]\d+)\.csv$", file)
            m === nothing && continue
            push!(get!(groups, m.captures[1], Tuple{Int,String}[]), (parse(Int, m.captures[2]), file))
        end
        for (quantity, files) in sort(collect(groups); by=first)
            sort!(files)
            ids = first.(files)
            efs = [get(output_efolds, id, Float64(id)) for id in ids]
            chosen = select_outputs(ids, efs; times, max_curves=min(max_curves, 4))
            panels = Plots.Plot[]
            for id in chosen
                file = files[findfirst(==(id), ids)][2]
                s = read_table(joinpath(sdir, file))
                n = Int(maximum(s["i"]))
                grid = fill(NaN, n, n)
                for (i, j, v) in zip(Int.(s["i"]), Int.(s["j"]), s["value"])
                    grid[i, j] = v
                end
                δ = grid .- mean(filter(isfinite, grid))
                lim = maximum(abs, filter(isfinite, δ); init=0.0)
                push!(panels, heatmap(1:n, 1:n, permutedims(δ); DEFAULTS..., c=:RdBu, clims=(-lim, lim) .+ (lim == 0 ? (-1, 1) : (0, 0)),
                    aspect_ratio=:equal, xlims=(0.5, n + 0.5), ylims=(0.5, n + 0.5), xlabel="i", ylabel="j",
                    title=@sprintf("%s − mean, N = %.2f", quantity, get(output_efolds, id, NaN))))
            end
            figure = plot(panels...; layout=(1, length(panels)), size=(560 * length(panels), 460),
                left_margin=5Plots.mm, right_margin=8Plots.mm, bottom_margin=5Plots.mm)
            push!(written, _save(figure, dir, "slices_$(quantity)", format))
        end
    end
    return written
end

"""Standardized one-point PDFs (x − μ)/σ from histogram rows, with a unit Gaussian."""
function _pdf_panel(h, quantity; ids=nothing, efolds=nothing, title="PDF")
    ids === nothing && (ids = unique(Int.(h["output_id"])))
    p = plot(; DEFAULTS..., yscale=:log10, xlabel="($(quantity) − μ)/σ", ylabel="PDF", title)
    x = range(-5, 5; length=300)
    for (c, id) in zip(_colors(length(ids)), ids)
        sel = Int.(h["output_id"]) .== id
        center = h["bin_center"][sel]; density = h["density"][sel]
        width = h["bin_right"][sel] .- h["bin_left"][sel]
        w = density .* width
        sum(w) > 0 || continue
        μ = sum(w .* center) / sum(w); σ = sqrt(sum(w .* (center .- μ) .^ 2) / sum(w))
        σ > 0 || continue
        N = efolds === nothing ? NaN : get(efolds, id, NaN)
        label = isfinite(N) ? @sprintf("N = %.2f", N) : "output $(id)"
        plot!(p, (center .- μ) ./ σ, _logsafe(density .* σ); c, lw=1.2, label)
    end
    plot!(p, x, exp.(-x .^ 2 ./ 2) ./ sqrt(2π); c=:black, ls=:dash, label="Gaussian")
    plot!(p; ylims=(1e-5, 1))
    return p
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    plot_run(run; output=joinpath(run, "figures"), format="png", times=nothing, max_curves=6)

Write every background and fluctuation figure that the run's CSV products
support, and return the list of written files.  `times` selects the outputs
(nearest e-folds) drawn in time-resolved figures; otherwise `max_curves`
outputs are spaced evenly in e-folds.
"""
function plot_run(run; output=joinpath(run, "figures"), format="png", times=nothing, max_curves=6)
    isdir(run) || throw(ArgumentError("run directory not found: $(abspath(run))"))
    gr()
    written = vcat(background_figures(run, output; format),
        fluctuation_figures(run, output; format, times, max_curves))
    return written
end

end # module MFILPlots
