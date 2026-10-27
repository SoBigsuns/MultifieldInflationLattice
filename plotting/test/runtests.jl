using MFILPlots, Test

ENV["GKSwstype"] = "nul"

"""Write a tiny synthetic run directory in the MFIL-CSV-1 layout (two fields, 4³ lattice)."""
function synthetic_run(dir)
    fields = ("phi", "psi")
    outputs = 0:5
    N = collect(0.5 .* outputs)
    a = exp.(N); H = 1e-5 .* exp.(-0.01 .* N)
    write(joinpath(dir, "data_dictionary.md"), "# Data dictionary\n## Field-name mapping\n- `phi` -> `phi`\n- `psi` -> `psi`\n")
    write(joinpath(dir, "config.effective.toml"), "[lattice]\nsize = 4\n")
    open(joinpath(dir, "background.csv"), "w") do io
        println(io, "output_id,step,tau,t,efolds,a,H,Hdot,rho,p,w,phi_mean,phi_velocity,phi_variance,psi_mean,psi_velocity,psi_variance")
        for (i, id) in enumerate(outputs)
            println(io, join((id, 10id, 0.1id, 0.1id, N[i], a[i], H[i], -1e-13, 3H[i]^2, -2.9H[i]^2, -0.97,
                4 - 0.1N[i], -1e-7, 1e-10, 4.0, 0.0, 1e-10), ','))
        end
    end
    open(joinpath(dir, "slowroll.csv"), "w") do io
        println(io, "output_id,step,efolds,epsilon_H,epsilon_V,eta_parallel,turn_rate,turn_rate_over_H,epsilonH_phi,epsilonH_psi,etaV_diag_phi,etaV_diag_psi,etaV_eig_1,etaV_eig_2")
        for (i, id) in enumerate(outputs)
            println(io, join((id, 10id, N[i], 0.01, 0.012, NaN, 1e-8, 1e-3, 0.01, 1e-5, 0.1, 0.01, 0.01, 0.1), ','))
        end
    end
    open(joinpath(dir, "energies.csv"), "w") do io
        println(io, "output_id,step,efolds,kinetic_total,gradient_total,potential_total,rho_total,kinetic_phi,gradient_phi,kinetic_psi,gradient_psi")
        for (i, id) in enumerate(outputs)
            println(io, join((id, 10id, N[i], 1e-12, 1e-15, 1e-9, 1.001e-9, 1e-12, 5e-16, 1e-16, 5e-16), ','))
        end
    end
    open(joinpath(dir, "diagnostics.csv"), "w") do io
        println(io, "output_id,step,efolds,dtau,cfl_ratio,H2_constraint,friedmann_residual,field_min,field_max,finite_state")
        for (i, id) in enumerate(outputs)
            println(io, join((id, 10id, N[i], 1.0, 0.005, H[i]^2, 1e-9, 3.9, 4.1, true), ','))
        end
    end
    kbins = 1:3
    open(joinpath(dir, "field_spectra.csv"), "w") do io
        println(io, "output_id,step,efolds,a,k_bin,k_comoving,k_physical,n_modes,field_i,field_j,power_dimensionless")
        for (i, id) in enumerate(outputs), b in kbins, (fi, fj) in (("phi", "phi"), ("phi", "psi"), ("psi", "psi"))
            P = fi == fj ? 1e-10 * b^2 * exp(-N[i]) : 1e-12
            println(io, join((id, 10id, N[i], a[i], b, 1e-4b, 1e-4b / a[i], 6b^2, fi, fj, P), ','))
        end
    end
    open(joinpath(dir, "linear_curvature_spectra.csv"), "w") do io
        println(io, "output_id,step,efolds,a,k_bin,k_comoving,k_physical,n_modes,component,field_i,field_j,power_dimensionless")
        for (i, id) in enumerate(outputs), b in kbins
            println(io, join((id, 10id, N[i], a[i], b, 1e-4b, 1e-4b / a[i], 6b^2, "total", "", "", 1e-9 * b), ','))
            println(io, join((id, 10id, N[i], a[i], b, 1e-4b, 1e-4b / a[i], 6b^2, "self", "phi", "phi", 1e-9 * b), ','))
        end
    end
    open(joinpath(dir, "deltaN_spectrum.csv"), "w") do io
        println(io, "efolds_reference,rho_reference,k_bin,k_comoving,k_physical,n_modes,power_dimensionless")
        for b in kbins
            println(io, join((N[end], 3H[end]^2, b, 1e-4b, 1e-4b / a[end], 6b^2, 2e-9), ','))
        end
    end
    open(joinpath(dir, "deltaN_summary.csv"), "w") do io
        println(io, "efolds_reference,rho_reference,mean,variance,stddev,skewness,excess_kurtosis,minimum,maximum,n_sites,n_failed,separate_universe_ratio,separate_universe_ok")
        println(io, join((N[end], 3H[end]^2, 0.0, 1e-10, 1e-5, 0.0, 0.0, -3e-5, 3e-5, 64, 0, 2.0, true), ','))
    end
    mkpath(joinpath(dir, "histograms")); mkpath(joinpath(dir, "slices"))
    for (file, quantity, ids) in (("phi.csv", "phi", outputs), ("zeta_deltan.csv", "zeta_deltaN", [outputs[end]]))
        open(joinpath(dir, "histograms", file), "w") do io
            println(io, "quantity,output_id,bin_left,bin_right,bin_center,count,density")
            edges = range(-3, 3; length=21)
            for id in ids, b in 1:20
                c = (edges[b] + edges[b+1]) / 2
                println(io, join((quantity, id, edges[b], edges[b+1], c, 10, exp(-c^2 / 2) / sqrt(2π)), ','))
            end
        end
    end
    for id in (0, 5)
        open(joinpath(dir, "slices", "phi_$(lpad(id, 8, '0'))_z2.csv"), "w") do io
            println(io, "i,j,x,y,value")
            for j in 1:4, i in 1:4
                println(io, join((i, j, 0.1(i - 1), 0.1(j - 1), 4 + 1e-5 * sin(i + j + id)), ','))
            end
        end
    end
    return dir
end

@testset "MFILPlots" begin
    run = synthetic_run(mktempdir())
    @testset "CSV reader" begin
        table = read_table(joinpath(run, "slowroll.csv"))
        @test table["epsilon_H"] isa Vector{Float64}
        @test all(isnan, table["eta_parallel"])
        spectra = read_table(joinpath(run, "linear_curvature_spectra.csv"))
        @test spectra["component"][1] == "total"
        @test spectra["field_i"][1] == ""
    end
    @testset "all figures" begin
        written = plot_run(run; output=joinpath(run, "figs"))
        names = Set(basename.(written))
        for name in ("background_overview", "field_trajectory", "slowroll", "field_variances",
                     "field_spectra", "field_spectra_evolution", "field_cross_correlation",
                     "linear_curvature_spectra", "deltaN", "histogram_phi", "slices_phi")
            @test "$(name).png" in names
        end
        @test all(isfile, written)
        @test all(p -> filesize(p) > 1000, written)
    end
    @testset "time selection and formats" begin
        written = plot_run(run; output=joinpath(run, "figs_pdf"), format="pdf", times=[0.0, 2.5], max_curves=2)
        @test all(endswith(".pdf"), written)
    end
end
