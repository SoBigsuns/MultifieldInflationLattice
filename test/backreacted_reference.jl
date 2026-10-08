using Test
include("../bin/backreacted_theory.jl")

@testset "Independent quadratic Fourier reference" begin
    raw=TOML.parsefile(joinpath(PROJECT,"configs","two_field_quadratic_zero_velocity_no_metric.toml"))
    raw["lattice"]["size"]=8;raw["output"]["slice_index"]=4
    raw["evolution"]["integrator"]="rk4";raw["initial"]["seed"]=1562660959514511085
    cfg=MF.config_from_dict(raw);init=MF.initialize_simulation(cfg);g=geometry(cfg)
    r=reference_initial(cfg,init.state,g);dy=similar(r.y)
    e=coupled_rhs!(dy,r.y,0.0,r,g.ks)
    serial_dy=copy(dy)
    ep=coupled_rhs!(dy,r.y,0.0,r,g.ks;parallel=true)
    @test dy ≈ serial_dy rtol=1e-13
    @test ep.H2 ≈ e.H2 rtol=1e-13
    real_energy=MF.energy_summary(init.state,init.model,init.lattice)
    @test e.kinetic ≈ real_energy.kinetic_total rtol=1e-10
    @test e.gradient ≈ real_energy.gradient_total rtol=1e-10
    @test e.potential ≈ real_energy.potential_total rtol=1e-12
    @test sqrt(e.H2) ≈ MF.hubble_code(init.state) rtol=1e-12
    # Compare two independent discretizations in time, FFT ODE and real-space RK4.
    state=deepcopy(init.state);integrator=MF.build_integrator("rk4",state)
    while state.efolds<0.1
        h=MF.choose_dtau(state,init.model,init.lattice,cfg)/2
        MF.step!(integrator,state,init.model,init.lattice,h)
    end
    result=coupled_reference(cfg,init.state,g,[state.efolds];dN=0.0005,phase=0.005)
    pred=predict(cfg,init.state,g,result.bg,result.transfer,1)
    fieldrows=MF.field_spectra(state,init.model,init.lattice,cfg,1)
    for row in fieldrows
        row.field_i==row.field_j || continue
        q=findfirst(==(row.field_i),cfg.model.field_names)
        @test row.power_dimensionless ≈ pred.cond[row.k_bin+1,q] rtol=1e-6
    end
    @test mean(state.field[:,:,:,1]) ≈ result.bg.backgrounds[1][1] rtol=1e-9
    @test mean(state.field[:,:,:,2]) ≈ result.bg.backgrounds[1][2] rtol=1e-9
    zetarows=MF.linear_curvature_spectra(state,init.model,init.lattice,cfg,1)
    for row in zetarows
        row.component=="total" || continue
        @test row.power_dimensionless ≈ pred.cond[row.k_bin+1,3] rtol=1e-6
    end
    # With no fluctuations the coupled reference must reduce to the old
    # homogeneous equations, even though modes can still be propagated.
    raw["initial"]["vacuum_fluctuations"]=false
    cfg0=MF.config_from_dict(raw);init0=MF.initialize_simulation(cfg0)
    ref0=coupled_reference(cfg0,init0.state,g,[0.5];dN=0.001,phase=0.01)
    bg0=background_cache(cfg0,[0.5],sqrt(maximum(g.ks));dN=0.001,phase_step=0.01)
    @test collect(ref0.bg.backgrounds[1]) ≈ collect(bg0.backgrounds[1]) rtol=1e-10
end
