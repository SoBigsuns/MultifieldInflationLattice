using Test
include("../bin/linear_theory.jl")
@testset "Independent mode integrator" begin
    coeff=(0.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.)
    function endpoint(h)
        q,p=1.,0.
        for _ in 1:round(Int,2/h);q,p=oscillator(q,p,h,coeff,2.,0.);end
        q,p
    end
    exact=(cos(2sqrt(2.)),-sqrt(2.)*sin(2sqrt(2.)))
    coarse=endpoint(.02);fine=endpoint(.01)
    @test maximum(abs.(fine.-exact))<1e-8
    @test maximum(abs.(coarse.-exact))/maximum(abs.(fine.-exact))>14
    config=MF.load_config(joinpath(PROJECT,"configs","two_field_quadratic_zero_velocity_no_metric.toml"))
    bg=background_cache(config,[.2,2.,10.],0.;dN=.002)
    transfer=mode_transfer(bg,[0.])
    # At k=0, each homogeneous field solves the same no-metric field equation.
    for t in eachindex(bg.targets),q in 1:2
        predicted=transfer[1,q,1,t]*config.initial.field_means[q]
        @test isapprox(predicted,bg.backgrounds[t][q];rtol=2e-10)
    end
    lattice=MF.build_lattice(config); plan=MF._spectral_shell_plan(lattice)
    @test sum(plan.counts)==lattice.N^3-1
end
