# Independent homogeneous background + no-metric linear mode solver.
# Uses exact lattice momenta and the actual initial field/derivative arrays.
include("ensemble.jl")

function csvread(path)
    lines=readlines(path); names=Symbol.(split(lines[1],','))
    [Dict(names .=> strip.(split(line,','),'"')) for line in lines[2:end] if !isempty(line)]
end
number(row,key)=parse(Float64,row[key])

@inline function bg_rhs(y,m2)
    p1,p2,u1,u2=y
    epsH=(u1^2+u2^2)/2
    H2=(m2[1]*p1^2+m2[2]*p2^2)/(2*(3-epsH))
    H2>0 || error("nonpositive background H²")
    d=3-epsH
    return (u1,u2,-d*u1-m2[1]*p1/H2,-d*u2-m2[2]*p2/H2),(d,inv(H2))
end
@inline addstage(y,v,h)=ntuple(i->y[i]+h*v[i],4)

function geometry(config)
    lattice=MF.build_lattice(config)
    plan=MF._spectral_shell_plan(lattice;use_lattice_momentum=true)
    ids=Dict{NTuple{3,Int},Int}(); ks=Float64[]; mapping=zeros(Int,lattice.N^3)
    for idx in CartesianIndices((lattice.N,lattice.N,lattice.N))
        linear=LinearIndices((lattice.N,lattice.N,lattice.N))[idx]
        plan.bins[linear]>=0 || continue
        key=Tuple(sort([abs(MF.signed_mode_index(i,lattice.N)) for i in Tuple(idx)]))
        if !haskey(ids,key);push!(ks,MF.lattice_k2(lattice,key...));ids[key]=length(ks);end
        mapping[linear]=ids[key]
    end
    return (;lattice,plan,ks,mapping)
end

function background_cache(config,targets,kmax;dN=0.002,phase_step=0.02)
    B=MF.resolve_rescaling_B(config); m2=(config.model.masses ./ B).^2
    p=config.initial.field_means; v=config.initial.cosmic_velocities ./ B
    H0=sqrt((sum(m2.*p.^2)+sum(abs2,v))/6); a0=config.initial.scale_factor
    y=(p[1],p[2],v[1]/H0,v[2]/H0); N=0.0
    hs=Float64[]; coeff=NTuple{12,Float64}[]; endpoints=Int[]; backgrounds=NTuple{4,Float64}[]
    for target in targets
        while N<target-1e-13
            _,(_,ih)=bg_rhs(y,m2)
            # Limit the initial subhorizon phase advance; then a fixed e-fold cap.
            h=min(dN,phase_step/(kmax*exp(-N)*sqrt(ih)/a0),target-N)
            a,c0=bg_rhs(y,m2)
            b,c1=bg_rhs(addstage(y,a,h/2),m2)
            c,c2=bg_rhs(addstage(y,b,h/2),m2)
            d,c3=bg_rhs(addstage(y,c,h),m2)
            stage(cs,n)=(cs[1],cs[2],exp(-2n)*cs[2]/a0^2)
            push!(coeff,(stage(c0,N)...,stage(c1,N+h/2)...,stage(c2,N+h/2)...,stage(c3,N+h)...))
            push!(hs,h)
            y=ntuple(i->y[i]+h*(a[i]+2b[i]+2c[i]+d[i])/6,4)
            N+=h
        end
        push!(endpoints,length(hs));push!(backgrounds,y)
    end
    return (;hs,coeff,endpoints,backgrounds,H0,m2,a0,targets)
end

@inline function oscillator(q,p,h,c,mass2,k2)
    d0,d1,d2,d3=c[1],c[4],c[7],c[10]
    f0,f1,f2,f3=mass2*c[2]+k2*c[3],mass2*c[5]+k2*c[6],mass2*c[8]+k2*c[9],mass2*c[11]+k2*c[12]
    a=p;b=-d0*p-f0*q
    cc=p+h*b/2;d=-d1*cc-f1*(q+h*a/2)
    e=p+h*d/2;f=-d2*e-f2*(q+h*cc/2)
    g=p+h*f;z=-d3*g-f3*(q+h*e)
    return q+h*(a+2*cc+2*e+g)/6,p+h*(b+2*d+2*f+z)/6
end

function mode_transfer(bg,ks;verbose=false)
    T=zeros(Float64,4,2,length(ks),length(bg.targets))
    completed=Threads.Atomic{Int}(0)
    verbose && println("TRANSFER steps=",length(bg.hs)," modes=",length(ks)," threads=",Threads.nthreads())
    flush(stdout)
    Threads.@threads :static for m in eachindex(ks)
      k2=ks[m]
      for field in 1:2
        q1,p1,q2,p2=1.0,0.0,0.0,1.0;step=0
        for (t,last) in enumerate(bg.endpoints)
            @inbounds for j in step+1:last
                q1,p1=oscillator(q1,p1,bg.hs[j],bg.coeff[j],bg.m2[field],k2)
                q2,p2=oscillator(q2,p2,bg.hs[j],bg.coeff[j],bg.m2[field],k2)
            end
            T[:,field,m,t].=(q1,p1,q2,p2);step=last
        end
      end
      done=Threads.atomic_add!(completed,1)+1
      if verbose && (done%4096==0 || done==length(ks))
          println("TRANSFER completed=",done,"/",length(ks));flush(stdout)
      end
    end
    all(isfinite,T) || error("nonfinite transfer")
    T
end

function observations(root,statuses)
    records=NamedTuple[]
    for (trial,s) in enumerate(statuses)
        dir=joinpath(root,@sprintf("trial_%02d",trial))
        fieldcsv=csvread(joinpath(s["run_dir"],"field_spectra.csv"))
        zetacsv=csvread(joinpath(s["run_dir"],"linear_curvature_spectra.csv"))
        bgs=csvread(joinpath(s["run_dir"],"background.csv"))
        times=sort(unique(number(r,:efolds) for r in fieldcsv))
        for request in (5.0,20.0,40.0,60.0)
            request<s["final_efolds"] || continue
            N=times[argmin(abs.(times.-request))]
            N>0 || continue
            fr=filter(r->number(r,:efolds)==N,fieldcsv)
            zr=filter(r->number(r,:efolds)==N,zetacsv)
            br=only(filter(r->number(r,:efolds)==N,bgs))
            v=[number(br,:phi_velocity),number(br,:chi_velocity)]
            coeff=-number(br,:H).*v/sum(abs2,v)
            push!(records,(trial,snapshot="N$(Int(request))",N,fr,zr,coeff))
        end
        push!(records,(trial,snapshot="final",N=s["final_efolds"],
            fr=csvread(joinpath(dir,"final_field_spectra.csv")),
            zr=csvread(joinpath(dir,"final_curvature_spectra.csv")),coeff=s["curvature_coefficients"]))
    end
    records
end

function predict(config,initial,g,bg,T,t;coeff_override=nothing)
    n=g.lattice.N; nb=length(g.plan.counts); B=MF.resolve_rescaling_B(config)
    F=[MF.lattice_dft(initial.field[:,:,:,q],g.lattice) for q in 1:2]
    P=[MF.lattice_dft(initial.field_prime[:,:,:,q],g.lattice) for q in 1:2]
    y=bg.backgrounds[t]; u=[y[3],y[4]]; zcoeff=coeff_override===nothing ? -u/sum(abs2,u) : coeff_override
    meanpower=zeros(nb,3); cond=zeros(nb,3); variance=zeros(nb,3)
    parts=zeros(nb,3); hconf=bg.a0*bg.H0
    for idx in CartesianIndices(F[1])
        li=LinearIndices(F[1])[idx];slot=g.plan.bins[li]+1;slot>0 || continue
        m=g.mapping[li];k2=g.ks[m];mu=zeros(2);z=zeros(ComplexF64,2)
        for q in 1:2
            t1,t2=T[1,q,m,t],T[3,q,m,t]
            z[q]=t1*F[q][idx]+t2*P[q][idx]/hconf
            omega=sqrt(k2+(config.initial.include_effective_mass ? bg.a0^2*bg.m2[q] : 0.0))
            initialvar=B^2*g.lattice.volume/(2bg.a0^2*omega)
            mu[q]=initialvar*((t1-t2)^2+(t2*omega/hconf)^2)
            MF._mode_allowed(Tuple(idx)...,g.lattice,config.initial) || (mu[q]=0.0)
        end
        z1,z2=zcoeff.*z
        means=(mu[1],mu[2],sum(zcoeff.^2 .*mu))
        actual=(abs2(z[1]),abs2(z[2]),abs2(z1+z2))
        for q in 1:3
            meanpower[slot,q]+=means[q];cond[slot,q]+=actual[q]
            # Both conjugate partners counted; self-conjugate modes also have Var=2μ².
            variance[slot,q]+=2means[q]^2
        end
        parts[slot,1]+=abs2(z1);parts[slot,2]+=abs2(z2)
        parts[slot,3]+=2real(z1*conj(z2))
    end
    for slot in eachindex(g.plan.counts)
        g.plan.counts[slot]>0 || continue
        factor=g.plan.kmean[slot]^3/(2pi^2*g.lattice.volume*g.plan.counts[slot])
        meanpower[slot,:].*=factor;cond[slot,:].*=factor
        variance[slot,:].=sqrt.(variance[slot,:]).*factor;parts[slot,:].*=factor
    end
    (;meanpower,cond,sigma=variance,parts,zcoeff)
end

function compare(root;dN=0.002,completed_only=false)
    dirs=sort(filter(isdir,joinpath.(root,filter(x->startswith(x,"trial_"),readdir(root)))))
    completed_only && (dirs=collect(Iterators.takewhile(d->isfile(joinpath(d,"status.toml")),dirs)))
    isempty(dirs) && error("no completed trials")
    statuses=[TOML.parsefile(joinpath(d,"status.toml")) for d in dirs]
    configs=[MF.load_config(joinpath(d,"config.toml")) for d in dirs];cfg=configs[1]
    all(c.model.masses==cfg.model.masses && c.lattice.size==cfg.lattice.size &&
        c.lattice.box_size==cfg.lattice.box_size && c.initial.field_means==cfg.initial.field_means &&
        c.initial.cosmic_velocities==cfg.initial.cosmic_velocities for c in configs) || error("nonuniform model/lattice")
    records=observations(root,statuses);targets=sort(unique(r.N for r in records));g=geometry(cfg)
    println("THEORY targets=",length(targets)," distinct momenta=",length(g.ks));flush(stdout)
    coarse=background_cache(cfg,targets,sqrt(maximum(g.ks));dN,phase_step=0.02)
    TC=mode_transfer(coarse,g.ks;verbose=true)
    println("THEORY coarse complete");flush(stdout)
    fine=background_cache(cfg,targets,sqrt(maximum(g.ks));dN=dN/2,phase_step=0.01)
    TF=mode_transfer(fine,g.ks;verbose=true)
    println("THEORY fine complete");flush(stdout)
    initial=[open(deserialize,joinpath(d,"initial_state.bin")) for d in dirs]
    crossstats=NamedTuple[]
    for (trial,state) in enumerate(initial)
        F=[MF.lattice_dft(state.field[:,:,:,q],g.lattice) for q in 1:2]
        z=[ComplexF64[] for _ in 1:2];n=g.lattice.N;li=LinearIndices(F[1])
        for idx in CartesianIndices(F[1])
            i,j,k=Tuple(idx)
            neg=CartesianIndex(mod(-(i-1),n)+1,mod(-(j-1),n)+1,mod(-(k-1),n)+1)
            li[idx]<li[neg] || continue
            MF._mode_allowed(i,j,k,g.lattice,cfg.initial) || continue
            k2=MF.lattice_k2_at_index(g.lattice,i,j,k)
            for q in 1:2
                omega=sqrt(k2+(cfg.initial.include_effective_mass ? fine.a0^2*fine.m2[q] : 0.0))
                variance=statuses[trial]["B"]^2*g.lattice.volume/(2fine.a0^2*omega)
                push!(z[q],F[q][idx]/sqrt(variance))
            end
        end
        push!(crossstats,(trial,independent_complex_modes=length(z[1]),
            real_component_correlation=cor(real.(z[1]),real.(z[2])),
            imaginary_component_correlation=cor(imag.(z[1]),imag.(z[2])),
            mean_normalized_cross_real=real(mean(z[1].*conj.(z[2])))))
    end
    csvwrite(joinpath(root,"initial_cross_statistics.csv"),crossstats)
    rows=NamedTuple[]; convergence=NamedTuple[];decomposition=NamedTuple[]
    for r in records
        t=searchsortedfirst(targets,r.N);config=configs[r.trial]
        pred=predict(config,initial[r.trial],g,fine,TF,t)
        predc=predict(config,initial[r.trial],g,coarse,TC,t)
        matched=predict(config,initial[r.trial],g,fine,TF,t;coeff_override=r.coeff)
        for slot in eachindex(g.plan.counts)
            bin=slot-1;g.plan.counts[slot]>0 || continue
            zrows=filter(x->parse(Int,x[:k_bin])==bin,r.zr)
            zobs=number(only(filter(x->x[:component]=="total",zrows)),:power_dimensionless)
            parts=[number(only(filter(x->x[:component]=="self" && x[:field_i]==name,zrows)),:power_dimensionless) for name in ("phi","chi")]
            cross=2number(only(filter(x->x[:component]=="cross",zrows)),:power_dimensionless)
            push!(decomposition,(trial=r.trial,snapshot=r.snapshot,efolds=r.N,k_bin=bin,
                observed_total=zobs,observed_phi=parts[1],observed_chi=parts[2],observed_cross_twice=cross,
                conditioned_phi=pred.parts[slot,1],conditioned_chi=pred.parts[slot,2],conditioned_cross_twice=pred.parts[slot,3],
                relative_identity_residual=(zobs-sum(parts)-cross)/zobs))
            for (q,quantity) in enumerate(("phi","chi","zeta_linear"))
                obs=q==3 ? zobs : number(only(filter(x->parse(Int,x[:k_bin])==bin && x[:field_i]==quantity && x[:field_j]==quantity,r.fr)),:power_dimensionless)
                expected=pred.meanpower[slot,q];conditioned=pred.cond[slot,q]
                expected>0 && conditioned>0 && isfinite(obs) || error("invalid power at $(r.snapshot), $quantity, $bin")
                push!(rows,(trial=r.trial,seed=statuses[r.trial]["seed"],snapshot=r.snapshot,
                    efolds=r.N,quantity,k_bin=bin,n_modes=g.plan.counts[slot],k_comoving=g.plan.kmean[slot]*statuses[r.trial]["B"],
                    observed=obs,theory_mean=expected,theory_sigma=pred.sigma[slot,q],conditioned_prediction=conditioned,
                    prediction_with_lattice_projection=matched.cond[slot,q],
                    residual_fraction=obs/conditioned-1,ensemble_ratio=obs/expected,
                    conservative_band=2<=bin<=min(8,cfg.lattice.size÷4)))
                push!(convergence,(trial=r.trial,snapshot=r.snapshot,quantity,k_bin=bin,
                    mean_relative_change=abs(predc.meanpower[slot,q]/expected-1),
                    conditioned_relative_change=abs(predc.cond[slot,q]/conditioned-1)))
            end
        end
    end
    csvwrite(joinpath(root,"spectra_comparison.csv"),rows)
    csvwrite(joinpath(root,"theory_convergence.csv"),convergence)
    csvwrite(joinpath(root,"curvature_decomposition.csv"),decomposition)
    backgroundrows=[(efolds=N,phi=fine.backgrounds[t][1],chi=fine.backgrounds[t][2],
        phi_dN=fine.backgrounds[t][3],chi_dN=fine.backgrounds[t][4]) for (t,N) in enumerate(targets)]
    csvwrite(joinpath(root,"theory_background.csv"),backgroundrows)
    tomlwrite(joinpath(root,"theory_diagnostics.toml"),Dict("metric_perturbations"=>false,
        "dN_coarse"=>dN,"dN_fine"=>dN/2,"initial_phase_cap_coarse"=>0.02,"initial_phase_cap_fine"=>0.01,
        "max_mean_power_relative_change"=>maximum(r.mean_relative_change for r in convergence),
        "max_conditioned_power_relative_change"=>maximum(r.conditioned_relative_change for r in convergence),
        "max_curvature_decomposition_relative_residual"=>maximum(abs(r.relative_identity_residual) for r in decomposition),
        "distinct_lattice_momenta"=>length(g.ks),"equation"=>"Q_I,NN+(3-epsilon_H)Q_I,N+[k_lat²/(aH)²+m_I²/H²]Q_I=0",
        "curvature"=>"zeta_lin=-sum(u_I Q_I)/sum(u_I²); includes actual cross terms",
        "background"=>"independent homogeneous two-field RK4; no metric perturbations",
        "initialization"=>"actual initial field and derivative, not initial shell rescaling",
        "endpoint"=>"exact integration endpoints for every selected lattice output"))
    println("COMPARISON_FINISHED ",root)
end
if abspath(PROGRAM_FILE)==@__FILE__
    compare(abspath(ARGS[1]);completed_only="--completed-only" in ARGS)
end
