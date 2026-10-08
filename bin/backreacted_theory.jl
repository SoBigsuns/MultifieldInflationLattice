# Independent finite-lattice Fourier reference, ONLY for two separable quadratic
# fields with a spatially homogeneous FLRW metric. Includes the exact realization's
# fluctuation stress-energy in Friedmann; no fit to evolved lattice data.
include("linear_theory.jl")

function reference_initial(config, initial, g)
    config.model.potential == "quadratic" || error("quadratic potential required")
    length(config.model.masses)==2 || error("exactly two fields required")
    config.model.reduced_planck_mass==1.0 || error("reference currently requires reduced Planck mass = 1")
    B=MF.resolve_rescaling_B(config); m2=(config.model.masses ./ B).^2
    a0=initial.a; H0=MF.hubble_code(initial)
    cov=zeros(3,2,length(g.ks))
    y=zeros(4+8length(g.ks))
    for q in 1:2
        y[q]=mean(initial.field[:,:,:,q])
        y[q+2]=mean(initial.field_prime[:,:,:,q])/(a0*H0)
        F=MF.lattice_dft(initial.field[:,:,:,q],g.lattice)
        P=MF.lattice_dft(initial.field_prime[:,:,:,q],g.lattice)/(a0*H0)
        for li in eachindex(F)
            m=g.mapping[li];m==0 && continue
            cov[1,q,m]+=abs2(F[li])/g.lattice.volume^2
            cov[2,q,m]+=real(F[li]*conj(P[li]))/g.lattice.volume^2
            cov[3,q,m]+=abs2(P[li])/g.lattice.volume^2
        end
    end
    for m in eachindex(g.ks),q in 1:2
        i=4+8(m-1)+4(q-1)
        y[i+1]=1; y[i+4]=1 # fundamental solutions (Q,Q_N)=(1,0),(0,1)
    end
    chunks=[floor(Int,(i-1)*length(g.ks)/Threads.nthreads())+1:floor(Int,i*length(g.ks)/Threads.nthreads())
            for i in 1:Threads.nthreads()]
    (;y,cov,m2,a0,H0,chunks,partials=zeros(3,length(chunks)))
end

@inline quadratic_form(a,b,c1,c2,c3)=max(0.0,a*a*c1+2a*b*c2+b*b*c3)
function mode_energy(y,ref,ks,modes,a)
    velocity2=0.0;potential=0.0;gradient=0.0
    @inbounds for m in modes,q in 1:2
        i=4+8(m-1)+4(q-1)
        c1,c2,c3=ref.cov[1,q,m],ref.cov[2,q,m],ref.cov[3,q,m]
        variance=quadratic_form(y[i+1],y[i+3],c1,c2,c3)
        velocity2+=quadratic_form(y[i+2],y[i+4],c1,c2,c3)
        potential+=ref.m2[q]*variance/2
        gradient+=ks[m]*variance/(2a*a)
    end
    (velocity2,potential,gradient)
end

function mode_rhs!(dy,y,ref,ks,modes,a,H2,damping)
    @inbounds for m in modes,q in 1:2
        i=4+8(m-1)+4(q-1)
        omega2=(ref.m2[q]+ks[m]/(a*a))/H2
        dy[i+1]=y[i+2];dy[i+2]=-damping*y[i+2]-omega2*y[i+1]
        dy[i+3]=y[i+4];dy[i+4]=-damping*y[i+4]-omega2*y[i+3]
    end
end

function coupled_rhs!(dy,y,N,ref,ks;parallel=length(ks)>4096)
    a=ref.a0*exp(N)
    if parallel && Threads.nthreads()>1
        Threads.@threads :static for chunk in eachindex(ref.chunks)
            vals=mode_energy(y,ref,ks,ref.chunks[chunk],a)
            for j in 1:3;ref.partials[j,chunk]=vals[j];end
        end
        velocity2=sum(view(ref.partials,1,:))
        potential=sum(view(ref.partials,2,:))
        gradient=sum(view(ref.partials,3,:))
    else
        velocity2,potential,gradient=mode_energy(y,ref,ks,eachindex(ks),a)
    end
    velocity2+=y[3]^2+y[4]^2
    potential+=(ref.m2[1]*y[1]^2+ref.m2[2]*y[2]^2)/2
    H2=(potential+gradient)/(3-velocity2/2)
    H2>0 && isfinite(H2) || error("invalid reference Friedmann constraint")
    epsH=velocity2/2+gradient/(3H2)
    damping=3-epsH
    dy[1]=y[3];dy[2]=y[4]
    dy[3]=-damping*y[3]-ref.m2[1]*y[1]/H2
    dy[4]=-damping*y[4]-ref.m2[2]*y[2]/H2
    if parallel && Threads.nthreads()>1
        Threads.@threads :static for chunk in eachindex(ref.chunks)
            mode_rhs!(dy,y,ref,ks,ref.chunks[chunk],a,H2,damping)
        end
    else
        mode_rhs!(dy,y,ref,ks,eachindex(ks),a,H2,damping)
    end
    (;H2,epsH,kinetic=H2*velocity2/2,gradient,potential)
end

function coupled_reference(config,initial,g,targets;dN=0.002,phase=0.02)
    ref=reference_initial(config,initial,g);y=copy(ref.y)
    k1=similar(y);k2=similar(y);k3=similar(y);k4=similar(y);work=similar(y)
    transfer=zeros(4,2,length(g.ks),length(targets))
    backgrounds=NTuple{4,Float64}[];energy=NamedTuple[]
    N=0.0;steps=0
    for (t,target) in enumerate(targets)
        target>=N || error("targets must increase")
        while N<target-1e-13
            e=coupled_rhs!(k1,y,N,ref,g.ks)
            frequency=sqrt((maximum(g.ks)/(ref.a0*exp(N))^2+maximum(ref.m2))/e.H2)
            h=min(dN,phase/frequency,target-N)
            @. work=y+h*k1/2
            coupled_rhs!(k2,work,N+h/2,ref,g.ks)
            @. work=y+h*k2/2
            coupled_rhs!(k3,work,N+h/2,ref,g.ks)
            @. work=y+h*k3
            coupled_rhs!(k4,work,N+h,ref,g.ks)
            @. y=y+h*(k1+2k2+2k3+k4)/6
            N+=h;steps+=1
        end
        push!(backgrounds,Tuple(y[1:4]))
        push!(energy,coupled_rhs!(k1,y,N,ref,g.ks))
        transfer[:,:,:,t].=reshape(view(y,5:length(y)),4,2,length(g.ks))
        println("REFERENCE_TARGET N=",target," steps=",steps," dN=",dN);flush(stdout)
    end
    println("COUPLED_REFERENCE steps=",steps," dN=",dN);flush(stdout)
    # predict() needs only these background fields and the fundamental solutions.
    bg=(;backgrounds,H0=ref.H0,m2=ref.m2,a0=ref.a0,targets)
    (;bg,transfer,energy,ref)
end

function compare_backreacted(root;dN=0.002)
    dirs=sort(filter(d->isfile(joinpath(d,"status.toml")),
        joinpath.(root,filter(x->startswith(x,"trial_"),readdir(root)))))
    isempty(dirs) && error("no completed trials")
    all(basename(d)==@sprintf("trial_%02d",i) for (i,d) in enumerate(dirs)) ||
        error("completed trials must be contiguous starting at trial_01")
    statuses=[TOML.parsefile(joinpath(d,"status.toml")) for d in dirs]
    records=observations(root,statuses)
    rows=NamedTuple[];backgroundrows=NamedTuple[];checks=NamedTuple[]
    for (trial,dir) in enumerate(dirs)
        cfg=MF.load_config(joinpath(dir,"config.toml"))
        initial=open(deserialize,joinpath(dir,"initial_state.bin"));g=geometry(cfg)
        rs=filter(r->r.trial==trial,records);targets=[r.N for r in rs]
        c=coupled_reference(cfg,initial,g,targets;dN,phase=0.02)
        f=coupled_reference(cfg,initial,g,targets;dN=dN/2,phase=0.01)
        for (t,r) in enumerate(rs)
            pc=predict(cfg,initial,g,c.bg,c.transfer,t)
            pf=predict(cfg,initial,g,f.bg,f.transfer,t)
            # Only the same-realization prediction is used: an ensemble mean on
            # this realization-dependent background is not an ensemble expectation.
            y=f.bg.backgrounds[t];e=f.energy[t]
            push!(backgroundrows,(trial,snapshot=r.snapshot,efolds=r.N,
                phi=y[1],chi=y[2],phi_dN=y[3],chi_dN=y[4],
                H_code=sqrt(e.H2),epsilon_H=e.epsH,
                kinetic_code=e.kinetic,gradient_code=e.gradient,potential_code=e.potential))
            for slot in eachindex(g.plan.counts)
                g.plan.counts[slot]>0 || continue;bin=slot-1
                for (q,quantity) in enumerate(("phi","chi","zeta_linear"))
                    obs=q==3 ? number(only(filter(x->parse(Int,x[:k_bin])==bin && x[:component]=="total",r.zr)),:power_dimensionless) :
                        number(only(filter(x->parse(Int,x[:k_bin])==bin && x[:field_i]==quantity && x[:field_j]==quantity,r.fr)),:power_dimensionless)
                    prediction=pf.cond[slot,q]
                    isfinite(prediction) && prediction>0 && isfinite(obs) || error("invalid comparison")
                    push!(rows,(trial,snapshot=r.snapshot,efolds=r.N,quantity,k_bin=bin,
                        n_modes=g.plan.counts[slot],k_comoving=g.plan.kmean[slot]*MF.resolve_rescaling_B(cfg),
                        observed=obs,conditioned_prediction=prediction,
                        residual_fraction=obs/prediction-1,
                        reference_relative_change=abs(pc.cond[slot,q]/prediction-1)))
                end
            end
        end
        initmodel=MF.build_model(cfg)
        # Initial energy check uses the real-space implementation as an independent
        # Parseval/normalization check, not as input to the reference evolution.
        er=MF.energy_summary(initial,initmodel,g.lattice)
        dy=similar(f.ref.y);ef=coupled_rhs!(dy,f.ref.y,0.0,f.ref,g.ks)
        push!(checks,(trial,initial_energy_relative_error=abs((ef.kinetic+ef.gradient+ef.potential)/er.rho-1)))
    end
    csvwrite(joinpath(root,"backreacted_comparison.csv"),rows)
    csvwrite(joinpath(root,"backreacted_background.csv"),backgroundrows)
    csvwrite(joinpath(root,"backreacted_initial_checks.csv"),checks)
    tomlwrite(joinpath(root,"backreacted_diagnostics.toml"),Dict(
        "scope"=>"two separable quadratic fields; same finite lattice, no metric perturbations",
        "background"=>"independently evolved with exact initial realization stress-energy; no fitted lattice trajectory",
        "max_reference_relative_change"=>maximum(r.reference_relative_change for r in rows),
        "max_initial_energy_relative_error"=>maximum(r.initial_energy_relative_error for r in checks),
        "dN_coarse"=>dN,"dN_fine"=>dN/2))
    println("BACKREACTED_COMPARISON_FINISHED ",root)
end
if abspath(PROGRAM_FILE)==@__FILE__
    compare_backreacted(abspath(ARGS[1]))
end
