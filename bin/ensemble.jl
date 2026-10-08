using MultifieldInflationLattice, Random, TOML, Dates, Serialization, SHA, Statistics, Printf
const MF = MultifieldInflationLattice
const PROJECT = dirname(@__DIR__)

function csvwrite(path, rows)
    isempty(rows) && error("empty table: $path")
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows
            println(io,join((x isa AbstractString ? "\""*replace(x,"\""=>"\"\"")*"\"" : string(x) for x in values(row)),','))
        end
    end
end
tomlwrite(path,data)=open(io->TOML.print(io,MF._io_plain(data)),path,"w")
arrayhash(x)=bytes2hex(sha256(reinterpret(UInt8,vec(x))))

function initial_checks(root, init, config)
    state, model, lattice=init.state,init.model,init.lattice
    h0=sqrt((MF.potential_value(model.potential,config.initial.field_means)+
             sum(abs2,config.initial.cosmic_velocities ./ model.B)/2)/3)
    n=lattice.N; linear=LinearIndices((n,n,n)); stats=NamedTuple[]; hist=NamedTuple[]
    hashes=String[]; primehashes=String[]
    transforms=[MF.lattice_dft(state.field[:,:,:,q],lattice) for q in 1:2]
    for q in 1:2
        field=state.field[:,:,:,q]; prime=state.field_prime[:,:,:,q]
        F=transforms[q]; P=MF.lattice_dft(prime,lattice)
        push!(hashes,arrayhash(field)); push!(primehashes,arrayhash(prime))
        re=Float64[]; imv=Float64[]; powers=Float64[]
        sumvar=0.0; sumprimevar=0.0; relation=Float64[]
        for idx in CartesianIndices(F)
            i,j,k=Tuple(idx); MF._mode_allowed(i,j,k,lattice,config.initial) || continue
            neg=CartesianIndex(mod(-(i-1),n)+1,mod(-(j-1),n)+1,mod(-(k-1),n)+1)
            omega=sqrt(MF.lattice_k2_at_index(lattice,i,j,k)+
                (config.initial.include_effective_mass ? (state.a*model.potential.masses[q])^2 : 0.0))
            mu=model.B^2*lattice.volume/(2state.a^2*omega)
            sumvar+=mu; sumprimevar+=(omega^2+(state.a*h0)^2)*mu
            if linear[idx]<linear[neg]
                push!(re,real(F[idx])/sqrt(mu/2)); push!(imv,imag(F[idx])/sqrt(mu/2))
                push!(powers,abs2(F[idx])/mu)
                push!(relation,abs(P[idx]-(-im*omega-state.a*h0)*F[idx])/sqrt((omega^2+(state.a*h0)^2)*mu))
            end
        end
        for (quantity,x) in (("mode_real_z",re),("mode_imag_z",imv),("mode_power_ratio",powers))
            avg=mean(x); variance=var(x;corrected=false); centered=x.-avg
            push!(stats,(field=model.field_names[q],quantity,samples=length(x),mean=avg,
                variance,skewness=mean(centered.^3)/variance^1.5,
                excess_kurtosis=mean(centered.^4)/variance^2-3,
                expected_mean=quantity=="mode_power_ratio" ? 1.0 : 0.0,expected_variance=1.0))
            edges=quantity=="mode_power_ratio" ? range(0,8;length=65) : range(-5,5;length=65)
            for b in 1:length(edges)-1
                countbin=count(v->edges[b]<=v<edges[b+1],x)
                push!(hist,(field=model.field_names[q],quantity,left=edges[b],right=edges[b+1],
                    count=countbin,density=countbin/(length(x)*(edges[b+1]-edges[b]))))
            end
        end
        tomlwrite(joinpath(root,"initial_$(model.field_names[q]).toml"),Dict(
            "field_seed"=>string(MF.field_seed(config.initial.seed,q)),"field_sha256"=>hashes[end],
            "prime_sha256"=>primehashes[end],"field_mean"=>mean(field),"prime_mean"=>mean(prime),
            "field_variance"=>var(field;corrected=false),"field_variance_expected"=>sumvar/lattice.volume^2,
            "prime_variance"=>var(prime;corrected=false),"prime_variance_expected"=>sumprimevar/lattice.volume^2,
            "max_normalized_initial_mode_relation_residual"=>maximum(relation)))
    end
    csvwrite(joinpath(root,"initial_statistics.csv"),stats)
    csvwrite(joinpath(root,"initial_histograms.csv"),hist)
    return hashes,primehashes
end

function worker_unlocked(configpath)
    config=MF.load_config(configpath); root=dirname(configpath)
    init=MF.initialize_simulation(config) # exactly once; these arrays enter the integrator
    open(io->serialize(io,init.state),joinpath(root,"initial_state.bin"),"w")
    hashes,primehashes=initial_checks(root,init,config)
    ctx=MF.create_run_directory(config,configpath)
    MF.initialize_outputs!(ctx,config,configpath,init.model,init.lattice)
    ctx.metadata["initial_diagnostics"]=MF._io_plain(init.initial_diagnostics)
    MF.write_metadata!(ctx)
    integrator=MF.build_integrator(config.evolution.integrator,init.state;
        deterministic_reductions=config.parallel.deterministic_reductions)
    started=time()
    run=MF._run_loop!(config,init.state,init.model,init.lattice,integrator,init.rng,ctx)
    elapsed=time()-started
    state,model,lattice=init.state,init.model,init.lattice
    # Explicit exact-final snapshots, independent of the scheduled CSV interval.
    csvwrite(joinpath(root,"final_field_spectra.csv"),MF.field_spectra(state,model,lattice,config,ctx.output_id+1))
    csvwrite(joinpath(root,"final_curvature_spectra.csv"),MF.linear_curvature_spectra(state,model,lattice,config,ctx.output_id+1))
    open(io->serialize(io,state),joinpath(root,"final_state.bin"),"w")
    info=MF.linear_curvature_coefficients(state)
    metadata=TOML.parsefile(joinpath(run,"metadata.toml"))
    tomlwrite(joinpath(root,"status.toml"),Dict("seed"=>config.initial.seed,
        "config"=>abspath(configpath),"run_dir"=>run,"elapsed_seconds"=>elapsed,
        "final_efolds"=>state.efolds,"steps"=>state.step,"B"=>model.B,
        "field_sha256"=>hashes,"prime_sha256"=>primehashes,
        "curvature_coefficients"=>info.coefficients,
        "end_reason"=>metadata["run"]["end_reason"],"initial_diagnostics"=>init.initial_diagnostics,
        "worker_source_sha256"=>bytes2hex(sha256(read(@__FILE__)))))
    println("WORKER_FINISHED ",root)
end

function worker(configpath)
    root=dirname(configpath);lock=joinpath(root,".worker-lock")
    # Claim a trial atomically: foreground and batch workers may share a queue.
    while true
        isfile(joinpath(root,"status.toml")) && return println("ALREADY_FINISHED ",root)
        try
            mkdir(lock)
            break
        catch err
            isdir(lock) || rethrow(err)
            sleep(1)
        end
    end
    try
        isfile(joinpath(root,"status.toml")) || worker_unlocked(configpath)
    finally
        rm(lock)
    end
end

function driver(args)
    runs=10; master=Int64(rand(RandomDevice(),UInt64)&UInt64(typemax(Int64)))
    smoke=false; output=""; factor=0.01; size=32; explicit_size=false; worker_threads=1
    integration_method="rk4"
    i=1
    while i<=length(args)
        key=args[i]
        if key=="--smoke";smoke=true;i+=1;continue;end
        i<length(args) || error("missing value: $key")
        value=args[i+1]
        if key=="--runs";runs=parse(Int,value)
        elseif key=="--master-seed";master=parse(Int64,value)
        elseif key=="--output";output=abspath(value)
        elseif key=="--timestep-factor";factor=parse(Float64,value)
        elseif key=="--size";size=parse(Int,value);explicit_size=true
        elseif key=="--worker-threads";worker_threads=parse(Int,value)
        elseif key=="--integrator";integration_method=value
        else;error("unknown option $key")
        end
        i+=2
    end
    runs>=1 || error("runs must be positive")
    size>=8 && ispow2(size) || error("size must be a power of two >= 8")
    worker_threads>=1 || error("worker-threads must be positive")
    integration_method in ("rk4","leapfrog") || error("integrator must be rk4 or leapfrog")
    rng=MersenneTwister(master); seeds=Int64[]
    while length(seeds)<runs
        seed=Int64(rand(rng,UInt64)&UInt64(typemax(Int64)))
        seed in seeds || push!(seeds,seed)
    end
    isempty(output) && (output=joinpath(PROJECT,"runs",(smoke ? "smoke_" : "ensemble_")*Dates.format(now(UTC),"yyyymmddTHHMMSSsssZ")))
    ispath(output) && error("output already exists: $output")
    mkpath(output)
    base=TOML.parsefile(joinpath(PROJECT,"configs","two_field_quadratic_zero_velocity_no_metric.toml"))
    base["model"]["field_names"]=["phi","chi"]
    base["lattice"]["size"]=size
    base["output"]["slice_index"]=size÷2
    base["evolution"]["timestep_factor"]=factor
    base["evolution"]["integrator"]=integration_method
    base["output"]["background_interval_efolds"]=0.25
    base["output"]["spectra_interval_efolds"]=0.5
    base["output"]["histogram_interval_efolds"]=5.0
    base["output"]["checkpoint_interval_efolds"]=10.0
    if smoke
        base["lattice"]["size"]=explicit_size ? size : 8
        base["output"]["slice_index"]=base["lattice"]["size"]÷2
        base["evolution"]["max_efolds"]=0.02
    end
    codehash=Dict(relpath(joinpath(d,f),PROJECT)=>bytes2hex(sha256(read(joinpath(d,f))))
        for folder in ("src","bin") for (d,_,files) in walkdir(joinpath(PROJECT,folder))
        for f in files if endswith(f,".jl"))
    tomlwrite(joinpath(output,"environment.toml"),Dict("master_seed"=>master,"seeds"=>seeds,
        "runs"=>runs,"smoke"=>smoke,"julia_version"=>string(VERSION),
        "threads"=>worker_threads,"fft_threads"=>1,"lattice_size"=>base["lattice"]["size"],"source_sha256"=>codehash))
    println("OUTPUT_DIR=",output);flush(stdout)
    for (trial,seed) in enumerate(seeds)
        dir=joinpath(output,@sprintf("trial_%02d",trial));mkpath(dir)
        cfg=deepcopy(base);cfg["initial"]["seed"]=seed
        cfg["output"]["root"]=joinpath(dir,"simulation")
        cfg["output"]["run_name"]="trial_$(trial)"
        path=joinpath(dir,"config.toml");tomlwrite(path,cfg)
        cmd=`$(Base.julia_cmd()) --project=$PROJECT --compiled-modules=existing --threads=$worker_threads $(joinpath(@__DIR__,"ensemble.jl")) --worker $path`
        println("START trial=",trial," seed=",seed);flush(stdout)
        open(joinpath(dir,"process.log"),"w") do io
            run(pipeline(cmd;stdout=io,stderr=io))
        end
        status=TOML.parsefile(joinpath(dir,"status.toml"))
        expected=smoke ? "max_efolds" : "epsilon_H_reached"
        status["end_reason"]==expected || error("unexpected end reason: $(status["end_reason"])")
        println("DONE trial=",trial," N=",status["final_efolds"]," seconds=",status["elapsed_seconds"]);flush(stdout)
    end
    statuses=[TOML.parsefile(joinpath(output,@sprintf("trial_%02d",i),"status.toml")) for i in 1:runs]
    all(length(unique(s["field_sha256"][q] for s in statuses))==runs for q in 1:2) || error("duplicate initial fields")
    all(length(unique(s["prime_sha256"][q] for s in statuses))==runs for q in 1:2) || error("duplicate initial derivatives")
    tomlwrite(joinpath(output,"summary.toml"),Dict("all_completed"=>true,"runs"=>runs,
        "unique_seeds_and_initial_arrays"=>true,"smoke"=>smoke))
    println("BATCH_FINISHED ",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    !isempty(ARGS) && ARGS[1]=="--worker" ? worker(abspath(ARGS[2])) : driver(ARGS)
end
