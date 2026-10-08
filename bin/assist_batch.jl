# Join an already-created ensemble queue, using the saved seeds/configuration.
# Workers use atomic trial locks and never repeat a completed trial.
include("ensemble.jl")
root=abspath(ARGS[1]);env=TOML.parsefile(joinpath(root,"environment.toml"))
base=TOML.parsefile(joinpath(root,"trial_01","config.toml"))
jobs=length(ARGS)>1 ? parse(Int,ARGS[2]) : 2
worker_threads=get(env,"threads",1)
queue=Channel{Int}(env["runs"])
for i in env["runs"]:-1:2;put!(queue,i);end
close(queue)
@sync for _ in 1:jobs
    @async for trial in queue
        dir=joinpath(root,@sprintf("trial_%02d",trial));mkpath(dir)
        isfile(joinpath(dir,"status.toml")) && continue
        path=joinpath(dir,"config.toml")
        if !isfile(path)
            cfg=deepcopy(base);cfg["initial"]["seed"]=env["seeds"][trial]
            cfg["output"]["root"]=joinpath(dir,"simulation")
            cfg["output"]["run_name"]="trial_$(trial)"
            tomlwrite(path,cfg)
        end
        println("ASSIST START ",trial);flush(stdout)
        cmd=`$(Base.julia_cmd()) --project=$PROJECT --compiled-modules=existing --threads=$worker_threads $(joinpath(@__DIR__,"ensemble.jl")) --worker $path`
        open(joinpath(dir,"parallel_process.log"),"w") do io
            run(pipeline(cmd;stdout=io,stderr=io))
        end
        println("ASSIST DONE ",trial);flush(stdout)
    end
end
