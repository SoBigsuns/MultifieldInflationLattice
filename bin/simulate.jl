#!/usr/bin/env julia

module MFILSimulateCLI

using MultifieldInflationLattice

const EXIT_SUCCESS = 0
const EXIT_CONFIG = 2
const EXIT_IO = 3
const EXIT_NUMERICAL = 4
const EXIT_INTERRUPTED = 130

function usage(io::IO=stdout)
    println(io, "Usage:")
    println(io, "  julia --project=. bin/simulate.jl --config PATH")
    println(io, "  julia --project=. bin/simulate.jl --resume CHECKPOINT.jld2")
    println(io)
    println(io, "Exactly one of --config and --resume is required.")
    println(io, "Set JULIA_NUM_THREADS before launch; FFT threads come from the TOML configuration.")
end

function parse_arguments(arguments)
    config = nothing
    checkpoint = nothing
    help = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            help = true
        elseif argument in ("--config", "--resume")
            index == length(arguments) && return (nothing, nothing, help, "$(argument) requires a path")
            index += 1
            if argument == "--config"
                config === nothing || return (nothing, nothing, help, "--config may be supplied only once")
                config = arguments[index]
            else
                checkpoint === nothing || return (nothing, nothing, help, "--resume may be supplied only once")
                checkpoint = arguments[index]
            end
        elseif startswith(argument, "--config=")
            config === nothing || return (nothing, nothing, help, "--config may be supplied only once")
            config = split(argument, '='; limit=2)[2]
        elseif startswith(argument, "--resume=")
            checkpoint === nothing || return (nothing, nothing, help, "--resume may be supplied only once")
            checkpoint = split(argument, '='; limit=2)[2]
        else
            return (nothing, nothing, help, "unknown argument: $(argument)")
        end
        index += 1
    end
    help && return (config, checkpoint, true, nothing)
    (config === nothing) == (checkpoint === nothing) && return (
        nothing, nothing, false, "exactly one of --config and --resume is required",
    )
    selected = config === nothing ? checkpoint : config
    isempty(selected) && return (nothing, nothing, false, "input path must not be empty")
    return (config, checkpoint, false, nothing)
end

function error_code(error)
    error isa InterruptException && return EXIT_INTERRUPTED
    name = lowercase(string(nameof(typeof(error))))
    occursin("config", name) && return EXIT_CONFIG
    error isa SystemError && return EXIT_IO
    error isa Base.IOError && return EXIT_IO
    error isa EOFError && return EXIT_IO
    occursin("checkpoint", name) && return EXIT_IO
    occursin("output", name) && return EXIT_IO
    error isa DomainError && return EXIT_NUMERICAL
    error isa OverflowError && return EXIT_NUMERICAL
    error isa InexactError && return EXIT_NUMERICAL
    occursin("numerical", name) && return EXIT_NUMERICAL
    return EXIT_NUMERICAL
end

function main(arguments=ARGS)
    config_path, checkpoint_path, help, argument_error = parse_arguments(arguments)
    if help
        usage(stdout)
        return EXIT_SUCCESS
    elseif argument_error !== nothing
        println(stderr, "ERROR: ", argument_error)
        usage(stderr)
        return EXIT_CONFIG
    end
    selected = config_path === nothing ? checkpoint_path : config_path
    isfile(selected) || begin
        println(stderr, "ERROR: input file not found: ", abspath(selected))
        return EXIT_IO
    end
    try
        result = config_path === nothing ? resume_simulation(checkpoint_path) : run_simulation(config_path)
        result === nothing || println("Completed: ", result)
        return EXIT_SUCCESS
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        return error_code(error)
    end
end

end # module MFILSimulateCLI

if abspath(PROGRAM_FILE) == @__FILE__
    exit(MFILSimulateCLI.main(ARGS))
end
