#!/usr/bin/env julia

module MFILAnalyzeCLI

using MultifieldInflationLattice

const EXIT_SUCCESS = 0
const EXIT_CONFIG = 2
const EXIT_IO = 3
const EXIT_NUMERICAL = 4
const EXIT_INTERRUPTED = 130

function usage(io::IO=stdout)
    println(io, "Usage:")
    println(io, "  julia --project=. bin/analyze.jl --input RUN_DIRECTORY")
    println(io)
    println(io, "Recompute supported analysis products from a completed or checkpointed run.")
end

function parse_arguments(arguments)
    input = nothing
    help = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            help = true
        elseif argument == "--input"
            index == length(arguments) && return (nothing, help, "--input requires a directory")
            input === nothing || return (nothing, help, "--input may be supplied only once")
            index += 1
            input = arguments[index]
        elseif startswith(argument, "--input=")
            input === nothing || return (nothing, help, "--input may be supplied only once")
            input = split(argument, '='; limit=2)[2]
            isempty(input) && return (nothing, help, "--input requires a directory")
        else
            return (nothing, help, "unknown argument: $(argument)")
        end
        index += 1
    end
    help && return (input, true, nothing)
    input === nothing && return (nothing, false, "missing --input RUN_DIRECTORY")
    return (input, false, nothing)
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
    return EXIT_NUMERICAL
end

function main(arguments=ARGS)
    run_directory, help, argument_error = parse_arguments(arguments)
    if help
        usage(stdout)
        return EXIT_SUCCESS
    elseif argument_error !== nothing
        println(stderr, "ERROR: ", argument_error)
        usage(stderr)
        return EXIT_CONFIG
    end
    isdir(run_directory) || begin
        println(stderr, "ERROR: run directory not found: ", abspath(run_directory))
        return EXIT_IO
    end
    try
        result = analyze_run(run_directory)
        println("Analysis completed: ", abspath(run_directory))
        if result !== nothing && hasproperty(result, :background)
            background = getproperty(result, :background)
            hasproperty(background, :step) && print("  step=", getproperty(background, :step))
            hasproperty(background, :efolds) && print(" efolds=", getproperty(background, :efolds))
            hasproperty(background, :H) && print(" H=", getproperty(background, :H))
            println()
        end
        return EXIT_SUCCESS
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        return error_code(error)
    end
end

end # module MFILAnalyzeCLI

if abspath(PROGRAM_FILE) == @__FILE__
    exit(MFILAnalyzeCLI.main(ARGS))
end
