#!/usr/bin/env julia

module MFILPlotCLI

using MFILPlots

const EXIT_SUCCESS = 0
const EXIT_CONFIG = 2
const EXIT_IO = 3
const EXIT_INTERRUPTED = 130
const FORMATS = ("png", "pdf", "svg")

function usage(io::IO=stdout)
    println(io, "Usage:")
    println(io, "  julia --project=plotting bin/plot.jl --input RUN_DIRECTORY [--output DIRECTORY]")
    println(io, "        [--format png|pdf|svg] [--times N1,N2,...] [--max-curves COUNT]")
    println(io)
    println(io, "Draw background and fluctuation figures from a run's CSV products.")
    println(io, "Figures go to OUTPUT/background and OUTPUT/fluctuations (default OUTPUT=RUN_DIRECTORY/figures).")
    println(io, "--times picks the outputs nearest to these e-folds for time-resolved figures;")
    println(io, "otherwise --max-curves outputs (default 6) are spaced evenly in e-folds.")
end

function parse_arguments(arguments)
    options = Dict{String,String}()
    help = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            help = true
        elseif startswith(argument, "--")
            name, value = occursin('=', argument) ? split(argument[3:end], '='; limit=2) :
                (argument[3:end], nothing)
            name in ("input", "output", "format", "times", "max-curves") ||
                return (options, help, "unknown argument: $(argument)")
            haskey(options, name) && return (options, help, "--$(name) may be supplied only once")
            if value === nothing
                index == length(arguments) && return (options, help, "--$(name) requires a value")
                index += 1
                value = arguments[index]
            end
            isempty(value) && return (options, help, "--$(name) requires a value")
            options[name] = value
        else
            return (options, help, "unknown argument: $(argument)")
        end
        index += 1
    end
    help && return (options, true, nothing)
    haskey(options, "input") || return (options, false, "missing --input RUN_DIRECTORY")
    get(options, "format", "png") in FORMATS ||
        return (options, false, "--format must be one of $(join(FORMATS, ", "))")
    if haskey(options, "times")
        all(s -> tryparse(Float64, strip(s)) !== nothing, split(options["times"], ',')) ||
            return (options, false, "--times must be a comma-separated list of e-folds")
    end
    if haskey(options, "max-curves")
        count = tryparse(Int, options["max-curves"])
        count !== nothing && count >= 1 || return (options, false, "--max-curves must be a positive integer")
    end
    return (options, false, nothing)
end

function main(arguments=ARGS)
    options, help, argument_error = parse_arguments(arguments)
    if help
        usage(stdout)
        return EXIT_SUCCESS
    elseif argument_error !== nothing
        println(stderr, "ERROR: ", argument_error)
        usage(stderr)
        return EXIT_CONFIG
    end
    run_directory = options["input"]
    isdir(run_directory) || begin
        println(stderr, "ERROR: run directory not found: ", abspath(run_directory))
        return EXIT_IO
    end
    output = get(options, "output", joinpath(run_directory, "figures"))
    times = haskey(options, "times") ? parse.(Float64, strip.(split(options["times"], ','))) : nothing
    max_curves = parse(Int, get(options, "max-curves", "6"))
    try
        written = plot_run(run_directory; output, format=get(options, "format", "png"), times, max_curves)
        println("Figures written: ", length(written))
        foreach(path -> println("  ", path), written)
        return EXIT_SUCCESS
    catch error
        error isa InterruptException && return EXIT_INTERRUPTED
        println(stderr, "ERROR: ", sprint(showerror, error))
        return EXIT_IO
    end
end

end # module MFILPlotCLI

if abspath(PROGRAM_FILE) == @__FILE__
    exit(MFILPlotCLI.main(ARGS))
end
