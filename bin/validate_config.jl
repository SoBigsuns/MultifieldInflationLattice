#!/usr/bin/env julia

module MFILValidateConfigCLI

using MultifieldInflationLattice

const EXIT_SUCCESS = 0
const EXIT_CONFIG = 2
const EXIT_IO = 3
const EXIT_NUMERICAL = 4
const EXIT_INTERRUPTED = 130

function usage(io::IO=stdout)
    println(io, "Usage:")
    println(io, "  julia --project=. bin/validate_config.jl --config PATH")
    println(io)
    println(io, "Validate all TOML keys, types, dimensions, numerical limits, and derived safety checks.")
end

function parse_arguments(arguments)
    isempty(arguments) && return (nothing, false, "missing --config PATH")
    config = nothing
    help = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            help = true
        elseif argument == "--config"
            index == length(arguments) && return (nothing, help, "--config requires a path")
            config === nothing || return (nothing, help, "--config may be supplied only once")
            index += 1
            config = arguments[index]
        elseif startswith(argument, "--config=")
            config === nothing || return (nothing, help, "--config may be supplied only once")
            config = split(argument, '='; limit=2)[2]
            isempty(config) && return (nothing, help, "--config requires a path")
        else
            return (nothing, help, "unknown argument: $(argument)")
        end
        index += 1
    end
    help && return (config, true, nothing)
    config === nothing && return (nothing, false, "missing --config PATH")
    return (config, false, nothing)
end

function error_code(error; context=:validate)
    error isa InterruptException && return EXIT_INTERRUPTED
    name = lowercase(string(nameof(typeof(error))))
    occursin("config", name) && return EXIT_CONFIG
    error isa SystemError && return EXIT_IO
    error isa Base.IOError && return EXIT_IO
    error isa EOFError && return EXIT_IO
    error isa DomainError && return EXIT_NUMERICAL
    error isa OverflowError && return EXIT_NUMERICAL
    occursin("checkpoint", name) && return EXIT_IO
    occursin("output", name) && return EXIT_IO
    return context == :validate ? EXIT_CONFIG : EXIT_NUMERICAL
end

function report_error(error)
    println(stderr, "ERROR: ", sprint(showerror, error))
end

function main(arguments=ARGS)
    config_path, help, argument_error = parse_arguments(arguments)
    if help
        usage(stdout)
        return EXIT_SUCCESS
    elseif argument_error !== nothing
        println(stderr, "ERROR: ", argument_error)
        usage(stderr)
        return EXIT_CONFIG
    end
    isfile(config_path) || begin
        println(stderr, "ERROR: configuration file not found: ", abspath(config_path))
        return EXIT_IO
    end
    try
        result = validate_config_file(config_path)
        result === false && begin
            println(stderr, "ERROR: configuration validation failed")
            return EXIT_CONFIG
        end
        println("Configuration is valid: ", abspath(config_path))
        return EXIT_SUCCESS
    catch error
        report_error(error)
        return error_code(error; context=:validate)
    end
end

end # module MFILValidateConfigCLI

if abspath(PROGRAM_FILE) == @__FILE__
    exit(MFILValidateConfigCLI.main(ARGS))
end
