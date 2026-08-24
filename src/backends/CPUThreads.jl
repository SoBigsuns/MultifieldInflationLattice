"""CPU-thread backend settings used by local kernels and reductions."""
struct CPUThreadsBackend
    deterministic_reductions::Bool
end

CPUThreadsBackend(config::ParallelConfig) = begin
    config.backend == "cpu_threads" || throw(ArgumentError(
        "CPUThreadsBackend requires parallel.backend = \"cpu_threads\"",
    ))
    CPUThreadsBackend(config.deterministic_reductions)
end

CPUThreadsBackend(; deterministic_reductions::Bool=false) =
    CPUThreadsBackend(deterministic_reductions)

"""Number of Julia worker threads visible to the CPU backend."""
cpu_thread_count() = Base.Threads.nthreads()

"""Fixed contiguous chunks used for reproducible threaded site traversal."""
function static_thread_ranges(
    count::Integer,
    chunks::Integer=Base.Threads.nthreads(),
)
    count >= 0 || throw(ArgumentError("item count must be non-negative"))
    chunks > 0 || throw(ArgumentError("chunk count must be positive"))
    n = Int(count)
    nchunks = min(Int(chunks), max(n, 1))
    quotient, remainder = divrem(n, nchunks)
    ranges = Vector{UnitRange{Int}}(undef, nchunks)
    first_index = 1
    @inbounds for chunk in 1:nchunks
        width = quotient + (chunk <= remainder ? 1 : 0)
        last_index = first_index + width - 1
        ranges[chunk] = first_index:last_index
        first_index = last_index + 1
    end
    return ranges
end

@inline function _kahan_sum(function_value, indices)
    total = 0.0
    compensation = 0.0
    @inbounds for index in indices
        value = Float64(function_value(index))
        isfinite(value) || return value
        corrected = value - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    end
    return total
end

"""
    threaded_sum(f, count; deterministic=false)

Sum `f(i)` for `i in 1:count`.  Deterministic mode uses a serial compensated
sum, giving a fixed order independent of the active thread count.  Normal mode
uses one compensated partial sum per statically assigned chunk, followed by a
fixed-order merge.
"""
function threaded_sum(
    function_value,
    count::Integer;
    deterministic::Bool=false,
)
    count >= 0 || throw(ArgumentError("item count must be non-negative"))
    n = Int(count)
    deterministic && return _kahan_sum(function_value, 1:n)
    n == 0 && return 0.0
    ranges = static_thread_ranges(n)
    partials = zeros(Float64, length(ranges))
    Base.Threads.@threads :static for chunk in eachindex(ranges)
        partials[chunk] = _kahan_sum(function_value, ranges[chunk])
    end
    return _kahan_sum(index -> partials[index], eachindex(partials))
end

threaded_sum(array::AbstractArray{<:Real}; deterministic::Bool=false) =
    threaded_sum(index -> array[index], length(array); deterministic)

function threaded_mean(array::AbstractArray{<:Real}; deterministic::Bool=false)
    isempty(array) && throw(ArgumentError("mean of an empty array is undefined"))
    return threaded_sum(array; deterministic) / length(array)
end

"""Apply `f(index)` to `1:count` using static CPU-thread scheduling."""
function threaded_foreach(function_value, count::Integer)
    count >= 0 || throw(ArgumentError("item count must be non-negative"))
    Base.Threads.@threads :static for index in 1:Int(count)
        function_value(index)
    end
    return nothing
end
