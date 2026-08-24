"""
    histogram_rows(values, quantity, output_id, nbins; limits=nothing)

Construct CSV-ready histogram rows.  Only finite samples participate.  Density
is normalised so that `sum(density * bin_width) == 1` up to roundoff.
"""
function histogram_rows(
    values,
    quantity,
    output_id,
    nbins::Integer;
    limits=nothing,
)
    nbins > 0 || throw(ArgumentError("histogram bin count must be positive"))
    finite_values = Float64[x for x in values if isfinite(x)]
    if isempty(finite_values)
        @warn "histogram omitted because it has no finite samples" quantity=quantity output_id=output_id
        return NamedTuple[]
    end

    if limits === nothing
        lower, upper = extrema(finite_values)
    else
        length(limits) == 2 || throw(ArgumentError("limits must contain lower and upper bounds"))
        lower, upper = Float64(limits[1]), Float64(limits[2])
    end
    isfinite(lower) && isfinite(upper) && lower <= upper ||
        throw(ArgumentError("histogram limits must be finite and ordered"))

    if lower == upper
        padding = max(abs(lower), 1.0) * sqrt(eps(Float64))
        lower -= padding
        upper += padding
    end
    width = (upper - lower) / nbins
    counts = zeros(Int, nbins)
    included = 0
    @inbounds for value in finite_values
        if lower <= value <= upper
            slot = value == upper ? nbins : floor(Int, (value - lower) / width) + 1
            counts[clamp(slot, 1, nbins)] += 1
            included += 1
        end
    end
    included > 0 || return NamedTuple[]

    rows = NamedTuple[]
    @inbounds for slot in 1:nbins
        bin_left = lower + (slot - 1) * width
        bin_right = slot == nbins ? upper : lower + slot * width
        push!(rows, (
            quantity=String(quantity),
            output_id=output_id,
            bin_left=bin_left,
            bin_right=bin_right,
            bin_center=0.5 * (bin_left + bin_right),
            count=counts[slot],
            density=counts[slot] / (included * width),
        ))
    end
    return rows
end

"""Return histogram rows for every field, keyed by the configured field name."""
function field_histogram_rows(state, model, output_id, nbins::Integer)
    nf = _obs_nfields(state)
    length(model.field_names) == nf || throw(DimensionMismatch("field name count differs from state"))
    rows = NamedTuple[]
    @inbounds for q in 1:nf
        append!(rows, histogram_rows(
            @view(state.field[:, :, :, q]),
            model.field_names[q],
            output_id,
            nbins,
        ))
    end
    return rows
end

function _normalise_slice_axis(axis)
    symbol = Symbol(lowercase(String(axis)))
    symbol in (:x, :y, :z) || throw(ArgumentError("slice axis must be x, y, or z"))
    return symbol
end

"""
    two_dimensional_slice(map3; axis=:z, index=cld(size(map3,3),2))

Copy a coordinate-aligned 2-D slice.  `index` is a Julia one-based array
index, matching the indices written by `slice_rows`.
"""
function two_dimensional_slice(map3::AbstractArray{T,3}; axis=:z, index=nothing) where {T}
    selected_axis = _normalise_slice_axis(axis)
    dimension = selected_axis === :x ? size(map3, 1) :
                selected_axis === :y ? size(map3, 2) : size(map3, 3)
    selected_index = index === nothing ? cld(dimension, 2) : Int(index)
    1 <= selected_index <= dimension || throw(BoundsError(map3, selected_index))

    if selected_axis === :x
        return copy(@view map3[selected_index, :, :])
    elseif selected_axis === :y
        return copy(@view map3[:, selected_index, :])
    else
        return copy(@view map3[:, :, selected_index])
    end
end

"""
    slice_rows(map3, lattice; axis=:z, index=nothing)

Create `i,j,x,y,value` rows for one 2-D section.  The coordinate column names
are schema-stable; for an x slice they represent `(y,z)`, for a y slice
`(x,z)`, and for a z slice `(x,y)`.
"""
function slice_rows(
    map3::AbstractArray{<:Real,3},
    lattice;
    axis=:z,
    index=nothing,
    coordinate_scale::Real=1.0,
)
    size(map3) == (lattice.N, lattice.N, lattice.N) ||
        throw(DimensionMismatch("map and lattice sizes differ"))
    section = two_dimensional_slice(map3; axis=axis, index=index)
    rows = NamedTuple[]
    spacing = lattice.dx * Float64(coordinate_scale)
    @inbounds for j in axes(section, 2), i in axes(section, 1)
        push!(rows, (
            i=i,
            j=j,
            x=(i - 1) * spacing,
            y=(j - 1) * spacing,
            value=Float64(section[i, j]),
        ))
    end
    return rows
end

"""Return one set of slice rows per field, keyed by field name."""
function field_slice_rows(
    state,
    model,
    lattice;
    axis=:z,
    index=nothing,
    coordinate_scale::Real=inv(model.B),
)
    nf = _obs_nfields(state)
    result = Dict{String,Vector{NamedTuple}}()
    @inbounds for q in 1:nf
        result[String(model.field_names[q])] = slice_rows(
            @view(state.field[:, :, :, q]), lattice;
            axis=axis, index=index, coordinate_scale=coordinate_scale,
        )
    end
    return result
end
