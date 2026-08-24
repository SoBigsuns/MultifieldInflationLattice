const FloatVector = Vector{Float64}
const FieldArray = Array{Float64,4}
const ScalarMap = Array{Float64,3}

"""Raised when an evolved state violates a required numerical invariant."""
struct NumericalError <: Exception
    message::String
end

Base.showerror(io::IO, error::NumericalError) = print(io, error.message)

"""Typed contents of the `[model]` TOML table."""
struct ModelConfig
    field_names::Vector{String}
    potential::String
    masses::FloatVector
    reduced_planck_mass::Float64
    rescale_B::Union{String,Float64}
end

"""Typed contents of the `[lattice]` TOML table."""
struct LatticeConfig
    size::Int
    box_size::Float64
    boundary::String
    laplacian::String
end

"""Typed contents of the `[initial]` TOML table."""
struct InitialConfig
    field_means::FloatVector
    cosmic_velocities::FloatVector
    scale_factor::Float64
    seed::UInt64
    vacuum_fluctuations::Bool
    include_effective_mass::Bool
    low_cutoff_index::Float64
    high_cutoff_index::Float64
end

"""Typed contents of the `[evolution]` TOML table."""
struct EvolutionConfig
    integrator::String
    timestep_factor::Float64
    end_epsilon_H::Float64
    max_efolds::Float64
    max_steps::Int
    max_wall_time::Float64
    friedmann_warning_tolerance::Float64
    friedmann_error_tolerance::Float64
end

"""Typed contents of the `[deltaN]` TOML table."""
struct DeltaNConfig
    enabled::Bool
    integrator::String
    reference_surface::String
    dN::Float64
    max_extra_efolds::Float64
    max_backward_efolds::Float64
    separate_universe_min_ratio::Float64
    strict_separate_universe::Bool
end

"""Typed contents of the `[spectra]` TOML table."""
struct SpectraConfig
    enabled::Bool
    use_lattice_momentum::Bool
    linear_zeta_min_speed2::Float64
end

"""Typed contents of the `[output]` TOML table."""
struct OutputConfig
    root::String
    run_name::String
    background_interval_efolds::Float64
    spectra_interval_efolds::Float64
    histogram_interval_efolds::Float64
    checkpoint_interval_efolds::Float64
    histogram_bins::Int
    save_2d_slices::Bool
    slice_axis::String
    slice_index::Int
    save_3d_snapshots::Bool
    overwrite::Bool
end

"""Typed contents of the `[parallel]` TOML table."""
struct ParallelConfig
    backend::String
    fft_threads::Int
    deterministic_reductions::Bool
end

"""Complete, validated effective run configuration."""
struct SimulationConfig
    model::ModelConfig
    lattice::LatticeConfig
    initial::InitialConfig
    evolution::EvolutionConfig
    deltaN::DeltaNConfig
    spectra::SpectraConfig
    output::OutputConfig
    parallel::ParallelConfig
end

"""
Description of the periodic cubic code-unit lattice.

FFT array indices are one based; `plus_index` and `minus_index` provide the
periodic nearest neighbours without a remainder operation in site loops.
"""
struct Lattice3D
    N::Int
    L::Float64
    dx::Float64
    inv_dx2::Float64
    volume::Float64
    kmin_lat::Float64
    kmax_lat::Float64
    plus_index::Vector{Int}
    minus_index::Vector{Int}
end

"""
Mutable synchronous simulation state in code units.

`field` and `field_prime` use `(x,y,z,field)` ordering.  `field_prime` is a
conformal-time derivative.  Method-specific staggering information belongs to
the integrator; `last_dtau` records the most recently completed interval for
restart and diagnostics.
"""
mutable struct SimulationState{T<:AbstractFloat}
    field::Array{T,4}
    field_prime::Array{T,4}
    a::T
    a_prime::T
    tau_code::T
    cosmic_time_code::T
    efolds::T
    step::Int
    last_dtau::T
end

function SimulationState(
    field::Array{T,4},
    field_prime::Array{T,4},
    a::T,
    a_prime::T,
    tau_code::T,
    cosmic_time_code::T,
    efolds::T,
    step::Integer,
) where {T<:AbstractFloat}
    return SimulationState(
        field,
        field_prime,
        a,
        a_prime,
        tau_code,
        cosmic_time_code,
        efolds,
        Int(step),
        zero(T),
    )
end

"""The concrete code-unit potential and immutable model metadata."""
struct SimulationModel{P<:AbstractPotential}
    potential::P
    B::Float64
    a0::Float64
    field_names::Vector{String}
    reduced_planck_mass::Float64
end

SimulationModel(potential::P, B::Real, a0::Real, field_names::Vector{String}) where {P<:AbstractPotential} =
    SimulationModel(potential, Float64(B), Float64(a0), copy(field_names), 1.0)

"""
Reusable arrays for lattice dynamics.  Per-thread vectors prevent allocation
when evaluating a general potential at every site.
"""
mutable struct SimulationCache{T<:AbstractFloat}
    laplacian::Array{T,4}
    dfield::Array{T,4}
    dprime::Array{T,4}
    work_field::Array{T,4}
    work_prime::Array{T,4}
    thread_fields::Matrix{T}
    thread_gradients::Matrix{T}
    deterministic_reductions::Bool
end

function SimulationCache(
    state::SimulationState{T};
    deterministic_reductions::Bool=false,
) where {T<:AbstractFloat}
    dims = size(state.field)
    dims == size(state.field_prime) || throw(DimensionMismatch(
        "field and field_prime must have identical dimensions",
    ))
    nf = dims[4]
    # `threadid()` ranges over all configured pools; interactive-pool IDs can
    # be greater than `nthreads()` on Julia 1.11 and later.
    nt = Base.Threads.maxthreadid()
    return SimulationCache(
        zeros(T, dims),
        zeros(T, dims),
        zeros(T, dims),
        similar(state.field),
        similar(state.field_prime),
        zeros(T, nf, nt),
        zeros(T, nf, nt),
        deterministic_reductions,
    )
end

"""Outcome of one accepted fixed time step."""
struct StepReport{T<:AbstractFloat}
    dtau::T
    synchronized::Bool
end

StepReport(dtau::Real; synchronized::Bool=false) =
    StepReport(Float64(dtau), synchronized)

"""Read-only-by-convention integer-time state returned for diagnostics."""
struct SynchronizedView{T<:AbstractFloat,A<:AbstractArray{T,4}}
    field::A
    field_prime::A
    a::T
    a_prime::T
    tau_code::T
    cosmic_time_code::T
    efolds::T
    step::Int
end

SynchronizedView(state::SimulationState{T}) where {T<:AbstractFloat} =
    SynchronizedView(
        state.field,
        state.field_prime,
        state.a,
        state.a_prime,
        state.tau_code,
        state.cosmic_time_code,
        state.efolds,
        state.step,
    )

"""Spatially averaged energy and pressure diagnostics in code units."""
struct EnergySummary
    kinetic_total::Float64
    gradient_total::Float64
    potential_total::Float64
    rho::Float64
    p::Float64
    w::Float64
    kinetic_by_field::FloatVector
    gradient_by_field::FloatVector
end

"""Hubble and potential slow-roll diagnostics."""
struct SlowRollSummary
    epsilon_H::Float64
    epsilon_V::Float64
    eta_parallel::Float64
    turn_rate::Float64
    turn_rate_over_H::Float64
    epsilonH_by_field::FloatVector
    etaV_diag::FloatVector
    etaV_eigenvalues::FloatVector
end

"""Result of the nonlinear separate-universe `δN` calculation."""
struct DeltaNResult{S}
    local_efolds::ScalarMap
    zeta::ScalarMap
    rho_reference::Float64
    su_ratio::Float64
    su_valid::Bool
    n_forward::Int
    n_backward::Int
    failed_indices::Vector{CartesianIndex{3}}
    mean::Float64
    variance::Float64
    skewness::Float64
    excess_kurtosis::Float64
    spectrum::S
end
