using Test
using TOML
using Statistics
using LinearAlgebra
using Random
using MultifieldInflationLattice

const MFIL = MultifieldInflationLattice
const SAMPLE_CONFIG = normpath(joinpath(@__DIR__, "..", "configs", "two_field_quadratic.toml"))

function test_config(; vacuum=false, integrator="rk4", max_steps=8,
                     spectra=true, deltaN=false, output_root="runs")
    data = effective_config_dict(load_config(SAMPLE_CONFIG))
    data["lattice"]["size"] = 8
    data["initial"]["vacuum_fluctuations"] = vacuum
    data["evolution"]["integrator"] = integrator
    data["evolution"]["max_steps"] = max_steps
    data["evolution"]["max_wall_time"] = 0.0
    data["deltaN"]["enabled"] = deltaN
    data["deltaN"]["dN"] = 1.0e-3
    data["deltaN"]["separate_universe_min_ratio"] = 1.0e-12
    data["spectra"]["enabled"] = spectra
    data["output"]["root"] = output_root
    data["output"]["background_interval_efolds"] = 1.0e-4
    data["output"]["spectra_interval_efolds"] = 1.0e-4
    data["output"]["histogram_interval_efolds"] = 1.0e-4
    data["output"]["checkpoint_interval_efolds"] = 1.0e-4
    data["output"]["histogram_bins"] = 8
    data["output"]["save_2d_slices"] = false
    data["output"]["slice_index"] = 4
    return MFIL.config_from_dict(data)
end

include("unit/config_potential_lattice.jl")
include("unit/initialization_observables.jl")
include("unit/bd_statistics.jl")
include("unit/advanced_contracts.jl")
include("unit/deltan_resume.jl")
include("unit/integrators_checkpoint.jl")
include("reference/reference_regression.jl")
include("integration/short_run.jl")
include("integration/deltan_phase_resume.jl")
