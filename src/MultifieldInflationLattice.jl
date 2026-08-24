module MultifieldInflationLattice

using Dates
using FFTW
using JLD2
using LinearAlgebra
using Logging
using Printf
using Random
using SHA
using Statistics
using TOML
using UUIDs
using Base.Threads

include("potentials/PotentialInterface.jl")
include("integrators/IntegratorInterface.jl")
include("Types.jl")
include("Units.jl")
include("Config.jl")
include("Lattice.jl")
include("backends/CPUThreads.jl")
include("potentials/QuadraticPotential.jl")
include("potentials/Registry.jl")
include("Dynamics.jl")
include("Initialization.jl")
include("integrators/RK4.jl")
include("integrators/Leapfrog.jl")
include("observables/Background.jl")
include("observables/Energy.jl")
include("observables/SlowRoll.jl")
include("observables/Spectra.jl")
include("observables/Histograms.jl")
include("curvature/LinearCurvature.jl")
include("curvature/DeltaN.jl")
include("io/Metadata.jl")
include("io/CSVOutput.jl")
include("io/Checkpoint.jl")
include("Runner.jl")

export SimulationConfig,
       SimulationState,
       SimulationModel,
       Lattice3D,
       AbstractPotential,
       QuadraticPotential,
       AbstractIntegrator,
       RK4Integrator,
       LeapfrogIntegrator,
       ConfigError,
       NumericalError,
       load_config,
       validate_config,
       validate_config_file,
       effective_config_dict,
       build_lattice,
       build_potential,
       potential_value,
       potential_gradient!,
       potential_hessian!,
       characteristic_mass,
       lattice_k2,
       laplacian!,
       initialize_simulation,
       choose_dtau,
       background_summary,
       energy_summary,
       physical_energy_summary,
       slowroll_summary,
       scalar_spectrum,
       scalar_cross_spectrum,
       field_spectra,
       linear_curvature_map,
       linear_curvature_spectra,
       compute_deltaN,
       save_checkpoint,
       load_checkpoint,
       run_simulation,
       resume_simulation,
       analyze_run

const PACKAGE_VERSION = v"0.1.0"
const OUTPUT_SCHEMA_VERSION = "MFIL-CSV-1"
const CHECKPOINT_SCHEMA_VERSION = "MFIL-JLD2-1"
const INFLATION_EASY_REFERENCE = "d4f0cfdde0d148fa8ffaa14fe37d705cd79b2366"

end # module
