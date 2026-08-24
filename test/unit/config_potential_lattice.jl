@testset "configuration" begin
    config = load_config(SAMPLE_CONFIG)
    @test config.model.field_names == ["phi", "psi"]
    @test config.model.masses == [9.0e-6, 1.0e-6]
    @test config.initial.field_means == [13.0, 13.0]
    @test config.initial.cosmic_velocities == [1.0e-10, 0.0]
    @test config.lattice.size == 32
    @test validate_config(config) === config

    unknown = effective_config_dict(config)
    unknown["lattice"]["mystery"] = 1
    @test_throws ConfigError MFIL.config_from_dict(unknown)

    mismatched = effective_config_dict(config)
    mismatched["model"]["masses"] = [1.0]
    @test_throws ConfigError MFIL.config_from_dict(mismatched)
end

@testset "quadratic potential" begin
    potential = QuadraticPotential([2.0, 3.0])
    fields = [1.5, -2.0]
    gradient = zeros(2)
    hessian = zeros(2, 2)
    @test potential_value(potential, fields) == 0.5 * (4 * 1.5^2 + 9 * 2^2)
    @test potential_gradient!(gradient, potential, fields) === nothing
    @test gradient == [6.0, -18.0]
    @test potential_hessian!(hessian, potential, fields) === nothing
    @test hessian == Diagonal([4.0, 9.0])
    @test characteristic_mass(potential, fields) == 3.0

    epsilon = 1.0e-6
    for index in eachindex(fields)
        plus = copy(fields); plus[index] += epsilon
        minus = copy(fields); minus[index] -= epsilon
        finite_difference = (potential_value(potential, plus) - potential_value(potential, minus)) / (2epsilon)
        @test isapprox(finite_difference, gradient[index]; rtol=1.0e-9)
    end
end

@testset "periodic seven-point Laplacian" begin
    lattice = Lattice3D(8, 2.0)
    constant_field = fill(3.25, 8, 8, 8)
    output = similar(constant_field)
    laplacian!(output, constant_field, lattice)
    @test output == zeros(size(output))

    mode = (1, 2, 0)
    plane = Array{Float64}(undef, 8, 8, 8)
    for k in 1:8, j in 1:8, i in 1:8
        phase = 2pi * (mode[1] * (i - 1) + mode[2] * (j - 1) + mode[3] * (k - 1)) / 8
        plane[i, j, k] = cos(phase)
    end
    laplacian!(output, plane, lattice)
    expected = -lattice_k2(lattice, mode) .* plane
    @test output ≈ expected atol=2.0e-13 rtol=2.0e-13
    @test output[1, 1, 1] == output[9 - 8, 1, 1]
end

