@testset "Bunch-Davies Hermitian and positive-frequency mode contract" begin
    lattice = Lattice3D(8, 2.0)
    initial = test_config(vacuum=true).initial
    a0, hconf0, B, effective_mass = 1.3, 0.17, 0.4, 0.2
    field, field_prime, removed = MFIL._bunch_davies_field(
        MersenneTwister(0x12345678), lattice, a0, hconf0, B,
        effective_mass, initial,
    )
    @test removed == 1 # the zero mode only when no explicit cutoffs are active
    coefficients = MFIL.FFTW.fft(field)
    prime_coefficients = MFIL.FFTW.fft(field_prime)
    linear = LinearIndices(coefficients)

    @test coefficients[1, 1, 1] ≈ 0.0 atol=2.0e-12
    @test prime_coefficients[1, 1, 1] ≈ 0.0 atol=2.0e-12
    for k in 1:lattice.N, j in 1:lattice.N, i in 1:lattice.N
        ci = MFIL._negative_fft_index(i, lattice.N)
        cj = MFIL._negative_fft_index(j, lattice.N)
        ck = MFIL._negative_fft_index(k, lattice.N)
        @test coefficients[ci, cj, ck] ≈ conj(coefficients[i, j, k])
        @test prime_coefficients[ci, cj, ck] ≈ conj(prime_coefficients[i, j, k])
        linear[i, j, k] <= linear[ci, cj, ck] || continue
        linear[i, j, k] == linear[ci, cj, ck] && continue
        omega = sqrt(
            MFIL._mode_k_lat(i, j, k, lattice)^2 +
            (a0 * effective_mass)^2,
        )
        @test prime_coefficients[i, j, k] ≈
              (-im * omega - hconf0) * coefficients[i, j, k] rtol=3.0e-12 atol=3.0e-12
    end
end

@testset "Bunch-Davies mode variance and independent fields" begin
    lattice = Lattice3D(8, 2.0)
    initial = test_config(vacuum=true).initial
    a0, hconf0, B, effective_mass = 1.3, 0.17, 0.4, 0.2
    nonself_index = CartesianIndex(2, 1, 1)
    self_index = CartesianIndex(5, 1, 1)
    sample_count = 512
    field_one = Vector{ComplexF64}(undef, sample_count)
    field_two = similar(field_one)
    self_field = Vector{Float64}(undef, sample_count)
    self_prime = similar(self_field)

    for sample in 1:sample_count
        root_seed = 100_000 + sample
        realization_one, prime_one, _ = MFIL._bunch_davies_field(
            MersenneTwister(MFIL.field_seed(root_seed, 1)),
            lattice, a0, hconf0, B, effective_mass, initial,
        )
        realization_two, _, _ = MFIL._bunch_davies_field(
            MersenneTwister(MFIL.field_seed(root_seed, 2)),
            lattice, a0, hconf0, B, effective_mass, initial,
        )
        transform_one = MFIL.FFTW.fft(realization_one)
        transform_prime = MFIL.FFTW.fft(prime_one)
        transform_two = MFIL.FFTW.fft(realization_two)
        field_one[sample] = transform_one[nonself_index]
        field_two[sample] = transform_two[nonself_index]
        self_field[sample] = real(transform_one[self_index])
        self_prime[sample] = real(transform_prime[self_index])
    end

    normalization = B * sqrt(lattice.volume / 2) / (a0 * lattice.dx^3)
    omega_nonself = sqrt(
        MFIL._mode_k_lat(Tuple(nonself_index)..., lattice)^2 +
        (a0 * effective_mass)^2,
    )
    omega_self = sqrt(
        MFIL._mode_k_lat(Tuple(self_index)..., lattice)^2 +
        (a0 * effective_mass)^2,
    )
    expected_nonself_variance = normalization^2 / omega_nonself
    expected_self_variance = normalization^2 / omega_self

    @test mean(abs2, field_one) ≈ expected_nonself_variance rtol=0.15
    @test mean(abs2, field_two) ≈ expected_nonself_variance rtol=0.15
    @test mean(abs2, self_field) ≈ expected_self_variance rtol=0.20
    @test mean(abs2, self_prime) ≈
          (omega_self^2 + hconf0^2) * expected_self_variance rtol=0.20
    normalized_cross = abs(mean(field_one .* conj.(field_two))) /
                       sqrt(mean(abs2, field_one) * mean(abs2, field_two))
    @test normalized_cross < 0.15

    # This is the physical-Riemann-sum form quoted in Initialization.jl.
    expected_physical_dft_variance =
        B^2 * lattice.volume / (2a0^2 * omega_nonself)
    @test lattice.dx^6 * mean(abs2, field_one) ≈
          expected_physical_dft_variance rtol=0.15
end
