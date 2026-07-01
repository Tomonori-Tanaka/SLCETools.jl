# P1 of the mean-field sampler (docs/specs/mfa-sampling.md): the single global,
# isotropic `MFASampler`. Validates the Langevin self-consistency `m = L(3m/τ)`, the
# τ ↔ m inversion, the boundary limits (ordered / near-uniform), and that the drawn
# configurations carry the self-consistent mean resultant length `m(τ)` — the same
# physics as Magesty's `MfaSampling`.

using Test
using SCEFitting
using SCETools
using LinearAlgebra
using Random
using Statistics: mean

const MR = SCETools

_lang(κ) = coth(κ) - 1 / κ

# A reference with `n` atoms all pointing along +z (so the mean of the drawn z-components
# directly estimates ⟨cosθ⟩ = m).
_zref(n) = repeat(Float64[0, 0, 1], 1, n)

@testset "MFA sampler (single global, P1)" begin
    @testset "self-consistency m = L(3m/τ) and its inverse" begin
        # boundary limits
        @test MR.thermal_averaged_m(1.0e-9) == 1.0
        @test MR.thermal_averaged_m(1.5) == 0.0
        # interior: m solves the self-consistency to machine precision
        for τ in (0.2, 0.4, 0.6, 0.8, 0.95)
            m = MR.thermal_averaged_m(τ)
            @test 0.0 < m < 1.0
            @test m ≈ _lang(3m / τ) atol = 1e-9
        end
        # monotone decreasing in τ
        ms = [MR.thermal_averaged_m(τ) for τ in 0.1:0.1:0.95]
        @test all(diff(ms) .< 0)
        # τ ↔ m round-trip
        for m0 in (0.2, 0.5, 0.85)
            τ = MR.tau_from_magnetization(m0)
            @test MR.thermal_averaged_m(τ) ≈ m0 atol = 1e-6
        end
        @test MR.tau_from_magnetization(0.0) == 1.0
        @test MR.tau_from_magnetization(1.0) == 0.0
    end

    @testset "constructor normalizes and validates the reference" begin
        s = MFASampler([2.0 0.0; 0.0 -3.0; 0.0 0.0])   # 3×2, non-unit columns
        @test s.reference[:, 1] ≈ [1.0, 0.0, 0.0]
        @test s.reference[:, 2] ≈ [0.0, -1.0, 0.0]
        @test_throws ArgumentError MFASampler(zeros(2, 3))          # wrong leading dim
        @test_throws ArgumentError MFASampler(reshape(Float64[], 3, 0))  # no atoms
        @test_throws ArgumentError MFASampler([0.0 1.0; 0.0 0.0; 0.0 0.0])  # zero column
        @test mfa_temperature_scale(s) == 1.0
    end

    @testset "drawn configs carry the self-consistent magnetization m(τ)" begin
        s = MFASampler(_zref(400))
        rng = MersenneTwister(2024)
        for τ in (0.3, 0.6, 0.85)
            samp = sample(s, 25; tau = τ, rng = rng)
            @test samp isa MFASample
            @test length(samp) == 25
            @test all(c -> size(c) == (3, 400), samp.configs)
            # every drawn column is a unit vector
            @test all(c -> all(a -> abs(norm(@view c[:, a]) - 1) < 1e-10, 1:400),
                      samp.configs)
            # mean of all z-components ≈ m(τ) = L(κ)
            zbar = mean(mean(@view c[3, :]) for c in samp.configs)
            @test zbar ≈ MR.thermal_averaged_m(τ) atol = 2e-2
            @test all(==(τ), samp.tau)
            @test all(v -> all(≈(MR.thermal_averaged_m(τ)), v), samp.m)
        end
    end

    @testset "control by magnetization matches control by τ" begin
        s = MFASampler(_zref(400))
        m0 = 0.7
        τ = MR.tau_from_magnetization(m0)
        zbar = mean(mean(@view c[3, :]) for c in sample(s, 40; m = m0,
                    rng = MersenneTwister(1)).configs)
        @test zbar ≈ m0 atol = 2e-2
        # τ label recovered from the m input
        @test sample(s, 1; m = m0, rng = MersenneTwister(1)).tau[1] ≈ τ
    end

    @testset "boundary limits: ordered returns the reference, hot is near-uniform" begin
        s = MFASampler(_zref(300))
        ordered = sample(s, 3; tau = 1.0e-9, rng = MersenneTwister(0))
        @test all(c -> c ≈ s.reference, ordered.configs)   # exactly the reference
        @test all(v -> all(==(1.0), v), ordered.m)
        hot = sample(s, 20; tau = 1.5, rng = MersenneTwister(0))
        @test abs(mean(mean(@view c[3, :]) for c in hot.configs)) < 5e-2  # ≈ isotropic
        @test all(v -> all(==(0.0), v), hot.m)
    end

    @testset "fixed / uniform / randomize keywords" begin
        s = MFASampler(_zref(50))
        # fixed atoms stay exactly at the reference
        samp = sample(s, 5; tau = 0.5, rng = MersenneTwister(3), fixed = [1, 7])
        @test all(c -> c[:, 1] ≈ [0, 0, 1.0] && c[:, 7] ≈ [0, 0, 1.0], samp.configs)
        # uniform atoms are isotropic: their mean z over many draws ≈ 0
        big = MFASampler(_zref(1))
        us = sample(big, 4000; tau = 0.2, rng = MersenneTwister(4), uniform = [1])
        @test abs(mean(c[3, 1] for c in us.configs)) < 5e-2
        # randomize keeps unit norm and changes the realized frame
        r1 = sample(s, 2; tau = 0.5, rng = MersenneTwister(5), randomize = true)
        @test all(c -> all(a -> abs(norm(@view c[:, a]) - 1) < 1e-10, 1:50), r1.configs)
        @test_throws ArgumentError sample(s, 1; tau = 0.5, fixed = [99])
    end

    @testset "sweep form and control-variable guards" begin
        s = MFASampler(_zref(10))
        sw = sample(s; tau = range(0.2, 0.9; length = 5), nsamples = 3,
                    rng = MersenneTwister(6))
        @test length(sw) == 15
        @test sw.tau[1:3] == fill(0.2, 3)            # value-outer / sample-inner ordering
        @test issorted(sw.tau)
        @test_throws ArgumentError sample(s, 5)                       # neither tau nor m
        @test_throws ArgumentError sample(s, 5; tau = 0.5, m = 0.5)   # both
        @test_throws ArgumentError sample(s, 5; tau = [0.3, 0.6])     # collection + n
    end

    @testset "draws are reproducible under a fixed seed" begin
        s = MFASampler(_zref(20))
        a = sample(s, 10; tau = 0.5, rng = MersenneTwister(42))
        b = sample(s, 10; tau = 0.5, rng = MersenneTwister(42))
        @test a.configs == b.configs
    end

    @testset "MFASample: array-like interface and per-config m (no aliasing)" begin
        s = MFASampler(_zref(4))
        sw = sample(s; tau = [0.3, 0.6], nsamples = 3, rng = MersenneTwister(11))
        @test length(sw) == 6
        @test sw[end] === sw.configs[end]           # lastindex defined ⇒ `end` works
        @test sw[begin] === sw.configs[1]
        # each config carries its own m vector: mutating one must not touch a co-τ sibling
        @test sw.m[1] !== sw.m[2]
        sw.m[1][1] = -99.0
        @test sw.m[2][1] != -99.0
        # the inner constructor rejects non-parallel configs/tau/m
        @test_throws DimensionMismatch SCETools.MFASample([zeros(3, 1)], [0.5, 0.6], [[1.0]])
    end

    @testset "sweep by a magnetization collection mirrors the τ collection" begin
        s = MFASampler(_zref(10))
        ms = [0.3, 0.7]
        sw = sample(s; m = ms, nsamples = 2, rng = MersenneTwister(13))
        @test length(sw) == 4
        @test sw.tau[1:2] == fill(MR.tau_from_magnetization(0.3), 2)
        @test sw.tau[3:4] == fill(MR.tau_from_magnetization(0.7), 2)
        @test all(v -> all(≈(0.3; atol = 1e-6), v), sw.m[1:2])
    end

    @testset "randomize really rotates the realized frame" begin
        # At τ = 0.2 the un-rotated draw is a tight cone about +ẑ (mean z ≈ 0.98); one
        # uniform global rotation per configuration moves that cone off +ẑ for a generic
        # seed, while every column stays unit.
        s = MFASampler(_zref(50))
        r = sample(s, 3; tau = 0.2, rng = MersenneTwister(5), randomize = true)
        zbars = [mean(@view c[3, :]) for c in r.configs]
        @test any(z -> abs(z) < 0.9, zbars)              # some frame is off +ẑ
        @test !all(≈(zbars[1]; atol = 1e-6), zbars)      # one independent rotation per config
        @test all(c -> all(a -> abs(norm(@view c[:, a]) - 1) < 1e-10, 1:50), r.configs)
    end

    @testset "negative reduced temperatures are rejected loudly" begin
        s = MFASampler(_zref(3))
        @test_throws ArgumentError sample(s, 1; tau = -0.2)
        @test_throws ArgumentError sample(s; tau = [0.5, -0.1], nsamples = 1)
        @test_throws ArgumentError mfa_sublattice_m(s, -0.5)
        @test_throws ArgumentError MR.thermal_averaged_m(-1.0)
    end
end
