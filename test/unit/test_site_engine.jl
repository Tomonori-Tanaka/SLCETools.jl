# P0 of the mean-field sampler (docs/specs/mfa-sampling.md): the single-site engine.
# Validates the single-site potential, the vMF closed form, the general Metropolis draw,
# and the spherical quadrature against analytic mean-field results — chiefly that all
# three reproduce the Langevin curve L(κ) for an l=1 field, and that Metropolis matches
# the deterministic quadrature on a non-vMF (l=2 Bingham) field.

using Test
using MagestyRebuild
using SCETools
using LinearAlgebra
using Random
using StaticArrays

const MR = SCETools

_langevin(κ) = coth(κ) - 1 / κ                 # ⟨cosθ⟩ for vMF concentration κ
_ez(v) = v[3]

# An l=1 field c with P(e) ∝ exp(κ·e_z): V = -κ e_z, and Z_10(e) = N·e_z, so c[1,0] = -κ/N.
function _l1_field_along_z(κ, lmax)
    c = zeros((lmax + 1)^2)
    N10 = MR.Harmonics.Zlm(1, 0, SVector{3,Float64}(0, 0, 1))
    c[MR.Harmonics.lm_index(1, 0)] = -κ / N10
    return c, N10
end

@testset "site engine (mean-field P0)" begin
    @testset "Gauss–Legendre × azimuth quadrature is well-formed" begin
        q = MR.sphere_quadrature(3)
        @test sum(q.weights) ≈ 4π                       # total solid angle
        @test all(v -> isapprox(norm(v), 1.0; atol = 1e-12), q.dirs)   # unit nodes
        # Uniform distribution (c = 0): every l ≥ 1 multipole vanishes by symmetry.
        avg = MR.multipole_average(q, zeros(16), 3)
        @test all(i -> abs(avg[i]) < 1e-10, 2:16)
        @test avg[1] ≈ MR.Harmonics.Zlm(0, 0, SVector{3,Float64}(0, 0, 1))  # ⟨Z_00⟩ = Z_00
    end

    @testset "l=1 field: vMF, Metropolis, quadrature all give Langevin L(κ)" begin
        for κ in (0.5, 2.0, 5.0)
            c, N10 = _l1_field_along_z(κ, 1)
            target = _langevin(κ)

            # the l=1 field vector is -κ ẑ ⇒ vMF(ẑ, κ)
            @test MR._l1_field(c) ≈ SVector{3,Float64}(0, 0, -κ) atol = 1e-10

            # deterministic quadrature: ⟨Z_10⟩ / N = ⟨e_z⟩ = L(κ); transverse parts vanish
            q = MR.sphere_quadrature(1)
            avg = MR.multipole_average(q, c, 1)
            @test avg[MR.Harmonics.lm_index(1, 0)] / N10 ≈ target atol = 1e-4
            @test abs(avg[MR.Harmonics.lm_index(1, 1)]) < 1e-6
            @test abs(avg[MR.Harmonics.lm_index(1, -1)]) < 1e-6

            # vMF closed-form draw and general Metropolis draw both recover ⟨e_z⟩ = L(κ)
            rng = MersenneTwister(1234)
            nv = 80_000
            mv = sum(_ez(MR.sample_vmf_field(rng, c)) for _ = 1:nv) / nv
            sm = MR.sample_site_metropolis(rng, c, 25_000; step = 0.7, nburn = 500, thin = 5)
            mm = sum(_ez, sm) / length(sm)
            @test mv ≈ target atol = 1e-2
            @test mm ≈ target atol = 2e-2
        end
    end

    @testset "quadrature auto-sizes to a sharply peaked (large-field) distribution" begin
        κ = 20.0                                        # a sharp vMF peak (low τ)
        c, N10 = _l1_field_along_z(κ, 1)
        target = _langevin(κ)
        lm10 = MR.Harmonics.lm_index(1, 0)
        # the harmonic-order default (ntheta = 8) under-resolves the peak ...
        coarse = MR.multipole_average(MR.sphere_quadrature(1), c, 1)[lm10] / N10
        @test !isapprox(coarse, target; atol = 1e-3)
        # ... but sizing the grid from the field strength recovers L(κ)
        fine = MR.multipole_average(c, 1)[lm10] / N10    # two-arg form auto-sizes
        @test fine ≈ target atol = 1e-4
        # malformed (non-square) fields are rejected
        @test_throws ArgumentError MR.multipole_average([1.0, 2.0, 3.0], 1)
    end

    @testset "general engine on a non-vMF (l=2 Bingham) field: Metropolis ↔ quadrature" begin
        # An easy-axis quadrupolar field: V = a·Z_20, P ∝ exp(-a Z_20). With a < 0 the
        # distribution concentrates toward the ±z axis (a girdle for a > 0) — a genuinely
        # non-vMF shape that the closed forms cannot represent, so it exercises the general
        # Metropolis path and cross-checks it against the deterministic quadrature.
        lmax = 2
        c = zeros((lmax + 1)^2)
        c[MR.Harmonics.lm_index(2, 0)] = -3.0          # easy-axis (concentrates along ±z)

        avg = MR.multipole_average(c, lmax)            # field-aware auto-sized grid
        z20 = avg[MR.Harmonics.lm_index(2, 0)]
        @test z20 > 0.05                                # genuine quadrupolar order (non-uniform)
        @test abs(avg[MR.Harmonics.lm_index(1, 0)]) < 1e-6   # no net dipole (±z symmetric)

        # Metropolis estimate of ⟨Z_20⟩ matches the quadrature, and ⟨Z_10⟩ ≈ 0 (bimodal ±z).
        rng = MersenneTwister(7)
        sm = MR.sample_site_metropolis(rng, c, 40_000; step = 0.6, nburn = 1000, thin = 5)
        z20_mc = sum(e -> MR.Harmonics.Zlm(2, 0, e), sm) / length(sm)
        z10_mc = sum(e -> MR.Harmonics.Zlm(1, 0, e), sm) / length(sm)
        @test z20_mc ≈ z20 atol = 2e-2
        @test abs(z10_mc) < 2e-2
    end

    @testset "draws are reproducible under a fixed seed" begin
        c, _ = _l1_field_along_z(2.0, 1)
        a = MR.sample_site_metropolis(MersenneTwister(99), c, 50; nburn = 50)
        b = MR.sample_site_metropolis(MersenneTwister(99), c, 50; nburn = 50)
        @test a == b
        va = MR.sample_vmf_field(MersenneTwister(5), c)
        vb = MR.sample_vmf_field(MersenneTwister(5), c)
        @test va == vb
    end
end
