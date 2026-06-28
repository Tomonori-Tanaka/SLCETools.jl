# P3 of the mean-field sampler (docs/specs/mfa-sampling.md): tensorial exchange (DMI +
# anisotropic) and single-ion anisotropy, drawn from the Bingham single-site potential via
# the general Metropolis engine. Validates the single-ion Bingham shapes (easy-axis cone
# sharpening / above-T_MF persistence, easy-plane girdle), that the Metropolis draw
# reproduces the quadrature self-consistency, and that DMI tilts a collinear reference.

using Test
using SCEFitting
using SCETools
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean

const MR = SCETools

_Z(l, m, v) = MR.Harmonics.Zlm(l, m, SVector{3,Float64}(v))
_mean_Z(l, m, configs, a) = mean(_Z(l, m, c[:, a]) for c in configs)

@testset "tensorial / single-ion sampler (P3)" begin
    @testset "single-ion makes the sampler tensorial; isotropic stays vMF" begin
        iso = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        @test !MR._needs_metropolis(iso)                        # closed-form vMF path
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)    # easy-axis along z
        ten = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                         reference = Float64[0 0; 0 0; 1 1])
        @test MR._needs_metropolis(ten)                         # Metropolis path
        @test occursin("tensorial", sprint(show, ten))
    end

    @testset "easy-axis single-ion sharpens the cone and persists above T_MF" begin
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)    # e'Ae = 1 − 3e_z² (min at ±z)
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[0 0; 0 0; 1 1])
        iso = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        for τ in (0.5, 0.9)
            @test mfa_sublattice_m(s, τ)[1] > mfa_sublattice_m(iso, τ)[1] + 1e-2
        end
        # above the exchange T_MF the isotropic order is gone but the single-ion persists
        @test mfa_sublattice_m(iso, 1.2)[1] == 0.0
        @test mfa_sublattice_m(s, 1.2)[1] > 0.3
        # the sampled distribution concentrates along ±z ⇒ ⟨Z_20⟩ > 0
        samp = sample(s, 3000; tau = 0.8, rng = MersenneTwister(1))
        @test _mean_Z(2, 0, samp.configs, 1) > 0.1
    end

    @testset "easy-plane single-ion makes a girdle (⟨Z_20⟩ < 0)" begin
        # hard axis z (e'Ae = k e_z², min in the xy-plane); reference in-plane along +x.
        A = SMatrix{3,3,Float64}(0, 0, 0, 0, 0, 0, 0, 0, 3)
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[1 1; 0 0; 0 0])
        samp = sample(s, 4000; tau = 0.7, rng = MersenneTwister(2))
        @test _mean_Z(2, 0, samp.configs, 1) < -0.05            # flattened out of z (lab frame)
        @test mean(c[1, 1] for c in samp.configs) > 0.3        # still ordered along +x
    end

    @testset "Metropolis draw reproduces the quadrature self-consistency" begin
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[0 0; 0 0; 1 1])
        τ = 1.1
        # quadrature reference: the converged single-site coefficients and their ⟨Z_lm⟩
        cs, mq = MR._tensor_state(s.exch, MR._ehat(s), s.rho, τ)
        avg = MR.multipole_average(cs[1], 2)
        samp = sample(s, 6000; tau = τ, rng = MersenneTwister(7))
        @test _mean_Z(2, 0, samp.configs, 1) ≈ avg[MR.Harmonics.lm_index(2, 0)] atol = 3e-2
        @test _mean_Z(1, 0, samp.configs, 1) ≈ avg[MR.Harmonics.lm_index(1, 0)] atol = 3e-2
        # the reported magnetization equals ⟨e·ê⟩ from the same draw
        @test mean(c[3, 1] for c in samp.configs) ≈ mq[1] atol = 3e-2
    end

    @testset "DMI tilts a collinear reference (and flags it non-stationary)" begin
        # ferro + a DM vector along x: on a collinear +z reference the molecular field gains
        # a transverse (−y) component, so the reference is not stationary and the drawn mean
        # tilts toward y. bilinear[1,2] = J·I + [D]× with D = (d,0,0).
        J = -1.0; d = 0.5
        Dx = @SMatrix [0.0 0 0; 0 0 -d; 0 d 0]
        M = J * SMatrix{3,3,Float64}(I) + Dx
        bil = Matrix{SMatrix{3,3,Float64,9}}(undef, 2, 2)
        bil[1, 1] = zero(SMatrix{3,3,Float64,9})
        bil[2, 2] = zero(SMatrix{3,3,Float64,9})
        bil[1, 2] = M
        bil[2, 1] = SMatrix{3,3,Float64}(transpose(M))
        exch = ExchangeModel(bil)
        @test !exch.isotropic                                   # DMI ⇒ tensorial
        ref = Float64[0 0; 0 0; 1 1]
        s = @test_logs (:warn,) match_mode = :any MFASampler(exch; reference = ref)
        samp = sample(s, 4000; tau = 0.4, rng = MersenneTwister(5))
        # the cone tilts off +z: a non-negligible mean transverse (y) component
        @test abs(mean(c[2, 1] for c in samp.configs)) > 0.05
        @test mean(c[3, 1] for c in samp.configs) > 0.5        # still mostly along +z
    end

    @testset "fully ordered limit saturates without exploding the quadrature" begin
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[0 0; 0 0; 1 1])
        @test mfa_sublattice_m(s, 1e-8) == [1.0, 1.0]          # early return, no hang
        samp = sample(s, 5; tau = 1e-8, rng = MersenneTwister(0))
        @test all(c -> c[3, 1] > 0.99 && c[3, 2] > 0.99, samp.configs)   # pinned near +z
    end

    @testset "tensorial draws are reproducible under a fixed seed" begin
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[0 0; 0 0; 1 1])
        a = sample(s, 20; tau = 0.6, rng = MersenneTwister(3))
        b = sample(s, 20; tau = 0.6, rng = MersenneTwister(3))
        @test a.configs == b.configs
    end
end
