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
        cs, mq = MR._tensor_state(s.source, MR._ehat(s), s.rho, τ)
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

    @testset "ordered limit: a single-ion-only atom has m → 0, not 1" begin
        # Atoms 1,2 carry the exchange ordering channel; atom 3 has no bilinear coupling, only
        # an easy-axis single-ion term. At τ → 0 its distribution is an e → −e symmetric
        # double-well along ±ê, so ⟨e·ê_3⟩ → 0 (no net l=1 field) — m must NOT saturate to 1.
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)        # easy-axis along z
        Jiso = [0.0 -1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0]
        s = MFASampler(ExchangeModel(Jiso; onsite = [zero(A), zero(A), A]);
                       reference = Float64[0 0 0; 0 0 0; 1 1 1])
        m = mfa_sublattice_m(s, 1e-8)
        @test m[1] ≈ 1.0 && m[2] ≈ 1.0                              # exchange-ordered
        @test m[3] == 0.0                                           # single-ion only ⇒ no net m
    end

    @testset "single-ion-only atom: the drawn ensemble matches its m ≈ 0 label" begin
        # The Metropolis regression for the flip proposal: atom 3 (no bilinear coupling,
        # strong easy-axis single-ion) has an e ↔ −e symmetric double-well, so its label
        # m₃ ≈ 0 — and the DRAWN configurations must agree (⟨e_z⟩ ≈ 0). A rotation-only
        # chain started at +ê₃ stays in the +ẑ lobe (barrier ≫ kT) and would report
        # ⟨e_z⟩ ≈ +1, silently contradicting the ensemble's own m label.
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)
        Jiso = [0.0 -1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0]
        s = MFASampler(ExchangeModel(Jiso; onsite = [zero(A), zero(A), 4 .* A]);
                       reference = Float64[0 0 0; 0 0 0; 1 1 1])
        τ = 0.5
        @test abs(mfa_sublattice_m(s, τ)[3]) < 1e-6                 # the quadrature label
        samp = sample(s, 1500; tau = τ, rng = MersenneTwister(9))
        @test abs(mean(c[3, 3] for c in samp.configs)) < 0.1        # not lobe-trapped
        @test mean(c[3, 1] for c in samp.configs) > 0.5             # exchange atoms ordered
    end

    @testset "fixed / uniform / randomize on the Metropolis path" begin
        # These keywords were only gated on the closed-form vMF path; the Metropolis sweep
        # implements them independently, so pin the same semantics here.
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[0 0; 0 0; 1 1])
        @test MR._needs_metropolis(s)
        # fixed atoms stay exactly at the reference (no randomize)
        samp = sample(s, 6; tau = 0.6, rng = MersenneTwister(4), fixed = [2])
        @test all(c -> c[:, 2] ≈ [0, 0, 1.0], samp.configs)
        # with randomize, fixed atoms ride the frame: unit norm, mutually parallel (both
        # references are +ẑ), and generically no longer along +ẑ itself
        sr = sample(s, 4; tau = 0.6, rng = MersenneTwister(4), fixed = [1, 2],
                    randomize = true)
        for c in sr.configs
            @test norm(c[:, 1]) ≈ 1 atol = 1e-12
            @test dot(c[:, 1], c[:, 2]) ≈ 1.0 atol = 1e-12
        end
        @test any(c -> abs(c[3, 1]) < 0.99, sr.configs)
        # uniform atoms are isotropic regardless of the tensorial field
        su = sample(s, 3000; tau = 0.3, rng = MersenneTwister(6), uniform = [1])
        @test abs(mean(c[3, 1] for c in su.configs)) < 0.06
        @test mean(c[3, 2] for c in su.configs) > 0.7
        # guards (shared _resolve_taus / index validation on this path too)
        @test_throws ArgumentError sample(s, 1; tau = 0.5, fixed = [99])
        @test_throws ArgumentError sample(s, 2; tau = -0.1)
    end
end
