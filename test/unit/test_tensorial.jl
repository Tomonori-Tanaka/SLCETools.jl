# P3 of the mean-field sampler (docs/specs/mfa-sampling.md): tensorial exchange (DMI +
# anisotropic) and single-ion anisotropy, drawn from the Bingham single-site potential via
# the general Metropolis engine. Validates the single-ion Bingham shapes (easy-axis cone
# sharpening / above-T_MF persistence, easy-plane girdle), that the Metropolis draw
# reproduces the quadrature self-consistency, and that DMI tilts a collinear reference.

using Test
using SLCE
using SLCETools
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean

const MR = SLCETools

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
        # Cross-METHOD consistency (stochastic vs deterministic integration of
        # the same converged field), not an independent oracle: both sides read
        # `_tensor_state`'s coefficients, so a wrong single-site field passes
        # here — it is caught by the rotational-covariance gate below instead.
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

    # `_l1_coeffs!` / `_l2_coeffs!` are the FORWARD of the core's `_l1_pair_matrix` /
    # `_l2_onsite_matrix`, and every other single-ion test in this package feeds them a
    # DIAGONAL tensor — `diag(1,1,-2)`, `diag(0,0,3)`. On a diagonal `A` four of the five
    # `l=2` branches are identically zero (`axy = ayz = axz = 0` and `axx − ayy = 0`), so
    # a swapped `axz ↔ ayz`, a sign flip, or a dropped factor of 2 in `m = ±1, ±2` was
    # silent everywhere. The path is reachable: a fitted low-symmetry single-ion tensor
    # flows `bilinear_terms → _extract_bilinear_onsite → onsite[a] → _site_coeffs →
    # site_potential`, and the visible symptom would be an anisotropy axis pointing the
    # wrong way — a plausible wrong number, not a crash.
    #
    # These pin the writers SEMANTICALLY, against `Zlm` itself, mirroring the core's own
    # `test_sunny.jl` gate on the inverse direction. A round-trip against
    # `_l2_onsite_matrix` would NOT do: it is satisfied by any pair of mutually
    # consistent but jointly wrong conventions, and in this direction it is not even the
    # identity — `_l2_coeffs!` discards the trace and the antisymmetric part, so
    # `_l2_onsite_matrix ∘ _l2_coeffs!` is a projection.
    @testset "the l=1/l=2 coefficient writers reproduce their forms against Zlm" begin
        rng = MersenneTwister(3)
        e1 = e2 = 0.0
        for _ = 1:200
            g = SVector{3,Float64}(randn(rng, 3))
            A = SMatrix{3,3,Float64}(randn(rng, 3, 3))   # neither symmetric nor traceless
            c = zeros(9)
            MR._l1_coeffs!(c, g)
            MR._l2_coeffs!(c, A)
            e = normalize(SVector{3,Float64}(randn(rng, 3)))
            # the traceless symmetric part is what `e' A e` reduces to on the unit sphere
            S = (A + transpose(A)) / 2
            At = S - (tr(S) / 3) * I
            e1 = max(e1, abs(sum(c[MR.Harmonics.lm_index(1, m)] * _Z(1, m, e)
                                 for m = -1:1) - dot(g, e)))
            e2 = max(e2, abs(sum(c[MR.Harmonics.lm_index(2, m)] * _Z(2, m, e)
                                 for m = -2:2) - dot(e, At * e)))
        end
        @test e1 < 1e-12
        @test e2 < 1e-12

        # non-vacuity: a diagonal tensor — what every other test here uses — leaves the
        # four branches this testset exists for at exactly zero.
        cd = zeros(9)
        MR._l2_coeffs!(cd, SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2))
        @test cd[MR.Harmonics.lm_index(2, 0)] != 0.0
        @test all(cd[MR.Harmonics.lm_index(2, m)] == 0.0 for m in (-2, -1, 1, 2))
    end

    # The end-to-end complement: helper-level pins do not prove the helper is REACHED.
    # Rotating the single-ion tensor and the reference by the same R must leave the
    # sublattice magnetization invariant — exact physics, and exactly what a wrong
    # off-diagonal branch breaks. The rotation is generic (no axis aligned with x/y/z),
    # so all four otherwise-untested branches carry weight.
    @testset "single-ion anisotropy is rotationally covariant" begin
        Rz(t) = SMatrix{3,3,Float64}(cos(t), sin(t), 0, -sin(t), cos(t), 0, 0, 0, 1)
        Ry(t) = SMatrix{3,3,Float64}(cos(t), 0, -sin(t), 0, 1, 0, sin(t), 0, cos(t))
        R = Rz(0.7) * Ry(0.9) * Rz(0.3)
        A0 = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)     # easy axis along z
        A = R * A0 * transpose(R)
        @test all(abs.((A[1, 2], A[2, 3], A[1, 3], A[1, 1] - A[2, 2])) .> 1e-3)

        n = R * SVector{3,Float64}(0, 0, 1)
        J = [0.0 -1.0; -1.0 0.0]
        ref_n = hcat(Vector(n), Vector(n))
        s_z = MFASampler(ExchangeModel(J; onsite = [A0, A0]);
                         reference = Float64[0 0; 0 0; 1 1])
        s_R = MFASampler(ExchangeModel(J; onsite = [A, A]); reference = ref_n)
        # control: rotate the reference but NOT the tensor — the invariance must FAIL,
        # or the comparison above would be measuring nothing.
        s_bad = MFASampler(ExchangeModel(J; onsite = [A0, A0]); reference = ref_n)
        for τ in (0.4, 0.8, 1.2)
            mz = mfa_sublattice_m(s_z, τ)[1]
            @test isapprox(mfa_sublattice_m(s_R, τ)[1], mz; atol = 1e-6)
            @test abs(mfa_sublattice_m(s_bad, τ)[1] - mz) > 0.1
        end
    end
end
