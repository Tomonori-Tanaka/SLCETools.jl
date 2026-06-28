# P2 of the mean-field sampler (docs/specs/mfa-sampling.md): the multi-sublattice
# isotropic `ExchangeModel` sampler. Validates the coupled per-atom self-consistency
# `m_a = L(3(Ā m)_a/τ)` — that it reduces to the single-global Langevin curve for a
# uniform ferromagnet, gives distinct sublattice collapse rates on a ferrimagnet at a
# single T_MF, is scale-invariant in the couplings, and that the drawn configs carry the
# per-atom magnetizations.

using Test
using SCEFitting
using SCETools
using LinearAlgebra
using Random
using Statistics: mean

const MR = SCETools

# project a drawn config's spins onto their reference directions, averaged over configs
function _proj_per_atom(configs, ref)
    n = size(ref, 2)
    return [mean(dot(c[:, a], ref[:, a] / norm(ref[:, a])) for c in configs) for a = 1:n]
end

@testset "ExchangeModel sampler (multi-sublattice, P2)" begin
    @testset "ExchangeModel construction validates and symmetrizes" begin
        m = ExchangeModel([0.0 -1.0; -1.0 0.0])
        @test m.natoms == 2
        @test m.Jiso == [0.0 -1.0; -1.0 0.0]
        @test_throws ArgumentError ExchangeModel([1.0 2.0 3.0])           # not square
        @test_throws ArgumentError ExchangeModel([0.0 1.0; -1.0 0.0])     # not symmetric
    end

    @testset "uniform ferromagnet reduces to the single-global Langevin curve" begin
        # 2-atom ferromagnet (E = J e·e, J<0 favors alignment), both spins +z.
        exch = ExchangeModel([0.0 -1.0; -1.0 0.0])
        ref = Float64[0 0; 0 0; 1 1]
        s = MFASampler(exch; reference = ref)
        @test mfa_temperature_scale(s) ≈ 1 / 3          # ρ = 1 ⇒ T_MF = 1/3
        for τ in (0.2, 0.5, 0.8, 0.95)
            mc = mfa_sublattice_m(s, τ)
            @test mc[1] ≈ mc[2] atol = 1e-10            # symmetric sublattices
            @test mc[1] ≈ thermal_averaged_m(τ) atol = 1e-9   # coupled ≡ single-global
        end
    end

    @testset "antiferromagnet folds to the same curve (reference in A)" begin
        # 2-atom AFM (E = J e·e, J>0), reference up/down: folding ê_a·ê_b = −1 makes it
        # ferromagnetic in the magnitudes, so m_a matches the ferromagnet.
        exch = ExchangeModel([0.0 1.0; 1.0 0.0])
        ref = Float64[0 0; 0 0; 1 -1]
        s = MFASampler(exch; reference = ref)
        @test mfa_temperature_scale(s) ≈ 1 / 3
        for τ in (0.3, 0.7)
            @test mfa_sublattice_m(s, τ)[1] ≈ thermal_averaged_m(τ) atol = 1e-9
        end
    end

    @testset "ferrimagnet: distinct sublattice collapse rates, single T_MF" begin
        # Star A–B1, A–B2 (antiferro J>0); A has coordination 2, each B coordination 1.
        # ρ = √2, leading eigenvector ∝ (√2, 1, 1) ⇒ m_A > m_B at every τ, same T_MF.
        J = zeros(3, 3)
        J[1, 2] = J[2, 1] = 1.0
        J[1, 3] = J[3, 1] = 1.0
        exch = ExchangeModel(J)
        ref = Float64[0 0 0; 0 0 0; 1 -1 -1]
        s = MFASampler(exch; reference = ref)
        @test mfa_temperature_scale(s) ≈ sqrt(2) / 3
        prev = nothing
        for τ in (0.2, 0.5, 0.8, 0.95)
            m = mfa_sublattice_m(s, τ)
            @test m[2] ≈ m[3] atol = 1e-10              # B1, B2 equivalent
            @test m[1] > m[2] + 1e-3                    # A more ordered than B
            @test all(0 .< m .< 1)
            prev === nothing || @test all(m .< prev)    # both decrease with τ
            prev = m
        end
        # τ → 0 saturates, τ → 1⁻ collapses
        @test all(≈(1.0), mfa_sublattice_m(s, 1e-7))
        @test all(<(0.05), mfa_sublattice_m(s, 0.999))
    end

    @testset "scale invariance: scaling all couplings leaves m_a(τ) unchanged" begin
        J = zeros(3, 3)
        J[1, 2] = J[2, 1] = 1.0
        J[1, 3] = J[3, 1] = 1.0
        ref = Float64[0 0 0; 0 0 0; 1 -1 -1]
        s1 = MFASampler(ExchangeModel(J); reference = ref)
        s7 = MFASampler(ExchangeModel(7.0 .* J); reference = ref)
        for τ in 0.1:0.1:0.95
            @test mfa_sublattice_m(s1, τ) ≈ mfa_sublattice_m(s7, τ) atol = 1e-12
        end
        @test mfa_temperature_scale(s7) ≈ 7 * mfa_temperature_scale(s1)   # T_MF does scale
    end

    @testset "drawn configs carry the per-atom magnetizations" begin
        J = zeros(3, 3)
        J[1, 2] = J[2, 1] = 1.0
        J[1, 3] = J[3, 1] = 1.0
        ref = Float64[0 0 0; 0 0 0; 1 -1 -1]
        s = MFASampler(ExchangeModel(J); reference = ref)
        samp = sample(s, 4000; tau = 0.5, rng = MersenneTwister(11))
        target = mfa_sublattice_m(s, 0.5)
        @test _proj_per_atom(samp.configs, ref) ≈ target atol = 2e-2
        @test samp.m[1] ≈ target                        # label carries the per-atom m
        @test all(c -> all(a -> abs(norm(@view c[:, a]) - 1) < 1e-10, 1:3), samp.configs)
    end

    @testset "from a fitted SCE: Heisenberg isotropic part is extracted as tr(M)/3" begin
        # This 4-atom fixture's single Heisenberg SALC couples atoms 1–2 only (a dimer);
        # atoms 3–4 are free. It exercises the tr(M)/3 isotropic extraction and the coupled
        # solve's handling of uncoupled (free) spins.
        lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
        b = SCEBasis(cr, Interaction(; nbody = 2, pair_cutoff = 2.6, lmax = [1], isotropy = true))
        model = SCEPredictor(b, 0.0, [0.0137], b.salc_basis.keys)
        ex = ExchangeModel(model)
        @test ex.natoms == 4
        @test ex.Jiso ≈ ex.Jiso'                                   # symmetric
        jnn = ex.Jiso[1, 2]
        @test abs(jnn) > 0                                         # the coupled dimer
        @test count(!=(0), ex.Jiso) == 2                          # only (1,2) and (2,1)
        # ordering pattern selected by the coupling sign (E = Σ J e·e)
        ref = jnn < 0 ? Float64[0 0 0 0; 0 0 0 0; 1 1 1 1] :
              Float64[0 0 0 0; 0 0 0 0; 1 -1 1 -1]
        s = MFASampler(ex; reference = ref)
        for τ in (0.3, 0.7, 0.9)
            m = mfa_sublattice_m(s, τ)
            @test m[1] ≈ thermal_averaged_m(τ) atol = 1e-9        # coupled dimer ≡ Langevin
            @test m[2] ≈ thermal_averaged_m(τ) atol = 1e-9
            @test m[3] ≈ 0 atol = 1e-12                           # free spins disorder fully
            @test m[4] ≈ 0 atol = 1e-12
        end
        # fully ordered limit must stay per-atom: coupled spins saturate, free spins do not
        mlo = mfa_sublattice_m(s, 1e-7)
        @test mlo[1] ≈ 1 && mlo[2] ≈ 1
        @test mlo[3] ≈ 0 && mlo[4] ≈ 0
        lo = sample(s, 50; tau = 1e-7, rng = MersenneTwister(2))
        @test all(c -> c[:, 1] ≈ ref[:, 1] && c[:, 2] ≈ ref[:, 2], lo.configs)  # dimer ordered
        @test abs(mean(c[3, 3] for c in lo.configs)) < 0.3                       # free spin random
    end

    @testset "from a fitted SCE: single-ion kept (P3), higher-l reported" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        # a single-ion (ls=[2]) model: now extracted into onsite (tensorial), not dropped
        cr1 = Crystal(lat, reshape([0.0, 0, 0], 3, 1), [1], ["Fe"])
        b1 = SCEBasis(cr1, Interaction(; nbody = 1, pair_cutoff = 1.5, lmax = [2], isotropy = false))
        m1 = SCEPredictor(b1, 0.0, ones(n_salcs(b1)), b1.salc_basis.keys)
        ex1 = ExchangeModel(m1)
        @test !ex1.isotropic
        @test norm(ex1.onsite[1]) > 0
        # higher-l 2-body channels ([1,2], [2,2]) are unsupported and reported
        cr2 = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        b2 = SCEBasis(cr2, Interaction(; nbody = 2, pair_cutoff = 1.5, lmax = [2], isotropy = false))
        m2 = SCEPredictor(b2, 0.0, ones(n_salcs(b2)), b2.salc_basis.keys)
        @test_logs (:warn,) ExchangeModel(m2)
    end

    @testset "guards: dimension mismatch and no-instability reference" begin
        exch = ExchangeModel([0.0 -1.0; -1.0 0.0])
        @test_throws DimensionMismatch MFASampler(exch; reference = Float64[0; 0; 1;;])
        # zero couplings ⇒ no ordering instability ⇒ rejected
        @test_throws ArgumentError MFASampler(ExchangeModel(zeros(2, 2));
                                              reference = Float64[0 0; 0 0; 1 1])
    end
end
