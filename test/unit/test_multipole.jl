# P4 of the mean-field sampler (docs/specs/mfa-sampling.md): the full multipole MFA over all
# SCE clusters and l (`MultipoleField` / `MFASampler(model::SCEModel)`). Validates that the
# many-body factorization `h_a^{lm} = Σ_φ jφ folded ∏_{b≠a} ⟨Z_b⟩` is built correctly — by
# the exact reduction to the single-global Langevin curve for a pure-bilinear model, by
# scale invariance, and (the headline higher-order check) by matching the single-site
# potential to the conditional mean SCE energy ⟨E | e_a⟩ of a biquadratic model.

using Test
using SCEFitting
using SCETools
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean

const MR = SCETools

# A 2-body lmax=[2] model carries [1,1] bilinear, [1,2]/[2,2] biquadratic, and [2]
# single-ion channels — a genuine higher-multipole many-body model.
function _biquadratic_model(seed)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    b = SCEBasis(cr, Interaction(; nbody = 2, pair_cutoff = 1.5, lmax = [2], isotropy = false))
    return SCEModel(b, 0.0, 0.05 .* randn(MersenneTwister(seed), nsalc(b)), b.salcs.keys)
end

# A clean ferromagnetic Heisenberg dimer (couples atoms 1–2; pure l=1).
function _dimer_model()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
    b = SCEBasis(cr, Interaction(; nbody = 2, pair_cutoff = 2.6, lmax = [1], isotropy = true))
    return SCEModel(b, 0.0, [-0.02], b.salcs.keys)   # negative ⇒ ferro along +z
end

@testset "full multipole sampler (P4)" begin
    @testset "MultipoleField construction" begin
        mf1 = MultipoleField(_dimer_model())
        @test mf1.natoms == 4
        @test mf1.lmax == 1
        @test length(mf1.terms) == 2                   # both directed members of the 1–2 bond
        mf2 = MultipoleField(_biquadratic_model(0))
        @test mf2.lmax == 2                             # the [2,2] biquadratic channel
        @test length(mf2.terms) > length(mf1.terms)
    end

    @testset "pure-bilinear model reduces to the single-global Langevin curve" begin
        model = _dimer_model()
        s = MFASampler(model; reference = Float64[0 0 0 0; 0 0 0 0; 1 1 1 1])
        @test MR._kind(s) == "multipole"
        for τ in (0.3, 0.6, 0.9)
            m = mfa_sublattice_m(s, τ)
            @test m[1] ≈ thermal_averaged_m(τ) atol = 1e-9   # coupled dimer
            @test m[2] ≈ thermal_averaged_m(τ) atol = 1e-9
            @test m[3] ≈ 0 atol = 1e-12                      # free spins
        end
    end

    @testset "scale invariance: scaling all couplings leaves m_a(τ) unchanged" begin
        b = _dimer_model().basis
        s1 = MFASampler(SCEModel(b, 0.0, [-0.02], b.salcs.keys);
                        reference = Float64[0 0 0 0; 0 0 0 0; 1 1 1 1])
        s9 = MFASampler(SCEModel(b, 0.0, [-0.18], b.salcs.keys);
                        reference = Float64[0 0 0 0; 0 0 0 0; 1 1 1 1])
        for τ in (0.3, 0.6, 0.9)
            @test mfa_sublattice_m(s1, τ) ≈ mfa_sublattice_m(s9, τ) atol = 1e-9
        end
        @test mfa_temperature_scale(s9) ≈ 9 * mfa_temperature_scale(s1)
    end

    @testset "many-body factorization: V_a/β equals the conditional mean energy ⟨E|e_a⟩" begin
        # The MFA single-site potential is V_a(e)/β = ⟨E | e_a = e⟩ − const, where neighbors
        # are averaged over their single-site distributions and the product factorizes
        # ⟨∏ Z⟩ → ∏⟨Z⟩. Check the e-dependence against a direct Monte-Carlo conditional mean
        # energy of the biquadratic model — an independent test of `_site_coeffs_all!`.
        model = _biquadratic_model(0)
        mf = MultipoleField(model)
        ref = Float64[0 0; 0 0; 1 1]
        ehat = [SVector{3,Float64}(ref[:, a]) for a = 1:2]
        ρ, _ = MR._perron(MR._mfa_matrix(mf.bilinear, ref))
        τ = 0.7
        cs, _ = MR._multipole_state(mf, ehat, ρ, τ)
        β = 3 / (ρ * τ)
        rng = MersenneTwister(1)
        # draw site 2 from its converged single-site distribution
        chain2 = MR.sample_site_metropolis(rng, cs[2], 6000; e_init = ehat[2], nburn = 500, thin = 8)
        cond_E(e1) = mean(predict_energy(model, hcat(Vector(e1), Vector(e2))) for e2 in chain2)
        e0 = SVector{3,Float64}(0, 0, 1)
        base = cond_E(e0)
        for e in (SVector{3,Float64}(1, 0, 0), SVector{3,Float64}(0, 1, 0),
                  normalize(SVector{3,Float64}(1, 1, 1)), normalize(SVector{3,Float64}(1, -2, 1)))
            dV = (MR._site_potential(cs[1], e) - MR._site_potential(cs[1], e0)) / β
            dE = cond_E(e) - base
            @test dV ≈ dE atol = 3e-2
        end
    end

    @testset "Metropolis draw reproduces the full multipole self-consistency" begin
        model = _biquadratic_model(0)
        mf = MultipoleField(model)
        ref = Float64[0 0; 0 0; 1 1]
        ehat = [SVector{3,Float64}(ref[:, a]) for a = 1:2]
        ρ, _ = MR._perron(MR._mfa_matrix(mf.bilinear, ref))
        τ = 0.8
        cs, _ = MR._multipole_state(mf, ehat, ρ, τ)
        avg = MR.multipole_average(cs[1], mf.lmax)
        rng = MersenneTwister(4)
        step = clamp(1.5 / sqrt(1 + MR._field_scale(cs[1])), 0.05, 0.8)
        chain = MR.sample_site_metropolis(rng, cs[1], 8000; e_init = ehat[1], step = step,
                                          nburn = 600, thin = 12)
        for (l, m) in ((1, 0), (2, 0), (2, 2))
            mc = mean(MR.Harmonics.Zlm(l, m, e) for e in chain)
            @test mc ≈ avg[MR.Harmonics.lm_index(l, m)] atol = 3e-2
        end
    end

    @testset "P4 draws are reproducible under a fixed seed" begin
        s = MFASampler(_dimer_model(); reference = Float64[0 0 0 0; 0 0 0 0; 1 1 1 1])
        a = sample(s, 15; tau = 0.6, rng = MersenneTwister(3))
        b = sample(s, 15; tau = 0.6, rng = MersenneTwister(3))
        @test a.configs == b.configs
    end
end
