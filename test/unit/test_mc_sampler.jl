# The Metropolis MC sampler (docs/specs/mc-sampling.md): single-spin Metropolis on the
# joint Boltzmann distribution of a fitted SCE at an absolute temperature. Gates: the
# machine-precision local↔global energy consistency (ΔE = c_a·ΔZ against `predict_energy`),
# the exact two-spin correlation ⟨e₁·e₂⟩ = −L(βJ), the single-site Langevin limit (where
# the mean field is exact, so this doubles as the MFA cross-link), the `randomize` global
# rotation (isotropic energy invariance), seed reproducibility, and the guards.

using Test
using SCEFitting
using SCETools
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean

const MR = SCETools

# The same fixtures as test_multipole.jl: a genuine higher-multipole two-atom model and a
# clean ferromagnetic Heisenberg dimer (couples atoms 1–2; atoms 3–4 free).
function _mc_biquadratic_model(seed)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, pair_cutoff = 1.5, lmax = [2], isotropy = false))
    return SCEPredictor(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

function _mc_dimer_model()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, pair_cutoff = 2.6, lmax = [1], isotropy = true))
    return SCEPredictor(b, 0.0, vcat([-0.02], zeros(n_salcs(b) - 1)))   # negative ⇒ ferro
end

# A single spin in an l=1 field along +z: V(e) = c0·Z_10(e) = c0·N1·e_z.
_mc_field_sampler(c0) = MetropolisSampler(1, 1, [MR._MFATerm(c0, [1], [1], [0.0, 1.0, 0.0])])

_mc_langevin(x) = coth(x) - 1 / x

_mc_rand_config(rng, n) = reduce(hcat, [Vector(MR._random_unit(rng)) for _ = 1:n])

@testset "Metropolis MC sampler" begin
    @testset "construction from a fitted model" begin
        s = MetropolisSampler(_mc_dimer_model())
        @test s isa AbstractSampler
        @test s.natoms == 4
        @test s.lmax == 1
        @test length(s.terms) == 2                    # both directed members of the 1–2 bond
        @test s.terms_of[1] == [1, 2] || length(s.terms_of[1]) == 2
        @test isempty(s.terms_of[3])                  # free spin: no terms
        @test s.reference === nothing
        sr = MetropolisSampler(_mc_dimer_model(); reference = Float64[0 0 0 0; 0 0 0 0; 2 2 2 2])
        @test sr.reference ≈ Float64[0 0 0 0; 0 0 0 0; 1 1 1 1]   # normalized
        @test sprint(show, s) == "MetropolisSampler(4 atoms, lmax=1, 2 terms)"
    end

    @testset "local update ↔ global energy, machine precision" begin
        model = _mc_biquadratic_model(0)
        s = MetropolisSampler(model)
        nlm = (s.lmax + 1)^2
        rng = MersenneTwister(7)
        cfg = _mc_rand_config(rng, s.natoms)
        Z = [MR._zlm_row!(zeros(nlm), SVector{3,Float64}(cfg[:, a]), s.lmax)
             for a = 1:s.natoms]

        # the total contraction reproduces predict_energy − j0 (fixture j0 = 0)
        @test MR._total_energy(s.terms, Z) ≈ predict_energy(model, cfg) atol = 1e-12

        for a = 1:s.natoms
            # single-site coefficients ≡ the a-th row of the mean-field all-sites build (β=1)
            c = zeros(nlm)
            for t in s.terms_of[a]
                term = s.terms[t]
                MR._accumulate_site_term!(c, a, term.coef, term.atoms, term.ls,
                                          term.folded, Z)
            end
            cs = [zeros(nlm) for _ = 1:s.natoms]
            MR._site_coeffs_all!(cs, s.terms, Z, 1.0, s.natoms)
            @test c ≈ cs[a] atol = 1e-13

            # ΔE = c_a·ΔZ ≡ the full-energy difference ≡ the model-energy difference
            e2 = MR._random_unit(rng)
            znew = MR._zlm_row!(zeros(nlm), e2, s.lmax)
            ΔE = dot(c, znew - Z[a])
            cfg2 = copy(cfg)
            cfg2[:, a] = e2
            @test ΔE ≈ MR._config_energy(s, cfg2) - MR._config_energy(s, cfg) atol = 1e-12
            @test ΔE ≈ predict_energy(model, cfg2) - predict_energy(model, cfg) atol = 1e-12
        end
    end

    @testset "exact two-spin gate: ⟨e₁·e₂⟩ = −L(βJ)" begin
        model = _mc_dimer_model()
        s = MetropolisSampler(model)
        J = ExchangeModel(model).Jiso[1, 2]
        @test J < 0                                    # ferro
        for (βJmag, nconf) in ((1.0, 4000), (2.5, 4000))
            T = abs(J) / βJmag
            samp = sample(s, nconf; temperature = T, burnin = 400, thin = 6,
                          rng = MersenneTwister(11))
            c12 = mean(dot(c[:, 1], c[:, 2]) for c in samp)
            exact = -_mc_langevin(J / T)
            @test c12 ≈ exact atol = 0.03
            # the energy diagnostic itself: only the 1–2 bond carries energy
            @test mean(samp.energy) ≈ J * exact atol = 0.03 * abs(J)
            # the free spins are uniform: vanishing mean direction
            m3 = mean(c[:, 3] for c in samp)
            @test norm(m3) < 0.05
        end
    end

    @testset "single-site Langevin limit (mean field exact here)" begin
        c0 = 0.05
        s = _mc_field_sampler(c0)
        h = c0 * MR.Harmonics.N1                       # V(e) = h·e_z
        for βh in (1.0, 3.0)
            T = h / βh
            samp = sample(s, 4000; temperature = T, burnin = 400, thin = 5,
                          rng = MersenneTwister(2))
            mz = mean(c[3, 1] for c in samp)
            @test mz ≈ -_mc_langevin(βh) atol = 0.03      # P ∝ exp(−βh·e_z)
        end
    end

    @testset "randomize: Haar rotation of the stored copy, isotropic energy invariant" begin
        s = MetropolisSampler(_mc_dimer_model())
        # Same seed ⇒ both chains are identical up to the first storage point; the
        # randomize run then rotates its stored copy. Isotropy ⇒ equal energy; the Gram
        # matrix (all relative angles) is rotation-invariant.
        a = sample(s, 1; temperature = 0.02, burnin = 50, thin = 5,
                   rng = MersenneTwister(5))
        b = sample(s, 1; temperature = 0.02, burnin = 50, thin = 5,
                   rng = MersenneTwister(5), randomize = true)
        @test !(a.configs[1] ≈ b.configs[1])           # actually rotated
        @test b.energy[1] ≈ a.energy[1] atol = 1e-12   # isotropic invariance
        @test transpose(b.configs[1]) * b.configs[1] ≈
              transpose(a.configs[1]) * a.configs[1] atol = 1e-12
        @test all(abs(norm(b.configs[1][:, k]) - 1) < 1e-12 for k = 1:4)
        # the stored energy label always matches the stored configuration
        @test b.energy[1] ≈ MR._config_energy(s, b.configs[1]) atol = 1e-12
    end

    @testset "seed reproducibility (byte-identical), incl. sweep + randomize" begin
        s = MetropolisSampler(_mc_dimer_model())
        kw = (; temperature = [0.05, 0.02], nsamples = 5, burnin = 30, thin = 3,
              randomize = true)
        a = sample(s; kw..., rng = MersenneTwister(3))
        b = sample(s; kw..., rng = MersenneTwister(3))
        @test a.configs == b.configs
        @test a.energy == b.energy
        @test a.acceptance == b.acceptance
        @test a.temperature == b.temperature
    end

    @testset "sweep structure, warm start, diagnostics" begin
        s = MetropolisSampler(_mc_dimer_model())
        samp = sample(s; temperature = [0.2, 0.005], nsamples = 4, burnin = 100,
                      thin = 4, rng = MersenneTwister(9))
        @test length(samp) == 8
        @test samp.temperature == [fill(0.2, 4); fill(0.005, 4)]   # value-outer
        @test all(0 .< samp.acceptance .<= 1)
        # annealing high → low: the ferro dimer's bond energy drops toward J
        @test mean(samp.energy[5:8]) < mean(samp.energy[1:4])
        # MCSample array interface
        @test samp[1] isa Matrix{Float64}
        @test samp[end] === samp.configs[8]
        @test collect(samp) == samp.configs
        @test eltype(MCSample) === Matrix{Float64}
        @test sprint(show, samp) == "MCSample(8 configs)"
    end

    @testset "init resolution" begin
        ref = Float64[0 0 0 0; 0 0 0 0; 1 1 1 1]
        s = MetropolisSampler(_mc_dimer_model(); reference = ref)
        # burnin = 0, thin = 1 at low T from the reference: stays near it
        samp = sample(s, 1; temperature = 1e-4, burnin = 0, thin = 1,
                      rng = MersenneTwister(1))
        @test mean(samp.configs[1][3, 1:2]) > 0.9
        # explicit init overrides the reference
        init = Float64[0 0 0 0; 0 0 0 0; -1 -1 -1 -1]
        samp2 = sample(s, 1; temperature = 1e-4, burnin = 0, thin = 1,
                       rng = MersenneTwister(1), init = init)
        @test mean(samp2.configs[1][3, 1:2]) < -0.9
    end

    @testset "guards" begin
        model = _mc_dimer_model()
        s = MetropolisSampler(model)
        @test_throws ArgumentError sample(s, 1; temperature = 0.0)
        @test_throws ArgumentError sample(s, 1; temperature = -0.1)
        @test_throws ArgumentError sample(s; temperature = [0.1, 0.0])
        @test_throws ArgumentError sample(s; temperature = Float64[])
        @test_throws ArgumentError sample(s, -1; temperature = 0.1)
        @test_throws ArgumentError sample(s, 1; temperature = [0.1, 0.2])  # scalar form
        @test_throws ArgumentError sample(s, 1; temperature = 0.1, thin = 0)
        @test_throws ArgumentError sample(s, 1; temperature = 0.1, burnin = -1)
        @test_throws ArgumentError sample(s, 1; temperature = 0.1, step = 0.0)
        @test_throws DimensionMismatch sample(s, 1; temperature = 0.1,
                                              init = Float64[0 0; 0 0; 1 1])
        @test_throws ArgumentError sample(s, 1; temperature = 0.1,
                                          init = Float64[0 0 0 0; 0 0 0 0; 1 1 1 0])
        # ctor invariants
        t = MR._MFATerm(1.0, [1], [1], [0.0, 1.0, 0.0])
        @test_throws ArgumentError MetropolisSampler(0, 1, [t])
        @test_throws ArgumentError MetropolisSampler(1, 1, MR._MFATerm[])
        @test_throws ArgumentError MetropolisSampler(1, 0, [t])              # lmax too small
        @test_throws ArgumentError MetropolisSampler(1, 1,
            [MR._MFATerm(1.0, [2], [1], [0.0, 1.0, 0.0])])                   # atom out of range
        @test_throws ArgumentError MetropolisSampler(2, 1,
            [MR._MFATerm(1.0, [1, 1], [1, 1], zeros(3, 3))])                 # repeated atom
        @test_throws DimensionMismatch MetropolisSampler(model;
            reference = Float64[0 0; 0 0; 1 1])
        # result-type parallel invariant
        @test_throws DimensionMismatch MCSample([zeros(3, 1)], [0.1, 0.2], [0.0], [1.0])
    end
end
