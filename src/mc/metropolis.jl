# Metropolis Monte-Carlo sampler over a fitted SLCE (see `docs/specs/mc-sampling.md`).
#
# Unlike the mean-field `MFASampler` (single-site, correlation-free), this samples the
# *joint* Boltzmann distribution of the fitted multipole Hamiltonian on the training cell
# by sequential single-spin Metropolis updates. The control variable is the absolute
# temperature `k_B·T` in the model's energy units — no reduced `τ`, no `l=1` Perron scale,
# so it works for models without a bilinear channel. It reuses the mean-field digest
# (`_MFATerm` via `_scaled_multipole_terms`, the `(4π)^(N/2)` scale applied exactly once
# there) and the engine's symmetric proposal primitives (`_rotate` + antipodal flip).
#
# The local update leans on the same leave-one-out structure as the mean-field
# `_site_coeffs_all!`, but contracted against the *concrete* neighbor harmonics
# `Z_lm(e_b)` of the current configuration instead of thermal averages: with every
# cluster's atoms distinct (asserted), site `a` appears exactly once per term, so its
# coefficient vector `c_a` is independent of `e_a` and the exact energy change of a
# single-spin move is `ΔE = c_a · (Z(e′) − Z(e))`. β enters only in the accept step —
# `c_a` and every stored energy stay in the model's energy units.

# `KB_EV` and the kelvin/kT resolution (`SLCE.resolve_kt`) are NOT defined here — they
# live in the core, which SLCEMonteCarlo re-exports too. Both packages used to carry a
# character-for-character copy; two copies of a unit conversion are two things that can
# drift apart while both suites stay green.

"""
    MetropolisSampler(model::SLCEModel; reference = nothing)

Single-spin Metropolis Monte-Carlo sampler of the **joint** Boltzmann distribution
`P({e}) ∝ exp(−E({e})/k_BT)` of a fitted SLCE on its training cell (periodic images are
already folded into the fitted terms). Complements the mean-field [`MFASampler`](@ref):
the draws carry the model's true inter-site correlations, at the cost of a Markov chain
(burn-in / thinning) instead of closed-form single-site draws.

The control variable of [`sample`](@ref) is the **absolute temperature** — `temperature`
in kelvin (converted with `KB_EV`; assumes an eV-fitted model) or `kT` directly
in the model's energy units — not the reduced `τ = T/T_MF` of the mean-field sampler.
Any body order works, including models without a bilinear (`l=1`) channel.

`reference` (optional, `3 × n_atoms` unit columns) sets the default chain start; without
it a chain starts from a uniform-random configuration (see `init` in [`sample`](@ref)).

The test-facing inner form `MetropolisSampler(n_atoms, lmax, terms; reference)` accepts a
hand-built term list.
"""
struct MetropolisSampler <: AbstractSampler
    n_atoms::Int
    lmax::Int
    terms::Vector{_MFATerm}            # coef = jϕ·(4π)^(N/2), model energy units
    terms_of::Vector{Vector{Int}}      # per-atom adjacency: indices into `terms`
    reference::Union{Nothing,Matrix{Float64}}

    function MetropolisSampler(n_atoms::Int, lmax::Int, terms::Vector{_MFATerm};
                               reference::Union{Nothing,AbstractMatrix{<:Real}} = nothing)
        n_atoms >= 1 || throw(ArgumentError("n_atoms must be ≥ 1; got $n_atoms"))
        isempty(terms) && throw(ArgumentError("the term list is empty"))
        for t in terms
            all(a -> 1 <= a <= n_atoms, t.atoms) || throw(ArgumentError(
                "term atoms $(t.atoms) outside 1:$n_atoms"))
            # Site `a` appearing once per term is what makes `c_a` independent of `e_a`
            # (the ΔE locality the sweep relies on); `_scaled_multipole_terms` guarantees
            # it for fitted models, assert it for hand-built term lists.
            allunique(t.atoms) || throw(ArgumentError(
                "term with a repeated atom $(t.atoms); the single-spin update assumes " *
                "distinct sites per cluster"))
            maximum(t.ls) <= lmax || throw(ArgumentError(
                "term ls $(t.ls) exceeds lmax = $lmax"))
        end
        ref = if reference === nothing
            nothing
        else
            r = _normalize_reference(reference)
            size(r, 2) == n_atoms || throw(DimensionMismatch(
                "reference has $(size(r, 2)) atoms but the sampler has $n_atoms"))
            r
        end
        terms_of = [Int[] for _ = 1:n_atoms]
        for (t, term) in enumerate(terms), a in term.atoms
            push!(terms_of[a], t)
        end
        return new(n_atoms, lmax, terms, terms_of, ref)
    end
end

MetropolisSampler(model::SLCEModel;
                  reference::Union{Nothing,AbstractMatrix{<:Real}} = nothing) = begin
    terms, lmax = _scaled_multipole_terms(model)
    MetropolisSampler(n_atoms(model), lmax, terms; reference = reference)
end

Base.show(io::IO, s::MetropolisSampler) =
    print(io, "MetropolisSampler(", s.n_atoms, " atoms, lmax=", s.lmax, ", ",
          length(s.terms), " terms)")

"""
    MCSample

Labeled result of [`sample`](@ref) on a [`MetropolisSampler`](@ref). `configs` is the
bare `Vector{Matrix{Float64}}` (each `3 × n_atoms` unit directions); the parallel labels
are `kT` (`k_B·T` in the model's energy units — always well-defined), `temperature`
(kelvin, `= kT / KB_EV` — meaningful for an eV-fitted model), `energy` (the SLCE energy
of that stored configuration, `j0` excluded — a constant shift), and `acceptance` (the
Metropolis accept fraction over the sweeps that produced it; the first configuration at
each temperature includes its burn-in window). The object is iterable and indexable as
its `configs`.
"""
struct MCSample
    configs::Vector{Matrix{Float64}}
    kT::Vector{Float64}
    temperature::Vector{Float64}
    energy::Vector{Float64}
    acceptance::Vector{Float64}

    function MCSample(configs::Vector{Matrix{Float64}}, kT::Vector{Float64},
                      energy::Vector{Float64}, acceptance::Vector{Float64})
        (length(configs) == length(kT) == length(energy) == length(acceptance)) ||
            throw(DimensionMismatch(
                "configs/kT/energy/acceptance must be parallel; got " *
                "$(length(configs)) / $(length(kT)) / $(length(energy)) / " *
                "$(length(acceptance))"))
        return new(configs, kT, kT ./ KB_EV, energy, acceptance)
    end
end

Base.length(s::MCSample) = length(s.configs)
Base.getindex(s::MCSample, i) = s.configs[i]
Base.firstindex(s::MCSample) = 1
Base.lastindex(s::MCSample) = length(s.configs)
Base.eltype(::Type{MCSample}) = Matrix{Float64}
Base.iterate(s::MCSample, st::Int = 1) =
    st > length(s.configs) ? nothing : (s.configs[st], st + 1)
Base.show(io::IO, s::MCSample) = print(io, "MCSample(", length(s.configs), " configs)")

# --- the sweep kernel -------------------------------------------------------------

# Tabulate the full tesseral row Z_lm(e), l = 0:lmax, into `z` (ordered by
# `Harmonics.lm_index`, which is sequential in this loop order). `e` must be unit.
function _zlm_row!(z::Vector{Float64}, e::SVector{3,Float64}, lmax::Int)::Vector{Float64}
    i = 0
    @inbounds for l = 0:lmax, m = -l:l
        i += 1
        z[i] = Harmonics.Zlm_unsafe(l, m, e)
    end
    return z
end

# Accumulate one cluster term's coefficient-of-Z_lm(e_a) into `c`, contracting `folded`
# against the concrete neighbor rows `Z` — `_accumulate_term!` restricted to the single
# position of atom `a` (unique by the ctor invariant). Rank-specialized barrier, as in
# the mean-field kernel (`folded` is rank-erased `Array{Float64}`).
@inline function _accumulate_site_term!(c::Vector{Float64}, a::Int, coef::Float64,
                                        atoms::Vector{Int}, ls::Vector{Int},
                                        folded::Array{Float64,D},
                                        Z::Vector{Vector{Float64}}) where {D}
    i = findfirst(==(a), atoms)::Int
    @inbounds for index in CartesianIndices(folded)
        w = coef * folded[index]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            k == i && continue
            μk = index[k] - ls[k] - 1
            p *= Z[atoms[k]][Harmonics.lm_index(ls[k], μk)]
        end
        p == 0.0 && continue
        μi = index[i] - ls[i] - 1
        c[Harmonics.lm_index(ls[i], μi)] += w * p
    end
    return c
end

# One term's full contraction against the concrete rows `Z` (rank-specialized barrier).
@inline function _term_energy(coef::Float64, atoms::Vector{Int}, ls::Vector{Int},
                              folded::Array{Float64,D},
                              Z::Vector{Vector{Float64}})::Float64 where {D}
    E = 0.0
    @inbounds for index in CartesianIndices(folded)
        w = folded[index]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            μk = index[k] - ls[k] - 1
            p *= Z[atoms[k]][Harmonics.lm_index(ls[k], μk)]
        end
        E += w * p
    end
    return coef * E
end

# The SLCE energy of the configuration whose tesseral rows are `Z`: the sum of every
# term's contribution (the introspection contract makes the terms plain summands), i.e.
# `predict_energy(model, config) − j0`.
function _total_energy(terms::Vector{_MFATerm}, Z::Vector{Vector{Float64}})::Float64
    E = 0.0
    for term in terms
        E += _term_energy(term.coef, term.atoms, term.ls, term.folded, Z)
    end
    return E
end

# Energy of an arbitrary unit-column configuration (used for the `randomize`d copies,
# whose rows are not the chain's `Zcur`).
function _config_energy(s::MetropolisSampler, config::Matrix{Float64})::Float64
    nlm = (s.lmax + 1)^2
    Z = [_zlm_row!(zeros(nlm),
                   SVector{3,Float64}(config[1, a], config[2, a], config[3, a]), s.lmax)
         for a = 1:s.n_atoms]
    return _total_energy(s.terms, Z)
end

# One lattice sweep: `n_atoms` sequential single-spin attempts (deterministic site order —
# a composition of per-site reversible kernels, so the Boltzmann distribution stays
# stationary; sequential scan consumes no RNG for site selection and keeps runs
# bit-reproducible). Mutates `config` / `Zcur` in place; `c` / `Znew` are scratch.
# Returns the number of accepted moves.
function _mc_sweep!(rng::AbstractRNG, config::Matrix{Float64},
                    Zcur::Vector{Vector{Float64}}, s::MetropolisSampler, β::Float64,
                    step::Float64, c::Vector{Float64}, Znew::Vector{Float64})::Int
    nacc = 0
    for a = 1:s.n_atoms
        fill!(c, 0.0)
        for t in s.terms_of[a]
            term = s.terms[t]
            _accumulate_site_term!(c, a, term.coef, term.atoms, term.ls, term.folded, Zcur)
        end
        e = SVector{3,Float64}(config[1, a], config[2, a], config[3, a])
        # The engine's symmetric two-component proposal: antipodal flip (inter-lobe
        # ergodicity on bimodal single-site potentials) + Rodrigues rotation.
        e2 = if rand(rng) < _METROPOLIS_FLIP_FRACTION
            -e
        else
            # project back onto the sphere: compounded Rodrigues rotations would
            # otherwise random-walk the column norm off unity over very long chains
            # (~ε·√n_accepted); the flip branch is exact and needs no correction
            er = _rotate(e, _random_unit(rng), step * randn(rng))
            er / norm(er)
        end
        _zlm_row!(Znew, e2, s.lmax)
        Za = Zcur[a]
        ΔE = 0.0
        @inbounds for k in eachindex(c)
            ck = c[k]
            ck == 0.0 && continue
            ΔE += ck * (Znew[k] - Za[k])
        end
        if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
            config[1, a], config[2, a], config[3, a] = e2[1], e2[2], e2[3]
            copyto!(Za, Znew)
            nacc += 1
        end
    end
    return nacc
end

# --- the `sample` verb ---------------------------------------------------------------

# Resolve the chain start: an explicit `init` matrix, else the sampler's reference, else
# a uniform-random configuration drawn from `rng`.
function _mc_initial_config(s::MetropolisSampler,
                            init::Union{Nothing,AbstractMatrix{<:Real}},
                            rng::AbstractRNG)::Matrix{Float64}
    if init !== nothing
        config = _normalize_reference(init)
        size(config, 2) == s.n_atoms || throw(DimensionMismatch(
            "init has $(size(config, 2)) atoms but the sampler has $(s.n_atoms)"))
        return config
    end
    s.reference === nothing || return copy(s.reference)
    config = Matrix{Float64}(undef, 3, s.n_atoms)
    for a = 1:s.n_atoms
        u = _random_unit(rng)
        config[1, a], config[2, a], config[3, a] = u[1], u[2], u[3]
    end
    return config
end

# The kelvin / kT resolution is `SLCE.resolve_kt` (imported at the top of the package).
# It mirrors the mean-field `_resolve_taus` tau/m pattern: the two controls live under
# distinct names so a kelvin value can never be silently read as an energy.

"""
    sample(s::MetropolisSampler, n; temperature, kT, burnin = 200, thin = 10, step = 0.6,
           rng, init = nothing, randomize = false) -> MCSample
    sample(s::MetropolisSampler; temperature, kT, nsamples = 1, burnin = 200, thin = 10,
           step = 0.6, rng, init = nothing, randomize = false) -> MCSample

Draw spin configurations from the joint Boltzmann distribution of the fitted SLCE by
single-spin Metropolis. Provide **exactly one** absolute-temperature control:
`temperature` in **kelvin** (converted with `KB_EV` — assumes the model's energy
unit is eV, the package convention) or `kT` — `k_B·T` directly in the model's energy
units (theory/test runs, non-eV models). Every entry must be `> 0`.

The first form draws `n` configurations at a single temperature. The second sweeps a
**collection** (`temperature` or `kT`) and draws `nsamples` configurations per value,
ordered value-outer / sample-inner; the chain state **carries over** between consecutive
temperatures (with a fresh `burnin` at each), so a high→low ordering is an annealing
run — call once per temperature for independent chains.

# Keyword arguments
- `burnin::Integer = 200`: equilibration sweeps (one sweep = `n_atoms` attempts) before
  the first stored configuration at each temperature.
- `thin::Integer = 10`: sweeps between stored configurations (decorrelation).
- `step::Real = 0.6`: proposal rotation-angle scale in radians. Tune against the
  `acceptance` diagnostic (aim for O(0.2–0.6); lower `step` at low temperature).
- `rng::AbstractRNG = default_rng()`: explicit, seeded RNG (reproducible draws).
- `init = nothing`: chain start — an explicit `3 × n_atoms` matrix, else the sampler's
  `reference`, else a uniform-random configuration.
- `randomize::Bool = false`: apply one uniform random global rotation per **stored**
  configuration (the chain itself is not rotated). For an isotropic model the rotated
  configurations are still exact Boltzmann samples (the energy is invariant) while their
  absolute orientation — a slowly diffusing zero mode of the local updates — becomes
  exactly uniform, e.g. for later anisotropy training data. For an anisotropic model the
  rotation changes the energy, so the result is data augmentation, not an equilibrium
  sample.

Returns an [`MCSample`](@ref): `.configs` with parallel labels `.kT` (model energy
units), `.temperature` (kelvin), `.energy` (the stored — rotated, if `randomize` —
configuration's SLCE energy, `j0` excluded), and `.acceptance` (accept fraction over the
sweeps producing each configuration).
"""
function sample(s::MetropolisSampler, n::Integer; temperature = nothing, kT = nothing,
                burnin::Integer = 200, thin::Integer = 10, step::Real = 0.6,
                rng::AbstractRNG = default_rng(),
                init::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                randomize::Bool = false)::MCSample
    n >= 0 || throw(ArgumentError("n must be ≥ 0; got $n"))
    kts = resolve_kt(temperature, kT)
    length(kts) == 1 ||
        throw(ArgumentError("the positional `n` form takes a scalar `temperature`/`kT`; " *
                            "pass a collection without `n` to sweep"))
    return _mc_run(s, kts, n, burnin, thin, step, rng, init, randomize)
end

function sample(s::MetropolisSampler; temperature = nothing, kT = nothing,
                nsamples::Integer = 1,
                burnin::Integer = 200, thin::Integer = 10, step::Real = 0.6,
                rng::AbstractRNG = default_rng(),
                init::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                randomize::Bool = false)::MCSample
    nsamples >= 0 || throw(ArgumentError("nsamples must be ≥ 0; got $nsamples"))
    return _mc_run(s, resolve_kt(temperature, kT), nsamples, burnin, thin, step,
                   rng, init, randomize)
end

# Core driver: for each k_B·T value, `burnin` sweeps then `per` stored configurations
# `thin` sweeps apart, warm-starting consecutive temperatures from the running chain.
function _mc_run(s::MetropolisSampler, kts::Vector{Float64}, per::Integer,
                 burnin::Integer, thin::Integer, step::Real, rng::AbstractRNG,
                 init::Union{Nothing,AbstractMatrix{<:Real}}, randomize::Bool)::MCSample
    burnin >= 0 || throw(ArgumentError("burnin must be ≥ 0; got $burnin"))
    thin >= 1 || throw(ArgumentError("thin must be ≥ 1; got $thin"))
    step > 0 || throw(ArgumentError("step must be > 0; got $step"))
    stepf = Float64(step)

    config = _mc_initial_config(s, init, rng)
    nlm = (s.lmax + 1)^2
    Zcur = [_zlm_row!(zeros(nlm),
                      SVector{3,Float64}(config[1, a], config[2, a], config[3, a]),
                      s.lmax) for a = 1:s.n_atoms]
    c = zeros(nlm)
    Znew = zeros(nlm)

    total = length(kts) * per
    configs = Vector{Matrix{Float64}}(undef, total)
    kt_lab = Vector{Float64}(undef, total)
    energy = Vector{Float64}(undef, total)
    acceptance = Vector{Float64}(undef, total)
    k = 0
    for kt in kts
        β = 1.0 / kt
        nacc, natt = 0, 0
        for _ = 1:burnin
            nacc += _mc_sweep!(rng, config, Zcur, s, β, stepf, c, Znew)
            natt += s.n_atoms
        end
        for _ = 1:per
            for _ = 1:thin
                nacc += _mc_sweep!(rng, config, Zcur, s, β, stepf, c, Znew)
                natt += s.n_atoms
            end
            out = copy(config)
            E = _total_energy(s.terms, Zcur)
            if randomize
                out = Matrix(_random_rotation(rng) * out)
                E = _config_energy(s, out)   # keep energy[k] ↔ configs[k] consistent
            end
            k += 1
            configs[k] = out
            kt_lab[k] = kt
            energy[k] = E
            acceptance[k] = nacc / natt
            nacc, natt = 0, 0
        end
    end
    return MCSample(configs, kt_lab, energy, acceptance)
end
