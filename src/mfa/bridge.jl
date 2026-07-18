# Mean-field sampler — extracting an `ExchangeModel` / `MultipoleModel` from a fitted SCE
# (see `docs/specs/mfa-sampling.md`). Reads the fitted Hamiltonian only through the core's
# public introspection surface (`SCEFitting.bilinear_terms` / `multipole_terms`), never
# the SALC-basis internals — so the basis representation can evolve independently.

# Build the directed bilinear tensor `bilinear[a,b] = S_ab` and single-ion `onsite[a] = A_a`
# from the core's public bilinear extraction (no warnings; callers decide what to report).
# Returns `(bilinear, onsite, nselfbond, nskipped)`; `nselfbond` counts on-site
# (a == image-of-a) bilinear terms, which the rigid-axis mean field does not represent (only
# reachable via AllImages); `nskipped` counts the higher-order / higher-`l` SALCs that are
# not bilinear (kept instead by the full `MultipoleModel` path).
# NOTE(asymmetry): the same repeated-atom cluster makes `MultipoleModel` /
# `MetropolisSampler` hard-error (`allunique` invariant in `_scaled_multipole_terms`)
# where this path skips with a warning. Unreachable today (MinimumImage never emits
# repeated-atom clusters); if AllImages lands upstream, decide then whether the
# multipole path should degrade gracefully like this one.
function _extract_bilinear_onsite(model::SCEPredictor)
    terms = bilinear_terms(model)
    n = n_atoms(model)
    bilinear = fill(zero(SMatrix{3,3,Float64,9}), n, n)
    nselfbond = 0
    for ((a, b, _), M) in terms.pairs
        if a == b
            nselfbond += 1
            continue
        end
        bilinear[a, b] += M
        bilinear[b, a] += SMatrix{3,3,Float64}(transpose(M))
    end
    onsite = fill(zero(SMatrix{3,3,Float64,9}), n)
    for (a, A) in terms.onsites
        onsite[a] += A
    end
    return bilinear, onsite, nselfbond, length(terms.skipped)
end

"""
    ExchangeModel(model::SCEPredictor) -> ExchangeModel

Extract the full bilinear exchange (`ls=[1,1]`: Heisenberg + DMI + anisotropic) and the
single-ion anisotropy (`ls=[2]`) of a fitted `SCEPredictor` into an [`ExchangeModel`](@ref),
via the core's public `bilinear_terms` extraction. The bond matrices are placed
directionally (`bilinear[a,b] = S_ab`, the reverse member transposed), so the molecular
field is `g_a = Σ_b S_ab ⟨e_b⟩`. Only the higher-order / higher-`l` SALCs (3-body and up)
are dropped — captured instead by the full [`MultipoleModel`](@ref) path — and reported via
`@warn`.
"""
function ExchangeModel(model::SCEPredictor)
    bilinear, onsite, nselfbond, nskipped = _extract_bilinear_onsite(model)
    nskipped > 0 && @warn "ExchangeModel keeps the bilinear (Heisenberg + DMI + anisotropic) " *
        "and single-ion channels; dropped $nskipped higher-order / higher-l SALC(s) " *
        "(use the full SCE `MFASampler(model; reference)` to keep them)."
    nselfbond > 0 && @warn "ExchangeModel: skipping $nselfbond on-site (a == image-of-a) " *
        "bilinear term(s); the rigid-axis mean field does not represent them (only reachable " *
        "via AllImages). The default MinimumImage selection drops such self-pairs."
    return ExchangeModel(bilinear; onsite = onsite)
end

# Digest the fitted model's multipole terms into `_MFATerm`s, applying the `(4π)^(body/2)`
# tesseral scale (`multipole_terms` returns the raw fitted `jϕ`). This is the package's
# single scale-application site — both the mean-field `MultipoleModel` and the Metropolis
# `MetropolisSampler` consume it; never re-apply downstream. Returns `(terms, lmax)`.
function _scaled_multipole_terms(model::SCEPredictor)
    terms = _MFATerm[]
    lmax = 0
    for mt in multipole_terms(model)
        # Both consumers require distinct sites per cluster: the mean-field contraction
        # decouples every site (⟨∏Z⟩ → ∏⟨Z⟩), and the Metropolis local update relies on
        # the site-a coefficients being independent of e_a. The cluster enumeration drops
        # reused-atom clusters, so this holds, but assert it (a repeated site would need
        # CG recoupling, not a self-⟨Z⟩ factor).
        allunique(mt.atoms) || throw(ArgumentError(
            "cluster member with a repeated atom $(mt.atoms); the per-site " *
            "factorization assumes distinct sites"))
        # Copy the fields out of the core's introspection view: never alias the fitted model's
        # internal SALC arrays into a long-lived term (value semantics, no upstream mutation).
        push!(terms, _MFATerm(mt.coef * (4π)^(mt.body / 2), copy(mt.atoms), copy(mt.ls),
                              copy(mt.folded)))
        lmax = max(lmax, maximum(mt.ls))
    end
    isempty(terms) && throw(ArgumentError(
        "the model has no spin-dependent SALCs with a nonzero coefficient"))
    return terms, lmax
end

"""
    MultipoleModel(model::SCEPredictor) -> MultipoleModel

Digest a fitted `SCEPredictor` into the full-multipole mean field (P4): one mean-field term per
cluster member / `l`-ordering (carrying `jϕ·(4π)^(N/2)`, the member atoms, the per-site
`ls`, and the folded coefficient tensor), the model `lmax`, and the bilinear
[`ExchangeModel`](@ref) used only for the `l=1` temperature scale. Reads the terms through
the core's public `multipole_terms` view, so it keeps **all** channels — bilinear,
single-ion, and higher-order / many-body — and the mean field iterates the full multipole
averages `⟨Z_lm⟩`.
"""
function MultipoleModel(model::SCEPredictor)
    terms, lmax = _scaled_multipole_terms(model)
    bilinear, onsite, _, _ = _extract_bilinear_onsite(model)
    return MultipoleModel(n_atoms(model), lmax, terms, ExchangeModel(bilinear; onsite = onsite))
end

"""
    MFASampler(model::SCEPredictor; reference) -> MFASampler

The full-multipole mean-field sampler (P4): build a [`MultipoleModel`](@ref) from the fitted
SCE (keeping every channel — bilinear, single-ion, and higher-order / many-body) and sample
about `reference`. The `l=1` temperature scale `T_MF = ρ/3` comes from the bilinear part;
the single-site distribution (a Bingham / higher-multipole shape) is drawn with the
Metropolis engine. See [`MFASampler(::ExchangeModel)`](@ref) for the bilinear-only path.
"""
MFASampler(model::SCEPredictor; reference::AbstractMatrix{<:Real}) =
    MFASampler(MultipoleModel(model); reference = reference)
