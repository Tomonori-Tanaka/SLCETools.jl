# Mean-field sampler — the coupling / sampler types (see `docs/specs/mfa-sampling.md`).
#
# Pure type definitions, their invariant-enforcing inner constructors, and `Base.show`. The
# algorithms live alongside: `exchange.jl` (ExchangeModel construction + the longitudinal
# molecular-field analysis), `selfconsistency.jl` (the mean-field solvers), `sampler.jl`
# (the MFASampler constructors + the `sample` verb), `bridge.jl` (from a fitted SCE).

# The 3×3 identity, used by the ExchangeModel classification / isotropic builder.
const _I3 = SMatrix{3,3,Float64}(I)

"""
    AbstractSampler

Dispatch seam for spin-configuration samplers: [`MFASampler`](@ref) (mean-field,
single-site) and [`MetropolisSampler`](@ref) (Metropolis MC, joint Boltzmann); future
spin-spiral samplers can slot in behind the `sample` verb.

The interface a subtype implements: `sample(s, n; <control>, rng, ...)` returning a
labeled result ([`MFASample`](@ref) / [`MCSample`](@ref) — configurations in the
`3 × n_atoms` unit-column layout with parallel per-config labels). The control variable
is sampler-specific (the reduced `tau`/`m` for the mean field, the absolute
`temperature` [K] / `kT` [energy units] for the MC), as is, where meaningful,
`mfa_temperature_scale(s)`.
"""
abstract type AbstractSampler end

# One mean-field interaction term: `coef = jϕ·(4π)^(N/2)`, the member's `atoms` and per-site
# `ls`, and the rank-N real coefficient tensor `folded`. The mean-field single-site potential
# on atom a gathers, over every term and every position i with `atoms[i] = a`, the
# coefficient of `Z_{ls[i],m}(e_a)` obtained by contracting `folded` against the *other*
# sites' multipole averages — the `accumulate_grad!` leave-one-out structure with the site-a
# harmonic left symbolic instead of differentiated.
struct _MFATerm
    coef::Float64
    atoms::Vector{Int}
    ls::Vector{Int}
    folded::Array{Float64}
end

"""
    ExchangeModel(Jiso; onsite = nothing)
    ExchangeModel(bilinear; onsite = nothing)
    ExchangeModel(model::SCEPredictor)

Neutral carrier of the bilinear exchange (and single-ion anisotropy) the mean-field
sampler needs.

- `ExchangeModel(Jiso::AbstractMatrix{<:Real}; onsite)` — isotropic exchange: `Jiso[a,b]`
  is the symmetric total Heisenberg coupling `Σ_R J_iso(a,b,R)` (sign convention of the SCE
  energy `E = Σ_{bonds} J e_a·e_b`). `onsite`, if given, is a length-`n` vector of `3×3`
  single-ion matrices `A_a` (`e' A_a e`).
- `ExchangeModel(bilinear::AbstractMatrix{<:SMatrix{3,3}}; onsite)` — full tensorial
  exchange: `bilinear[a,b] = S_ab` with `bilinear[b,a] ≈ S_ab'` (the field is
  `g_a = Σ_b S_ab ⟨e_b⟩`); the symmetric/antisymmetric parts carry anisotropic / DM
  exchange.
- `ExchangeModel(model)` — extract `bilinear` and `onsite` from a fitted `SCEPredictor`
  by reusing the core's bilinear (`ls=[1,1]`) and single-ion (`ls=[2]`) extraction; only the
  higher-order / higher-`l` channels are dropped (a P4 extension) and reported via `@warn`.
"""
struct ExchangeModel
    natoms::Int
    Jiso::Matrix{Float64}                              # tr(S)/3, the isotropic part
    bilinear::Matrix{SMatrix{3,3,Float64,9}}           # S_ab: field g_a = Σ_b S_ab ⟨e_b⟩
    onsite::Vector{SMatrix{3,3,Float64,9}}             # single-ion A_a (zero ⇒ none)
    isotropic::Bool                                    # all S_ab ∝ I and onsite all zero

    # Inner constructor (the only way to build an ExchangeModel): validate shapes and the
    # energy symmetry S_ba = S_ab', symmetrize exactly, and classify isotropic vs tensorial.
    # The convenience outer constructors (Jiso / bilinear keyword forms, and from a fitted
    # model) all route through here, so the invariants cannot be bypassed.
    function ExchangeModel(bilinear::Matrix{SMatrix{3,3,Float64,9}},
                           onsite::Vector{SMatrix{3,3,Float64,9}})
        n = size(bilinear, 1)
        size(bilinear, 2) == n ||
            throw(ArgumentError("bilinear must be square; got $(size(bilinear))"))
        length(onsite) == n ||
            throw(ArgumentError("onsite must have one matrix per atom ($n); got $(length(onsite))"))
        bil = Matrix{SMatrix{3,3,Float64,9}}(undef, n, n)
        scale = 0.0
        @inbounds for a = 1:n, b = 1:n
            scale = max(scale, maximum(abs.(bilinear[a, b])))
        end
        @inbounds for a = 1:n, b = 1:n
            maximum(abs.(bilinear[b, a] - bilinear[a, b]')) <= 1.0e-9 * (1 + scale) ||
                throw(ArgumentError("bilinear must satisfy bilinear[b,a] = bilinear[a,b]' " *
                                    "(a real bilinear energy); violated at ($a,$b)"))
            bil[a, b] = 0.5 * (bilinear[a, b] + bilinear[b, a]')   # symmetrize the energy exactly
        end
        Jiso = Matrix{Float64}(undef, n, n)
        iso = true
        @inbounds for a = 1:n, b = 1:n
            j = (bil[a, b][1, 1] + bil[a, b][2, 2] + bil[a, b][3, 3]) / 3
            Jiso[a, b] = j
            maximum(abs.(bil[a, b] - j * _I3)) <= 1.0e-9 * (1 + scale) || (iso = false)
        end
        osc = 0.0
        @inbounds for a = 1:n
            osc = max(osc, maximum(abs.(onsite[a])))
        end
        osc > 1.0e-12 * (1 + scale) && (iso = false)
        return new(n, Jiso, bil, onsite, iso)
    end
end

Base.show(io::IO, m::ExchangeModel) =
    print(io, "ExchangeModel(", m.natoms, " atoms", m.isotropic ? ", isotropic" : ", tensorial", ")")

"""
    MultipoleModel

The digested full-multipole mean field of a fitted SCE (P4): every cluster term
(`_MFATerm`), the `lmax`, and the bilinear [`ExchangeModel`](@ref) (used only for the
`l=1` temperature scale `ρ`). Built by `MultipoleModel(model::SCEPredictor)`; consumed by the
[`MFASampler`](@ref) tensorial/Metropolis path.

Renamed from `MultipoleField` (it is a coupling *model*, the full-fidelity sibling of
[`ExchangeModel`](@ref), not a field).
"""
struct MultipoleModel
    natoms::Int
    lmax::Int
    terms::Vector{_MFATerm}
    bilinear::ExchangeModel

    # Inner constructor: enforce the structural invariants the mean-field solver relies on.
    function MultipoleModel(natoms::Int, lmax::Int, terms::Vector{_MFATerm},
                            bilinear::ExchangeModel)
        isempty(terms) && throw(ArgumentError("MultipoleModel needs at least one term"))
        bilinear.natoms == natoms || throw(DimensionMismatch(
            "bilinear ExchangeModel has $(bilinear.natoms) atoms but the model has $natoms"))
        maxl = maximum(maximum(t.ls) for t in terms)
        lmax >= maxl || throw(ArgumentError("lmax=$lmax does not cover the terms' max l=$maxl"))
        @inbounds for t in terms, a in t.atoms
            1 <= a <= natoms ||
                throw(ArgumentError("term atom index $a outside 1:$natoms"))
        end
        return new(natoms, lmax, terms, bilinear)
    end
end

Base.show(io::IO, mf::MultipoleModel) =
    print(io, "MultipoleModel(", mf.natoms, " atoms, lmax=", mf.lmax, ", ",
          length(mf.terms), " terms)")

"""
    MFASampler(reference) <: AbstractSampler
    MFASampler(exch::ExchangeModel; reference)
    MFASampler(model::SCEPredictor; reference)

Mean-field spin-configuration sampler. Every spin is drawn from `vMF(ê_a, κ_a)` about its
reference direction; the per-atom concentration is set by the mean-field self-consistency
at the reduced temperature `τ` (or magnetization `m`), via [`sample`](@ref).

- `MFASampler(reference)` — the single global, isotropic sampler (P1): `reference` is a
  `3 × n_atoms` matrix of seed directions (columns normalized on construction), and all
  atoms share one concentration `κ = 3m/τ`, `m = L(3m/τ)`. No couplings; works in reduced
  units (`mfa_temperature_scale` returns `1.0`).
- `MFASampler(exch; reference)` — the [`ExchangeModel`](@ref)-backed sampler (P2/P3): the
  per-atom magnetizations `m_a(τ)` are solved from the coupled mean-field self-consistency
  about the given `reference` state, so distinct sublattices disorder at distinct rates
  (a single `T_MF`). For purely isotropic exchange the draw is the closed-form vMF (P2);
  with DMI / anisotropic exchange or single-ion anisotropy it is the general Metropolis
  draw on the Bingham single-site potential (P3).
- `MFASampler(model; reference)` — the full-multipole sampler (P4), backed by a
  [`MultipoleModel`](@ref) keeping every SCE channel.

The backing `source` is the coupling model the sampler draws from: `nothing` for the single
global sampler, an [`ExchangeModel`](@ref) for the bilinear/single-ion path (P2/P3), or a
[`MultipoleModel`](@ref) for the full-multipole path (P4); the sampler is parametric on its
type so dispatch is type-stable. `Abar` is the normalized molecular-field matrix `Ā` (spectral
radius 1), `rho` its Perron eigenvalue, and `Tmf = ρ/3` the linearized mean-field `T_MF`.
"""
struct MFASampler{S} <: AbstractSampler
    reference::Matrix{Float64}                 # 3 × n_atoms, unit columns
    source::S                                  # Nothing | ExchangeModel | MultipoleModel
    Abar::Matrix{Float64}                      # normalized Ā (ρ=1); 0×0 for the global sampler
    rho::Float64                               # Perron eigenvalue (1.0 for global)
    Tmf::Float64                               # linearized T_MF = ρ/3 (model units); 1.0 if global

    # Inner constructor: enforce the source ↔ coupling-matrix invariant (a global sampler
    # carries no Ā / ρ; a model-backed one carries an n×n Ā and ρ > 0).
    function MFASampler(reference::Matrix{Float64}, source::S, Abar::Matrix{Float64},
                        rho::Float64, Tmf::Float64) where {S}
        n = size(reference, 2)
        if source === nothing
            size(Abar) == (0, 0) ||
                throw(ArgumentError("the global sampler carries no molecular-field matrix"))
        else
            size(Abar) == (n, n) || throw(DimensionMismatch(
                "Ā is $(size(Abar)) but the reference has $n atoms"))
            rho > 0 || throw(ArgumentError("the Perron eigenvalue ρ must be positive; got $rho"))
        end
        return new{S}(reference, source, Abar, rho, Tmf)
    end
end

"""
    MFASample

Labeled result of [`sample`](@ref) (decision D1). `configs` is the bare
`Vector{Matrix{Float64}}` (each `3 × n_atoms` unit directions); the parallel `tau` and `m`
hold each config's reduced temperature and per-atom magnetization vector. The object is
iterable and indexable as its `configs`.
"""
struct MFASample
    configs::Vector{Matrix{Float64}}
    tau::Vector{Float64}
    m::Vector{Vector{Float64}}

    function MFASample(configs::Vector{Matrix{Float64}}, tau::Vector{Float64},
                       m::Vector{Vector{Float64}})
        (length(configs) == length(tau) == length(m)) || throw(DimensionMismatch(
            "configs/tau/m must be parallel; got $(length(configs)) / $(length(tau)) / $(length(m))"))
        return new(configs, tau, m)
    end
end

Base.length(s::MFASample) = length(s.configs)
Base.getindex(s::MFASample, i) = s.configs[i]
Base.firstindex(s::MFASample) = 1
Base.lastindex(s::MFASample) = length(s.configs)
Base.eltype(::Type{MFASample}) = Matrix{Float64}
Base.iterate(s::MFASample, st::Int = 1) =
    st > length(s.configs) ? nothing : (s.configs[st], st + 1)
Base.show(io::IO, s::MFASample) = print(io, "MFASample(", length(s.configs), " configs)")
