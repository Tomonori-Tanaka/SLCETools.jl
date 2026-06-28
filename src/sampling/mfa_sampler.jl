# Mean-field sampler — P1/P2 (see `docs/specs/mfa-sampling.md`).
#
# Builds on the P0 single-site engine: each spin `a` is drawn from a von Mises–Fisher
# distribution `vMF(ê_a, κ_a)` about its reference direction, with a per-atom
# concentration set by the mean-field self-consistency in the reduced temperature
# `τ = T/T_MF`.
#
#   P1 — single global magnetization (no model): every atom shares one
#        `κ = 3m/τ` from the scalar self-consistency `m = L(3m/τ)`. Magesty-equivalent.
#   P2 — multi-sublattice isotropic exchange (from an `ExchangeModel`): the coupled
#        per-atom self-consistency `m_a = L(3(Ā m)_a/τ)` with the normalized molecular-
#        field matrix `Ā` (spectral radius 1), `κ_a = 3(Ā m)_a/τ`.
#
# `AbstractSampler` is the dispatch seam the later model-backed samplers (tensorial /
# higher-order MFA, P3+) slot into; `sample` is the one verb. The P2 coupling extraction
# and the `ExchangeModel` carrier live in `sampling/exchange.jl`.

"""
    AbstractSampler

Dispatch seam for spin-configuration samplers. [`MFASampler`](@ref) (mean-field) is the
first; future Metropolis-MC / spin-spiral samplers can slot in behind the `sample` verb.
"""
abstract type AbstractSampler end

# Numerical guards on the reduced temperature τ = T/T_MF, mirroring the reference
# sampler so the ordered / near-uniform boundaries reproduce it exactly.
const _MFA_MIN_TAU = 1.0e-5        # below this: fully ordered, return the reference
const _MFA_MAX_TAU = 0.99999       # above this: use a vanishing concentration
const _MFA_KAPPA_UNIFORM = 1.0e-6  # the near-uniform-limit concentration
# Bracket for the self-consistent magnetization root m ∈ (0, 1). A distinct physical
# quantity from the temperature guards above, so it carries its own name. The upper end
# sits at 1⁻ʳ so the near-saturated root for τ just above _MFA_MIN_TAU stays bracketed.
const _MFA_M_MIN = 1.0e-5
const _MFA_M_MAX = 1.0 - 1.0e-9

# The Langevin function L(κ) = coth κ − 1/κ = ⟨cosθ⟩ for a vMF field of concentration κ.
# `coth κ − 1/κ` cancels catastrophically as κ → 0, so a Maclaurin series is used there
# (the cone half-width regime near T_MF, where the coupled solve spends its iterations).
function _langevin(κ::Real)::Float64
    κf = Float64(κ)
    a = abs(κf)
    if a < 0.1
        return κf * (1 / 3 - κf^2 / 45 + 2 * κf^4 / 945)
    end
    return coth(κf) - 1 / κf
end

# Bisection root of a function that brackets a sign change on [lo, hi]; orientation-
# agnostic. Used for the monotone mean-field self-consistency (avoids a Roots dependency).
function _bisect(f, lo::Float64, hi::Float64; tol::Float64 = 1.0e-12, maxit::Int = 200)::Float64
    a, b = lo, hi
    fa = f(a)
    for _ = 1:maxit
        m = 0.5 * (a + b)
        fm = f(m)
        (abs(fm) <= tol || (b - a) <= tol) && return m
        if (fa < 0) == (fm < 0)
            a, fa = m, fm
        else
            b = m
        end
    end
    return 0.5 * (a + b)
end

"""
    thermal_averaged_m(τ) -> Float64

Solve the classical-Heisenberg mean-field self-consistency `m = L(3m/τ)` (equivalently
`m = coth(3m/τ) − τ/3m`) for the thermally averaged magnetization `m` at reduced
temperature `τ = T/T_MF`. Returns `1.0` for `τ < $(_MFA_MIN_TAU)` (fully ordered) and
`0.0` for `τ > $(_MFA_MAX_TAU)` (fully disordered). This is the single-sublattice case of
the coupled self-consistency [`MFASampler`](@ref) solves for `ExchangeModel` sources.
"""
function thermal_averaged_m(τ::Real)::Float64
    τf = Float64(τ)
    τf < _MFA_MIN_TAU && return 1.0
    τf > _MFA_MAX_TAU && return 0.0
    # f(m) = m − L(3m/τ) is increasing on (0,1): f(m_min) = m(1−1/τ) < 0 (for τ<1) and
    # f(m_max) = 1⁻ − L(3/τ) > 0 across the whole interior τ range.
    f(m) = m - _langevin(3m / τf)
    return _bisect(f, _MFA_M_MIN, _MFA_M_MAX)
end

"""
    tau_from_magnetization(m) -> Float64

Invert the mean-field self-consistency: the reduced temperature `τ = T/T_MF` whose
thermally averaged magnetization is `m`. Returns `1.0` for `m ≤ 0` and `0.0` for
`m ≥ 1`; magnetizations whose temperature falls outside `[$(_MFA_MIN_TAU),
$(_MFA_MAX_TAU)]` clamp to the disordered (`1.0`) or ordered (`0.0`) limit.
"""
function tau_from_magnetization(m::Real)::Float64
    mf = Float64(m)
    mf <= 0.0 && return 1.0
    mf >= 1.0 && return 0.0
    g(τ) = mf - _langevin(3mf / τ)               # increasing in τ
    g_lo = g(_MFA_MIN_TAU)
    g_hi = g(_MFA_MAX_TAU)
    if (g_lo < 0) == (g_hi < 0)                   # root outside the bracket: clamp
        return abs(g_lo) <= abs(g_hi) ? 0.0 : 1.0
    end
    return _bisect(g, _MFA_MIN_TAU, _MFA_MAX_TAU)
end

# Uniform random rotation in SO(3) via Shoemake's unit-quaternion construction. Used by
# the optional `randomize` global-frame randomization (a no-op on isotropic statistics,
# but it diversifies the realized frame for later anisotropic/SOC training).
function _random_rotation(rng::AbstractRNG)::SMatrix{3,3,Float64}
    u1, u2, u3 = rand(rng), rand(rng), rand(rng)
    x = sqrt(1 - u1) * sin(2π * u2)
    y = sqrt(1 - u1) * cos(2π * u2)
    z = sqrt(u1) * sin(2π * u3)
    w = sqrt(u1) * cos(2π * u3)
    return @SMatrix [
        1-2(y*y+z*z)  2(x*y-z*w)    2(x*z+y*w)
        2(x*y+z*w)    1-2(x*x+z*z)  2(y*z-x*w)
        2(x*z-y*w)    2(y*z+x*w)    1-2(x*x+y*y)
    ]
end

"""
    MFASampler(reference) <: AbstractSampler
    MFASampler(exch::ExchangeModel; reference)

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
  draw on the Bingham single-site potential (P3). See `sampling/exchange.jl`.

`exch` is the backing `ExchangeModel` (`nothing` for the single global sampler); `Abar` is
the normalized molecular-field matrix `Ā` (spectral radius 1), `rho` its Perron eigenvalue,
and `Tmf = ρ/3` the linearized mean-field `T_MF`.
"""
struct MFASampler <: AbstractSampler
    reference::Matrix{Float64}                 # 3 × n_atoms, unit columns
    exch::Union{Nothing,ExchangeModel}         # bilinear+single-ion backing (P2/P3); else nothing
    multipole::Union{Nothing,MultipoleField}   # full-multipole backing (P4); else nothing
    Abar::Matrix{Float64}                      # normalized Ā (ρ=1); 0×0 for the global sampler
    rho::Float64                               # Perron eigenvalue (1.0 for global)
    Tmf::Float64                               # linearized T_MF = ρ/3 (model units); 1.0 if global
end

# P1: single global, no couplings. Normalizes and validates the reference.
function MFASampler(reference::AbstractMatrix{<:Real})
    return MFASampler(_normalize_reference(reference), nothing, nothing, zeros(0, 0), 1.0, 1.0)
end

# P4: the full-multipole sampler, backed by a `MultipoleField` (all SCE clusters / l). The
# l=1 temperature scale ρ comes from the bilinear part; the draw is always Metropolis.
function MFASampler(mf::MultipoleField; reference::AbstractMatrix{<:Real})
    ref = _normalize_reference(reference)
    size(ref, 2) == mf.natoms || throw(DimensionMismatch(
        "reference has $(size(ref, 2)) atoms but the MultipoleField has $(mf.natoms)"))
    A = _mfa_matrix(mf.bilinear, ref)
    ρ, signdef = _perron(A)
    # lmax ≥ 1 is guaranteed here: ρ > 0 requires an l=1 bilinear channel (and every term
    # has some l ≥ 1), so the l=1 multipole slots used by `_l1_field` always exist.
    ρ > 1.0e-12 * (1 + maximum(abs.(A); init = 0.0)) || throw(ArgumentError(
        "the model has no l=1 (bilinear) ordering instability about this reference: the " *
        "bilinear molecular-field matrix has spectral radius ≈ 0 (no l=1 / Heisenberg " *
        "channel, or the reference is not its ordering mode), so the reduced-temperature " *
        "scale T_MF = ρ/3 is undefined. A purely biquadratic / anisotropic model needs an " *
        "l=1 exchange channel to set the temperature scale."))
    signdef || @warn "the leading molecular-field eigenvector is not sign-definite; the " *
        "reference may be frustrated or not the model's ordered ground state."
    _check_reference_stationary(mf.bilinear, ref)
    return MFASampler(ref, nothing, mf, A ./ ρ, ρ, ρ / 3)
end

# P2/P3: backed by an ExchangeModel. Builds the longitudinal molecular-field matrix about
# the reference, its Perron eigenvalue (T_MF = ρ/3), and checks the reference is a clean,
# stationary ordered state (warns otherwise — exact for collinear isotropic references, D2).
function MFASampler(exch::ExchangeModel; reference::AbstractMatrix{<:Real})
    ref = _normalize_reference(reference)
    size(ref, 2) == exch.natoms || throw(DimensionMismatch(
        "reference has $(size(ref, 2)) atoms but the ExchangeModel has $(exch.natoms)"))
    A = _mfa_matrix(exch, ref)
    ρ, signdef = _perron(A)
    ρ > 1.0e-12 * (1 + maximum(abs.(A); init = 0.0)) || throw(ArgumentError(
        "the exchange model has no ordering instability about this reference " *
        "(spectral radius ≈ 0); use the single global `MFASampler(reference)` instead"))
    signdef || @warn "the leading molecular-field eigenvector is not sign-definite; the " *
        "reference may be frustrated or not the model's ordered ground state — the " *
        "self-consistency still runs but T_MF/m_a(τ) may not reflect the intended order."
    _check_reference_stationary(exch, ref)
    return MFASampler(ref, exch, nothing, A ./ ρ, ρ, ρ / 3)
end

# Normalize a 3 × n_atoms reference matrix to unit columns, validating shape and norms.
function _normalize_reference(reference::AbstractMatrix{<:Real})::Matrix{Float64}
    size(reference, 1) == 3 ||
        throw(ArgumentError("reference must be 3 × n_atoms; got $(size(reference))"))
    n = size(reference, 2)
    n >= 1 || throw(ArgumentError("reference must have ≥ 1 atom"))
    ref = Matrix{Float64}(undef, 3, n)
    for a = 1:n
        v = SVector{3,Float64}(reference[1, a], reference[2, a], reference[3, a])
        nv = norm(v)
        nv > 1.0e-10 || throw(ArgumentError(
            "reference column $a has ~zero norm; cannot define a direction"))
        ref[:, a] = v / nv
    end
    return ref
end

_natoms(s::MFASampler)::Int = size(s.reference, 2)

# Does the sampler need the general Metropolis draw (a Bingham / higher-multipole single-site
# potential), or does the closed-form vMF suffice (single global, or isotropic exchange)?
_needs_metropolis(s::MFASampler)::Bool =
    s.multipole !== nothing || (s.exch !== nothing && !s.exch.isotropic)

function _kind(s::MFASampler)::String
    s.multipole !== nothing && return "multipole"
    s.exch === nothing && return "global"
    return s.exch.isotropic ? "isotropic" : "tensorial"
end

Base.show(io::IO, s::MFASampler) =
    print(io, "MFASampler(", _natoms(s), " atoms, ", _kind(s), ")")

# The per-atom single-site coefficient vectors (for the Metropolis draw) and magnetizations
# m_a at reduced temperature τ, for a sampler that needs the Metropolis path.
function _coeffs_and_m(s::MFASampler, τ::Float64)
    if s.multipole !== nothing
        return _multipole_state(s.multipole::MultipoleField, _ehat(s), s.rho, τ)
    end
    return _tensor_state(s.exch::ExchangeModel, _ehat(s), s.rho, τ)
end

"""
    mfa_temperature_scale(sampler) -> Float64

The mean-field ordering temperature `T_MF` in the sampler's energy units, so a caller can
convert `T = τ·T_MF` (decision D4). The single global [`MFASampler`](@ref) carries no
couplings and returns `T_MF = 1.0` (reduced units; `τ` is itself the temperature); an
`ExchangeModel`-backed sampler returns the linearized `T_MF = ρ(A)/3` from its coupling
spectrum (calibrated only up to the overall coupling scale — see decision D4).
"""
mfa_temperature_scale(s::MFASampler)::Float64 = s.Tmf

"""
    mfa_sublattice_m(sampler, τ) -> Vector{Float64}

The self-consistent per-atom magnetizations `m_a(τ)` at reduced temperature `τ`
(symmetry-equivalent atoms coincide). For the single global sampler every entry equals
`thermal_averaged_m(τ)`.
"""
function mfa_sublattice_m(s::MFASampler, τ::Real)::Vector{Float64}
    if _needs_metropolis(s)
        _, m = _coeffs_and_m(s, Float64(τ))
        return m
    end
    _, _, m = _mfa_state(s, Float64(τ))
    return m
end

# The reference directions as SVectors (the rigid cone axes).
_ehat(s::MFASampler)::Vector{SVector{3,Float64}} =
    [SVector{3,Float64}(s.reference[1, a], s.reference[2, a], s.reference[3, a])
     for a = 1:_natoms(s)]

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
end

Base.length(s::MFASample) = length(s.configs)
Base.getindex(s::MFASample, i) = s.configs[i]
Base.eltype(::Type{MFASample}) = Matrix{Float64}
Base.iterate(s::MFASample, st::Int = 1) =
    st > length(s.configs) ? nothing : (s.configs[st], st + 1)
Base.show(io::IO, s::MFASample) = print(io, "MFASample(", length(s.configs), " configs)")

# Validate 1-based atom indices against the atom count.
function _check_atom_indices(idx::AbstractVector{<:Integer}, n::Int, name::AbstractString)
    for i in idx
        (1 <= i <= n) || throw(ArgumentError("$name entry $i is outside 1:$n"))
    end
    return nothing
end

# The per-atom vMF draw state at reduced temperature τ: (ordered, κ::Vector, m::Vector).
# Single global ⇒ one κ broadcast to all atoms; isotropic exchange ⇒ the solved per-atom
# fields. (The tensorial path uses `_tensor_state` and the Metropolis draw instead.)
function _mfa_state(s::MFASampler, τ::Float64)
    n = _natoms(s)
    if s.exch === nothing
        if τ < _MFA_MIN_TAU
            return (true, fill(Inf, n), ones(n))
        elseif τ > _MFA_MAX_TAU
            return (false, fill(_MFA_KAPPA_UNIFORM, n), zeros(n))
        end
        m = thermal_averaged_m(τ)
        return (false, fill(3m / τ, n), fill(m, n))
    end
    return _coupled_state(s.Abar, τ, n)
end

# Draw one configuration with per-atom concentration `κ` (ordered ⇒ copy the reference).
# `uniform` columns are redrawn isotropically (overriding the vMF draw); a global rotation
# and the `fixed` columns are applied last, so `fixed` takes precedence over everything.
function _draw_config(rng::AbstractRNG, ref::Matrix{Float64}, ordered::Bool,
                      κ::Vector{Float64}, randomize::Bool,
                      fixed::AbstractVector{<:Integer},
                      uniform::AbstractVector{<:Integer})::Matrix{Float64}
    n = size(ref, 2)
    out = Matrix{Float64}(undef, 3, n)
    @inbounds for a = 1:n
        ea = SVector{3,Float64}(ref[1, a], ref[2, a], ref[3, a])
        v = ordered ? ea : sample_vmf(rng, ea, κ[a])
        out[1, a], out[2, a], out[3, a] = v[1], v[2], v[3]
    end
    for a in uniform
        u = _random_unit(rng)
        out[1, a], out[2, a], out[3, a] = u[1], u[2], u[3]
    end
    if randomize
        R = _random_rotation(rng)
        out = Matrix(R * out)
        for a in fixed
            out[:, a] = R * SVector{3,Float64}(ref[1, a], ref[2, a], ref[3, a])
        end
    else
        for a in fixed
            out[:, a] = SVector{3,Float64}(ref[1, a], ref[2, a], ref[3, a])
        end
    end
    return out
end

# Resolve exactly one of `tau` / `m` (scalars or collections) into a τ vector.
function _resolve_taus(tau, m)::Vector{Float64}
    (tau === nothing) == (m === nothing) &&
        throw(ArgumentError("provide exactly one of `tau` or `m`"))
    if tau !== nothing
        return tau isa Real ? [Float64(tau)] : Float64[Float64(t) for t in tau]
    end
    return m isa Real ? [tau_from_magnetization(m)] :
           Float64[tau_from_magnetization(mi) for mi in m]
end

"""
    sample(sampler::MFASampler, n; tau, m, rng, randomize, fixed, uniform) -> MFASample
    sample(sampler::MFASampler; tau, m, nsamples = 1, rng, randomize, fixed, uniform) -> MFASample

Draw spin configurations from the mean-field sampler. Provide **exactly one** control
variable: the reduced temperature `tau = T/T_MF` or the magnetization `m` (the latter is
only meaningful for the single global sampler, where it maps to a `τ`). Each spin is drawn
from `vMF(ê_a, κ_a)` about its reference direction with the self-consistent per-atom
concentration.

The first form draws `n` configurations at a single control value. The second form sweeps
a **collection** `tau` (or `m`) and draws `nsamples` configurations per value, ordered
value-outer / sample-inner.

# Keyword arguments
- `rng::AbstractRNG = default_rng()`: explicit, seeded RNG (reproducible draws).
- `randomize::Bool = false`: apply one uniform random global rotation per configuration.
- `fixed::AbstractVector{<:Integer} = Int[]`: atoms held at the reference direction
  (rotated with the frame when `randomize`); takes precedence over `uniform`.
- `uniform::AbstractVector{<:Integer} = Int[]`: atoms redrawn isotropically (the
  disordered limit) regardless of the control value.

Returns an [`MFASample`](@ref): `.configs` (each `3 × n_atoms` unit directions) with
parallel labels `.tau` and `.m` (per-atom magnetization).
"""
function sample(sampler::MFASampler, n::Integer; tau = nothing, m = nothing,
                rng::AbstractRNG = default_rng(), randomize::Bool = false,
                fixed::AbstractVector{<:Integer} = Int[],
                uniform::AbstractVector{<:Integer} = Int[])::MFASample
    n >= 0 || throw(ArgumentError("n must be ≥ 0; got $n"))
    taus = _resolve_taus(tau, m)
    length(taus) == 1 ||
        throw(ArgumentError("the positional `n` form takes a scalar `tau`/`m`; " *
                            "pass a collection without `n` to sweep"))
    return _sample_sweep(sampler, taus, n, rng, randomize, fixed, uniform)
end

function sample(sampler::MFASampler; tau = nothing, m = nothing, nsamples::Integer = 1,
                rng::AbstractRNG = default_rng(), randomize::Bool = false,
                fixed::AbstractVector{<:Integer} = Int[],
                uniform::AbstractVector{<:Integer} = Int[])::MFASample
    nsamples >= 0 || throw(ArgumentError("nsamples must be ≥ 0; got $nsamples"))
    taus = _resolve_taus(tau, m)
    return _sample_sweep(sampler, taus, nsamples, rng, randomize, fixed, uniform)
end

# Core sweep: for each τ draw `per` configurations, value-outer / sample-inner. Dispatches
# to the closed-form vMF draw (global / isotropic) or the Metropolis draw (tensorial).
function _sample_sweep(sampler::MFASampler, taus::Vector{Float64}, per::Integer,
                       rng::AbstractRNG, randomize::Bool,
                       fixed::AbstractVector{<:Integer},
                       uniform::AbstractVector{<:Integer})::MFASample
    natoms = size(sampler.reference, 2)
    _check_atom_indices(fixed, natoms, "fixed")
    _check_atom_indices(uniform, natoms, "uniform")
    _needs_metropolis(sampler) &&
        return _sweep_metropolis(sampler, taus, per, rng, randomize, fixed, uniform)
    ref = sampler.reference
    total = length(taus) * per
    configs = Vector{Matrix{Float64}}(undef, total)
    tau_lab = Vector{Float64}(undef, total)
    m_lab = Vector{Vector{Float64}}(undef, total)
    k = 0
    for τ in taus
        ordered, κ, mval = _mfa_state(sampler, τ)
        for _ = 1:per
            k += 1
            configs[k] = _draw_config(rng, ref, ordered, κ, randomize, fixed, uniform)
            tau_lab[k] = τ
            m_lab[k] = mval
        end
    end
    return MFASample(configs, tau_lab, m_lab)
end

# Metropolis sweep for the tensorial (P3) / full-multipole (P4) sampler. Per τ, solve the
# single-site coefficients `cs` and run one Metropolis chain per (non-fixed, non-uniform)
# atom — each chain initialized at the reference axis ê_a — to produce `per` decorrelated
# draws; assemble the configurations, applying `uniform` / `fixed` / `randomize` as in the
# vMF path (the global rotation is applied to the whole configuration).
function _sweep_metropolis(sampler::MFASampler, taus::Vector{Float64}, per::Integer,
                           rng::AbstractRNG, randomize::Bool,
                           fixed::AbstractVector{<:Integer},
                           uniform::AbstractVector{<:Integer})::MFASample
    ref = sampler.reference
    n = size(ref, 2)
    ehat = _ehat(sampler)
    isfixed = falses(n); isuniform = falses(n)
    for a in fixed; isfixed[a] = true; end
    for a in uniform; isuniform[a] = true; end
    total = length(taus) * per
    configs = Vector{Matrix{Float64}}(undef, total)
    tau_lab = Vector{Float64}(undef, total)
    m_lab = Vector{Vector{Float64}}(undef, total)
    k = 0
    for τ in taus
        cs, mval = _coeffs_and_m(sampler, τ)
        chains = Vector{Vector{SVector{3,Float64}}}(undef, n)
        @inbounds for a = 1:n
            if isfixed[a] || isuniform[a]
                chains[a] = SVector{3,Float64}[]
                continue
            end
            # scale the proposal to the peak sharpness (∝ 1/√concentration): a sharp Bingham
            # at low τ needs small steps to mix, a broad one near T_MF needs large steps.
            step = clamp(1.5 / sqrt(1 + _field_scale(cs[a])), 0.05, 0.8)
            chains[a] = sample_site_metropolis(rng, cs[a], per; e_init = ehat[a],
                                               step = step, nburn = 300, thin = 15)
        end
        for s = 1:per
            out = Matrix{Float64}(undef, 3, n)
            @inbounds for a = 1:n
                v = isfixed[a] ? ehat[a] : isuniform[a] ? _random_unit(rng) : chains[a][s]
                out[1, a], out[2, a], out[3, a] = v[1], v[2], v[3]
            end
            randomize && (out = Matrix(_random_rotation(rng) * out))
            k += 1
            configs[k] = out
            tau_lab[k] = τ
            m_lab[k] = mval
        end
    end
    return MFASample(configs, tau_lab, m_lab)
end
