# Mean-field sampler — the `MFASampler` constructors and the `sample` verb (see
# `docs/specs/mfa-sampling.md`). The sampler types live in `types.jl`; the self-consistency
# solvers in `selfconsistency.jl`; the single-site draws in the `MeanFieldEngine` submodule.
#
# Each spin `a` is drawn from a von Mises–Fisher distribution `vMF(ê_a, κ_a)` about its
# reference direction (or the general Metropolis draw for a Bingham / higher-multipole
# single-site potential), with a per-atom concentration set by the mean-field
# self-consistency in the reduced temperature `τ = T/T_MF`.

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

# P1: single global, no couplings. Normalizes and validates the reference.
function MFASampler(reference::AbstractMatrix{<:Real})
    return MFASampler(_normalize_reference(reference), nothing, zeros(0, 0), 1.0, 1.0)
end

# P4: the full-multipole sampler, backed by a `MultipoleModel` (all SLCE clusters / l). The
# l=1 temperature scale ρ comes from the bilinear part; the draw is always Metropolis.
function MFASampler(mf::MultipoleModel; reference::AbstractMatrix{<:Real})
    ref = _normalize_reference(reference)
    size(ref, 2) == mf.n_atoms || throw(DimensionMismatch(
        "reference has $(size(ref, 2)) atoms but the MultipoleModel has $(mf.n_atoms)"))
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
    return MFASampler(ref, mf, A ./ ρ, ρ, ρ / 3)
end

# P2/P3: backed by an ExchangeModel. Builds the longitudinal molecular-field matrix about
# the reference, its Perron eigenvalue (T_MF = ρ/3), and checks the reference is a clean,
# stationary ordered state (warns otherwise — exact for collinear isotropic references, D2).
function MFASampler(exch::ExchangeModel; reference::AbstractMatrix{<:Real})
    ref = _normalize_reference(reference)
    size(ref, 2) == exch.n_atoms || throw(DimensionMismatch(
        "reference has $(size(ref, 2)) atoms but the ExchangeModel has $(exch.n_atoms)"))
    A = _mfa_matrix(exch, ref)
    ρ, signdef = _perron(A)
    ρ > 1.0e-12 * (1 + maximum(abs.(A); init = 0.0)) || throw(ArgumentError(
        "the exchange model has no ordering instability about this reference " *
        "(spectral radius ≈ 0); use the single global `MFASampler(reference)` instead"))
    signdef || @warn "the leading molecular-field eigenvector is not sign-definite; the " *
        "reference may be frustrated or not the model's ordered ground state — the " *
        "self-consistency still runs but T_MF/m_a(τ) may not reflect the intended order."
    _check_reference_stationary(exch, ref)
    return MFASampler(ref, exch, A ./ ρ, ρ, ρ / 3)
end

_natoms(s::MFASampler)::Int = size(s.reference, 2)

# Does the sampler need the general Metropolis draw (a Bingham / higher-multipole single-site
# potential), or does the closed-form vMF suffice (single global, or isotropic exchange)?
_needs_metropolis(::MFASampler{Nothing})::Bool = false
_needs_metropolis(s::MFASampler{ExchangeModel})::Bool = !s.source.isotropic
_needs_metropolis(::MFASampler{MultipoleModel})::Bool = true

_kind(::MFASampler{Nothing})::String = "global"
_kind(s::MFASampler{ExchangeModel})::String = s.source.isotropic ? "isotropic" : "tensorial"
_kind(::MFASampler{MultipoleModel})::String = "multipole"

Base.show(io::IO, s::MFASampler) =
    print(io, "MFASampler(", _natoms(s), " atoms, ", _kind(s), ")")

# The per-atom single-site coefficient vectors (for the Metropolis draw) and magnetizations
# m_a at reduced temperature τ, for a sampler that needs the Metropolis path.
_coeffs_and_m(s::MFASampler{MultipoleModel}, τ::Float64) =
    _multipole_state(s.source, _ehat(s), s.rho, τ)
_coeffs_and_m(s::MFASampler{ExchangeModel}, τ::Float64) =
    _tensor_state(s.source, _ehat(s), s.rho, τ)

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
    τ >= 0 || throw(ArgumentError("the reduced temperature τ must be ≥ 0; got $τ"))
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
function _mfa_state(s::MFASampler{Nothing}, τ::Float64)
    n = _natoms(s)
    if τ < _MFA_MIN_TAU
        return (true, fill(Inf, n), ones(n))
    elseif τ > _MFA_MAX_TAU
        return (false, fill(_MFA_KAPPA_UNIFORM, n), zeros(n))
    end
    m = thermal_averaged_m(τ)
    return (false, fill(3m / τ, n), fill(m, n))
end

# Isotropic ExchangeModel (the only model-backed sampler that reaches the closed-form vMF
# path; the tensorial / multipole sources go through _coeffs_and_m + Metropolis instead).
_mfa_state(s::MFASampler{ExchangeModel}, τ::Float64) = _coupled_state(s.Abar, τ, _natoms(s))

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
    taus = if tau !== nothing
        tau isa Real ? [Float64(tau)] : Float64[Float64(t) for t in tau]
    else
        m isa Real ? [tau_from_magnetization(m)] :
            Float64[tau_from_magnetization(mi) for mi in m]
    end
    # Reject negative τ loudly — it would otherwise fall into the τ < _MFA_MIN_TAU branch
    # and silently alias the fully ordered limit.
    all(>=(0.0), taus) ||
        throw(ArgumentError("the reduced temperature τ must be ≥ 0; got $(minimum(taus))"))
    return taus
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
    nat = size(sampler.reference, 2)      # a local, not the `n_atoms` generic
    _check_atom_indices(fixed, nat, "fixed")
    _check_atom_indices(uniform, nat, "uniform")
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
            m_lab[k] = copy(mval)   # one m vector per config (no aliasing across co-τ draws)
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
            step = clamp(1.5 / sqrt(1 + field_scale(cs[a])), 0.05, 0.8)
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
            m_lab[k] = copy(mval)   # one m vector per config (no aliasing across co-τ draws)
        end
    end
    return MFASample(configs, tau_lab, m_lab)
end
