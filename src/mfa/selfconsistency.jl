# Mean-field sampler — the reduced-temperature self-consistency (see `docs/specs/mfa-sampling.md`).
#
# Solves the per-atom magnetizations m_a(τ) (and the single-site coefficient vectors for the
# Metropolis draw) at reduced temperature τ = T/T_MF, in three fidelities:
#   - isotropic exchange (l=1-only):   m_a = L(3(Ā m)_a/τ)        — `_coupled_state` (closed-form vMF)
#   - tensorial / single-ion (Bingham): m_a = ⟨e·ê_a⟩ by quadrature — `_tensor_state`
#   - full multipole / many-body:       iterate ⟨Z_lm⟩_a            — `_multipole_state`
# All three drive `_anderson_solve` (depth-1 Anderson, for the critical slowing as τ → 1⁻).

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

# --- the coupled self-consistency (Anderson-accelerated) ---------------------------

# Solve m = G(m) by depth-1 Anderson acceleration from m = m0 (selects the stable ordered
# branch over the trivial solution), clamping iterates to [mlo, mhi]. `G!(out, m)` writes
# the self-consistency map. Anderson (not damped Picard) because the contraction rate → 1 as
# τ → 1⁻ (critical slowing). Returns (m, residual ‖G(m) − m‖∞).
function _anderson_solve(G!, m0::Vector{Float64}, mlo::Float64, mhi::Float64;
                         tol::Float64 = 1.0e-13, maxit::Int = 2000)
    m = copy(m0)
    g = similar(m); f = similar(m); gp = zero(m); fp = zero(m); df = similar(m)
    G!(g, m)
    @. f = g - m
    Δ = maximum(abs, f)
    for k = 1:maxit
        Δ <= tol && break
        if k == 1
            @. m = g
        else
            @. df = f - fp
            d2 = dot(df, df)
            β = d2 > 1.0e-30 ? dot(f, df) / d2 : 0.0
            @. m = clamp(g - β * (g - gp), mlo, mhi)
        end
        @. gp = g
        @. fp = f
        G!(g, m)
        @. f = g - m
        Δ = maximum(abs, f)
    end
    return m, Δ
end

# Is atom `a` coupled (a non-zero row in the molecular-field matrix)? A free spin (all-zero
# row, no single-ion) feels no field and disorders for every τ > 0.
function _is_coupled(Abar::Matrix{Float64}, a::Int)::Bool
    @inbounds for b = 1:size(Abar, 2)
        abs(Abar[a, b]) > 1.0e-12 && return true
    end
    return false
end

# The isotropic (l=1-only) closed-form path: m_a = L(3(Ā m)_a/τ), per-atom vMF concentration
# κ_a = 3(Ā m)_a/τ. Returns (ordered, κ::Vector, m::Vector).
function _coupled_state(Abar::Matrix{Float64}, τ::Float64, n::Int)
    if τ < _MFA_MIN_TAU
        m = Vector{Float64}(undef, n)
        κ = Vector{Float64}(undef, n)
        @inbounds for a = 1:n
            c = _is_coupled(Abar, a)
            m[a] = c ? 1.0 : 0.0
            κ[a] = c ? Inf : 0.0
        end
        return (false, κ, m)
    elseif τ > _MFA_MAX_TAU
        return (false, fill(_MFA_KAPPA_UNIFORM, n), zeros(n))
    end
    G!(out, m) = begin
        mul!(out, Abar, m)
        @inbounds for a = 1:n
            out[a] = clamp(_langevin(3 * out[a] / τ), 0.0, 1.0)
        end
        out
    end
    m, Δ = _anderson_solve(G!, ones(n), 0.0, 1.0)
    Δ > 1.0e-6 && @warn "the coupled mean-field self-consistency did not converge at " *
        "τ = $τ (residual $Δ); m_a may be inaccurate (critical slowing near T_MF, or a " *
        "frustrated reference)."
    κ = Vector{Float64}(undef, n)
    f = Abar * m
    @inbounds for a = 1:n
        κ[a] = max(0.0, 3 * f[a] / τ)
    end
    return (false, κ, m)
end

# The tensorial / single-ion (Bingham) path: solve m_a = ⟨e·ê_a⟩ under P ∝ exp(−V_a) by
# sphere quadrature, with V_a = β(e·g_a + e' A_a e), β = 3/(ρτ). Returns (cs, m): the
# per-atom single-site coefficient vectors `cs` (for the Metropolis draw) and the
# magnetizations `m`. τ is floored at _MFA_MIN_TAU so β stays finite (T = 0 limit).
function _tensor_state(exch::ExchangeModel, ehat::Vector{SVector{3,Float64}}, ρ::Float64,
                       τ::Float64)
    n = exch.natoms
    β = 3 / (ρ * max(τ, _MFA_MIN_TAU))
    if τ < _MFA_MIN_TAU
        # Fully ordered limit: every atom with a molecular field or a single-ion term
        # saturates along ê_a (m=1); a free spin (no field, no single-ion) stays disordered
        # (m=0). Skip the iteration — at the τ floor β is enormous and the quadrature would
        # be needlessly fine — and build sharply-peaked coefficients from the ordered means.
        cs = Vector{Vector{Float64}}(undef, n)
        m = Vector{Float64}(undef, n)
        ones_m = ones(n)
        @inbounds for a = 1:n
            g = _molecular_field(exch, ehat, ones_m, a)
            # The order parameter saturates (m → 1) only with a net l=1 molecular field. An
            # atom with a purely single-ion (l=2) mean field has an e → −e symmetric Bingham
            # distribution, so ⟨e·ê_a⟩ → 0 — continuous with the τ just above the floor (where
            # m = (4π/3) _l1_field(⟨Z⟩)·ê_a → 0), not 1. The sharply-peaked cs (built from the
            # ordered means) still gives the correct symmetric draw.
            m[a] = norm(g) > 1.0e-12 ? 1.0 : 0.0
            cs[a] = _site_coeffs(g, exch.onsite[a], β)
        end
        return cs, m
    end
    G!(out, m) = begin
        @inbounds for a = 1:n
            g = _molecular_field(exch, ehat, m, a)
            c = _site_coeffs(g, exch.onsite[a], β)
            avg = multipole_average(c, 2)
            emean = (4π / 3) * _l1_field(avg)         # ⟨e⟩ from ⟨Z_1m⟩
            out[a] = clamp(dot(emean, ehat[a]), -1.0, 1.0)
        end
        out
    end
    m, Δ = _anderson_solve(G!, ones(n), -1.0, 1.0)
    Δ > 1.0e-6 && @warn "the tensorial mean-field self-consistency did not converge at " *
        "τ = $τ (residual $Δ); m_a may be inaccurate (critical slowing near T_MF, or a " *
        "frustrated reference)."
    cs = Vector{Vector{Float64}}(undef, n)
    @inbounds for a = 1:n
        cs[a] = _site_coeffs(_molecular_field(exch, ehat, m, a), exch.onsite[a], β)
    end
    return cs, m
end

# --- P4: the full multipole mean field over all SCE clusters / l --------------------

# Reference multipoles ⟨Z_lm⟩ = Z_lm(ê_a) (the fully ordered state), per atom, length
# (lmax+1)² and ordered by `Harmonics.lm_index`.
function _ref_multipoles(ehat::Vector{SVector{3,Float64}}, lmax::Int)::Vector{Vector{Float64}}
    return [[Harmonics.Zlm_unsafe(l, m, ehat[a]) for l = 0:lmax for m = -l:l]   # ehat is unit
            for a = 1:length(ehat)]
end

# Build every atom's single-site coefficient vector c_a from the current multipole averages
# `Zavg` (per atom): c_a[lm(ls[i],μ_i)] += coef·folded[idx]·∏_{k≠i} ⟨Z_{ls[k],μ_k}⟩_{atoms[k]},
# summed over all terms and positions i with atoms[i] = a, then scaled by β. Mirrors
# `accumulate_grad!` with the site-a harmonic left symbolic. `cs[a]` is zeroed first.
function _site_coeffs_all!(cs::Vector{Vector{Float64}}, terms::Vector{_MFATerm},
                           Zavg::Vector{Vector{Float64}}, β::Float64, n::Int)
    @inbounds for a = 1:n
        fill!(cs[a], 0.0)
    end
    # Dispatch through a rank-specialized barrier: `_MFATerm.folded` is `Array{Float64}` (rank
    # erased), so `CartesianIndices(folded)` / `idx[k]` would dynamically dispatch on every
    # access in this hottest many-body loop; the barrier recovers the concrete rank D.
    for term in terms
        _accumulate_term!(cs, term.coef, term.atoms, term.ls, term.folded, Zavg)
    end
    @inbounds for a = 1:n
        cs[a] .*= β
    end
    return cs
end

# Accumulate one cluster term's leave-one-out contribution into `cs`, specialized on the
# concrete tensor rank D = length(atoms) = ndims(folded). Same arithmetic/order as before.
@inline function _accumulate_term!(cs::Vector{Vector{Float64}}, coef::Float64,
                                   atoms::Vector{Int}, ls::Vector{Int}, folded::Array{Float64,D},
                                   Zavg::Vector{Vector{Float64}}) where {D}
    @inbounds for idx in CartesianIndices(folded)
        w = coef * folded[idx]
        w == 0.0 && continue
        for i = 1:D
            p = 1.0
            for k = 1:D
                k == i && continue
                μk = idx[k] - ls[k] - 1
                p *= Zavg[atoms[k]][Harmonics.lm_index(ls[k], μk)]
            end
            p == 0.0 && continue
            μi = idx[i] - ls[i] - 1
            cs[atoms[i]][Harmonics.lm_index(ls[i], μi)] += w * p
        end
    end
    return cs
end

# The full multipole mean-field state at reduced temperature τ: iterate the per-atom
# multipole averages ⟨Z_lm⟩_a (β = 3/(ρτ), ρ the l=1 bilinear Perron) to self-consistency,
# then return (cs, m): the single-site coefficient vectors for the Metropolis draw and the
# magnetizations m_a = ⟨e·ê_a⟩. τ floored at _MFA_MIN_TAU so β stays finite.
function _multipole_state(mf::MultipoleModel, ehat::Vector{SVector{3,Float64}}, ρ::Float64,
                          τ::Float64)
    n = mf.natoms
    lmax = mf.lmax
    nlm = (lmax + 1)^2
    β = 3 / (ρ * max(τ, _MFA_MIN_TAU))
    Zref = _ref_multipoles(ehat, lmax)
    cs = [zeros(Float64, nlm) for _ = 1:n]
    if τ < _MFA_MIN_TAU
        # Fully ordered limit: ⟨Z⟩ = Z(ê_a). An atom saturates (m → 1) only with a net l=1
        # molecular field; one whose mean field is purely even-l (e.g. single-ion only) has an
        # e → −e symmetric distribution, so ⟨e·ê_a⟩ → 0 — continuous with the τ just above the
        # floor. Gate on the l=1 part of the single-site potential (β-independent ratio).
        _site_coeffs_all!(cs, mf.terms, Zref, β, n)
        m = Vector{Float64}(undef, n)
        @inbounds for a = 1:n
            g = _l1_field(cs[a])
            m[a] = norm(g) > 1.0e-12 * (1 + norm(cs[a])) ? 1.0 : 0.0
        end
        return cs, m
    end
    Zavg = [copy(Zref[a]) for a = 1:n]
    flat0 = reduce(vcat, Zref)
    G! = (out, x) -> begin
        @inbounds for a = 1:n, j = 1:nlm
            Zavg[a][j] = x[(a - 1) * nlm + j]
        end
        _site_coeffs_all!(cs, mf.terms, Zavg, β, n)
        @inbounds for a = 1:n
            avg = multipole_average(cs[a], lmax)
            for j = 1:nlm
                out[(a - 1) * nlm + j] = avg[j]
            end
        end
        out
    end
    flat, Δ = _anderson_solve(G!, flat0, -1.5, 1.5)
    Δ > 1.0e-6 && @warn "the full-multipole mean-field self-consistency did not converge " *
        "at τ = $τ (residual $Δ); the multipole averages may be inaccurate (critical " *
        "slowing near T_MF, or a frustrated reference)."
    @inbounds for a = 1:n, j = 1:nlm
        Zavg[a][j] = flat[(a - 1) * nlm + j]
    end
    _site_coeffs_all!(cs, mf.terms, Zavg, β, n)
    m = [dot((4π / 3) * _l1_field(Zavg[a]), ehat[a]) for a = 1:n]
    return cs, m
end
