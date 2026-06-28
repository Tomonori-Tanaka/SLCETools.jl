# Mean-field sampler — P2/P3: the `ExchangeModel` carrier and the coupled mean-field
# self-consistency (see `docs/specs/mfa-sampling.md`).
#
# `ExchangeModel` carries the bilinear couplings (and single-ion anisotropy) the mean-field
# sampler needs, in a DFT-code-neutral form:
#   - `bilinear[a,b]` is the full 3×3 matrix `S_ab` coupling atom `a` to every periodic
#     image of atom `b`, so the molecular field on `a` is `g_a = Σ_b S_ab ⟨e_b⟩`. Its
#     isotropic part is Heisenberg exchange, its antisymmetric part the DM vector, its
#     traceless-symmetric part the anisotropic (Γ) exchange.
#   - `onsite[a]` is the single-ion anisotropy matrix `A_a` (the `ls=[2]` channel), a
#     temperature-independent quadratic form `e' A_a e` (decision §1.3).
#   - `Jiso[a,b] = tr(S_ab)/3` is the isotropic part, kept for the P2 fast path.
#
# Mean-field decoupling of the bilinear energy `E = Σ_{bonds} e_a' M_ab e_b` gives, with the
# neighbors at their rigid-axis means `⟨e_b⟩ = m_b ê_b` (decision D2), a single-site
# potential on atom `a`
#   V_a(e) = β [ e·g_a + e' A_a e ],   g_a = Σ_b S_ab m_b ê_b,   β = 3/(ρ τ),
# whose l=1 part is the molecular field (vMF cone) and whose l=2 part (single-ion + the
# rigid-axis bilinear feeds only l=1) is a Bingham factor. The longitudinal linearization
#   A[a,b] = −ê_a' S_ab ê_b
# folds the reference directions in (any collinear ferro/antiferro/ferri order becomes
# ferromagnetic in the magnitudes m_a), so its Perron eigenvalue ρ gives `T_MF = ρ/3`, and
# in the reduced temperature τ = T/T_MF the self-consistency for the magnetizations is
#   m_a = ⟨e·ê_a⟩ under P(e) ∝ exp(−V_a(e)).
# For purely isotropic exchange (no DMI/anisotropy, no single-ion) `V_a` is l=1-only and
# this reduces to `m_a = L(3(Ā m)_a/τ)`, Ā = A/ρ — the closed-form vMF P2 path. Otherwise
# the Bingham shape needs the general quadrature/Metropolis engine (P3). Scaling all
# couplings scales A and ρ together, so `Ā` and every `m_a(τ)` is scale-free (D4): only the
# coupling *ratios* matter (the single-ion strength relative to the exchange is physical).

const _I3 = SMatrix{3,3,Float64}(I)
# Tesseral normalization constants (shared with the Sunny export `_l1_pair_matrix` /
# `_l2_onsite_matrix`): Z_{1,m} = √(3/4π)·(component); the l=2 constants below.
const _N1 = sqrt(3 / (4π))
const _A2 = sqrt(15 / (16π))
const _B2 = sqrt(5 / (16π))

"""
    ExchangeModel(Jiso; onsite = nothing)
    ExchangeModel(bilinear; onsite = nothing)
    ExchangeModel(model::SCEModel)

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
- `ExchangeModel(model)` — extract `bilinear` and `onsite` from a fitted [`SCEModel`](@ref)
  by reusing the Sunny bilinear (`ls=[1,1]`) and single-ion (`ls=[2]`) extraction; only the
  higher-order / higher-`l` channels are dropped (a P4 extension) and reported via `@warn`.
"""
struct ExchangeModel
    natoms::Int
    Jiso::Matrix{Float64}                              # tr(S)/3, the isotropic part
    bilinear::Matrix{SMatrix{3,3,Float64,9}}           # S_ab: field g_a = Σ_b S_ab ⟨e_b⟩
    onsite::Vector{SMatrix{3,3,Float64,9}}             # single-ion A_a (zero ⇒ none)
    isotropic::Bool                                    # all S_ab ∝ I and onsite all zero
end

Base.show(io::IO, m::ExchangeModel) =
    print(io, "ExchangeModel(", m.natoms, " atoms", m.isotropic ? ", isotropic" : ", tensorial", ")")

# Canonical builder: validate shapes and the energy symmetry S_ba = S_ab', then classify.
function _build_exchange(bilinear::Matrix{SMatrix{3,3,Float64,9}},
                         onsite::Vector{SMatrix{3,3,Float64,9}})::ExchangeModel
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
    return ExchangeModel(n, Jiso, bil, onsite, iso)
end

# Normalize an `onsite` keyword (nothing ⇒ zeros) to a length-n vector of SMatrices.
function _onsite_vec(onsite, n::Int)::Vector{SMatrix{3,3,Float64,9}}
    onsite === nothing && return fill(zero(SMatrix{3,3,Float64,9}), n)
    length(onsite) == n ||
        throw(ArgumentError("onsite must have one matrix per atom ($n); got $(length(onsite))"))
    return [SMatrix{3,3,Float64}(A) for A in onsite]
end

function ExchangeModel(Jiso::AbstractMatrix{<:Real}; onsite = nothing)
    n = size(Jiso, 1)
    size(Jiso, 2) == n || throw(ArgumentError("Jiso must be square; got $(size(Jiso))"))
    J = Matrix{Float64}(Jiso)
    maximum(abs.(J - J'); init = 0.0) <= 1.0e-10 * (1 + maximum(abs.(J); init = 0.0)) ||
        throw(ArgumentError("Jiso must be symmetric (Jiso[a,b] = Jiso[b,a])"))
    bil = [J[a, b] * _I3 for a = 1:n, b = 1:n]
    return _build_exchange(bil, _onsite_vec(onsite, n))
end

function ExchangeModel(bilinear::AbstractMatrix{<:SMatrix{3,3}}; onsite = nothing)
    bil = Matrix{SMatrix{3,3,Float64,9}}(bilinear)
    return _build_exchange(bil, _onsite_vec(onsite, size(bil, 1)))
end

# --- the longitudinal molecular-field matrix and its Perron analysis ---------------

# A[a,b] = −ê_a' S_ab ê_b, the linearized (longitudinal) coupling. For isotropic S = Jiso·I
# this is −Jiso (ê_a·ê_b); symmetric because S_ba = S_ab'.
function _mfa_matrix(exch::ExchangeModel, ref::Matrix{Float64})::Matrix{Float64}
    n = exch.natoms
    A = zeros(Float64, n, n)
    @inbounds for b = 1:n
        eb = SVector{3,Float64}(ref[1, b], ref[2, b], ref[3, b])
        for a = 1:n
            ea = SVector{3,Float64}(ref[1, a], ref[2, a], ref[3, a])
            A[a, b] = -dot(ea, exch.bilinear[a, b] * eb)
        end
    end
    return A
end

# Largest eigenvalue ρ of the symmetric A and whether its eigenvector is sign-definite
# (the reference is a clean ordering mode).
function _perron(A::Matrix{Float64})
    F = eigen(Symmetric(A))
    ρ = F.values[end]
    v = F.vectors[:, end]
    tol = 1.0e-8 * maximum(abs.(v); init = 1.0)
    pos = count(>(tol), v)
    neg = count(<(-tol), v)
    return ρ, (pos == 0 || neg == 0)
end

# Verify the reference is a stationary state: the bilinear molecular field at full order,
# h_a = −Σ_b S_ab ê_b, must be parallel to ê_a (transverse ≈ 0) and aligned (ê_a·h_a > 0).
# Exact for any collinear, isotropic reference; warns otherwise (D2; e.g. DMI / anisotropy
# cant the true ground state, so a collinear reference is then not stationary).
function _check_reference_stationary(exch::ExchangeModel, ref::Matrix{Float64})
    n = exch.natoms
    worst_t = 0.0
    scale = 0.0
    antialigned = false
    @inbounds for a = 1:n
        ea = SVector{3,Float64}(ref[1, a], ref[2, a], ref[3, a])
        h = zero(SVector{3,Float64})
        for b = 1:n
            eb = SVector{3,Float64}(ref[1, b], ref[2, b], ref[3, b])
            h -= exch.bilinear[a, b] * eb
        end
        λ = dot(ea, h)
        worst_t = max(worst_t, norm(h - λ * ea))
        scale = max(scale, norm(h))
        λ < -1.0e-8 * (1 + norm(h)) && (antialigned = true)
    end
    worst_t > 1.0e-6 * (1 + scale) && @warn "the reference is not a stationary state of " *
        "the exchange model (bilinear molecular field not ∥ ê_a; max transverse $(worst_t)); " *
        "the rigid-axis MFA is approximate here — a rotating cone axis ê_a(τ) is a later extension."
    antialigned && @warn "the bilinear molecular field is antiparallel to the reference on " *
        "some atom; the reference is not a stable ordered state of the exchange model."
    return nothing
end

# --- single-site tesseral coefficients from the molecular field and single-ion ------

# Write the l=1 coefficients of the linear form e·g into c (length (lmax+1)²): with
# Z_{1,1}=N1 e_x, Z_{1,-1}=N1 e_y, Z_{1,0}=N1 e_z, the coefficient of Z_{1,m} is component/N1.
@inline function _l1_coeffs!(c::Vector{Float64}, g::SVector{3,Float64})
    c[Harmonics.lm_index(1, 1)] = g[1] / _N1
    c[Harmonics.lm_index(1, -1)] = g[2] / _N1
    c[Harmonics.lm_index(1, 0)] = g[3] / _N1
    return c
end

# Write the l=2 coefficients of the quadratic form e' A e (traceless symmetric part) into c.
# Inverse of `_l2_onsite_matrix`: folded[m] for m=−2..2 from the matrix entries.
@inline function _l2_coeffs!(c::Vector{Float64}, A::SMatrix{3,3,Float64,9})
    axx = A[1, 1]; ayy = A[2, 2]; azz = A[3, 3]
    t = (axx + ayy + azz) / 3                         # remove the trace (a constant shift)
    axx -= t; ayy -= t; azz -= t
    axy = (A[1, 2] + A[2, 1]) / 2                      # symmetric part
    ayz = (A[2, 3] + A[3, 2]) / 2
    axz = (A[1, 3] + A[3, 1]) / 2
    c[Harmonics.lm_index(2, -2)] = axy / _A2
    c[Harmonics.lm_index(2, -1)] = ayz / _A2
    c[Harmonics.lm_index(2, 0)] = azz / (2 * _B2)
    c[Harmonics.lm_index(2, 1)] = axz / _A2
    c[Harmonics.lm_index(2, 2)] = (axx - ayy) / (2 * _A2)
    return c
end

# The single-site potential coefficients c (length 9, lmax=2) of V_a = β(e·g + e' A e).
function _site_coeffs(g::SVector{3,Float64}, A::SMatrix{3,3,Float64,9}, β::Float64)::Vector{Float64}
    c = zeros(Float64, 9)
    _l1_coeffs!(c, g)
    _l2_coeffs!(c, A)
    c .*= β
    return c
end

# The bilinear molecular field g_a = Σ_b S_ab m_b ê_b on atom a (rigid-axis neighbor means).
function _molecular_field(exch::ExchangeModel, ehat::Vector{SVector{3,Float64}},
                          m::Vector{Float64}, a::Int)::SVector{3,Float64}
    g = zero(SVector{3,Float64})
    @inbounds for b = 1:exch.natoms
        g += exch.bilinear[a, b] * (m[b] * ehat[b])
    end
    return g
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
            coupled = norm(g) > 1.0e-12 || maximum(abs.(exch.onsite[a])) > 1.0e-12
            m[a] = coupled ? 1.0 : 0.0
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
    MultipoleField

The digested full-multipole mean field of a fitted SCE (P4): every cluster term
(`_MFATerm`), the `lmax`, and the bilinear [`ExchangeModel`](@ref) (used only for the
`l=1` temperature scale `ρ`). Built by `MultipoleField(model::SCEModel)`; consumed by the
[`MFASampler`](@ref) tensorial/Metropolis path.
"""
struct MultipoleField
    natoms::Int
    lmax::Int
    terms::Vector{_MFATerm}
    bilinear::ExchangeModel
end

Base.show(io::IO, mf::MultipoleField) =
    print(io, "MultipoleField(", mf.natoms, " atoms, lmax=", mf.lmax, ", ",
          length(mf.terms), " terms)")

# Does atom a appear in any term (i.e. feel a mean field)?
_atom_in_terms(mf::MultipoleField, a::Int)::Bool = any(t -> a in t.atoms, mf.terms)

# Reference multipoles ⟨Z_lm⟩ = Z_lm(ê_a) (the fully ordered state), per atom, length
# (lmax+1)² and ordered by `Harmonics.lm_index`.
function _ref_multipoles(ehat::Vector{SVector{3,Float64}}, lmax::Int)::Vector{Vector{Float64}}
    return [[Harmonics.Zlm(l, m, ehat[a]) for l = 0:lmax for m = -l:l]
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
    @inbounds for term in terms
        atoms = term.atoms
        ls = term.ls
        folded = term.folded
        N = length(atoms)
        for idx in CartesianIndices(folded)
            w = term.coef * folded[idx]
            w == 0.0 && continue
            for i = 1:N
                p = 1.0
                for k = 1:N
                    k == i && continue
                    μk = idx[k] - ls[k] - 1
                    p *= Zavg[atoms[k]][Harmonics.lm_index(ls[k], μk)]
                end
                p == 0.0 && continue
                μi = idx[i] - ls[i] - 1
                cs[atoms[i]][Harmonics.lm_index(ls[i], μi)] += w * p
            end
        end
    end
    @inbounds for a = 1:n
        cs[a] .*= β
    end
    return cs
end

# The full multipole mean-field state at reduced temperature τ: iterate the per-atom
# multipole averages ⟨Z_lm⟩_a (β = 3/(ρτ), ρ the l=1 bilinear Perron) to self-consistency,
# then return (cs, m): the single-site coefficient vectors for the Metropolis draw and the
# magnetizations m_a = ⟨e·ê_a⟩. τ floored at _MFA_MIN_TAU so β stays finite.
function _multipole_state(mf::MultipoleField, ehat::Vector{SVector{3,Float64}}, ρ::Float64,
                          τ::Float64)
    n = mf.natoms
    lmax = mf.lmax
    nlm = (lmax + 1)^2
    β = 3 / (ρ * max(τ, _MFA_MIN_TAU))
    Zref = _ref_multipoles(ehat, lmax)
    cs = [zeros(Float64, nlm) for _ = 1:n]
    if τ < _MFA_MIN_TAU
        # Fully ordered limit: ⟨Z⟩ = Z(ê_a); each atom appearing in a term saturates (m=1).
        _site_coeffs_all!(cs, mf.terms, Zref, β, n)
        m = [_atom_in_terms(mf, a) ? 1.0 : 0.0 for a = 1:n]
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
