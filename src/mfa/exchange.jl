# Mean-field sampler — P2/P3: building an `ExchangeModel` and its longitudinal molecular-field
# analysis (see `docs/specs/mfa-sampling.md`). The struct itself lives in `types.jl`; this file
# holds the convenience constructors and the supporting linear algebra.
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

# Tesseral normalization constants (shared with the Sunny export `_l1_pair_matrix` /
# `_l2_onsite_matrix`): Z_{1,m} = √(3/4π)·(component); the l=2 constants below.
const _N1 = sqrt(3 / (4π))
const _A2 = sqrt(15 / (16π))
const _B2 = sqrt(5 / (16π))

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
    bil = Matrix{SMatrix{3,3,Float64,9}}([J[a, b] * _I3 for a = 1:n, b = 1:n])
    return ExchangeModel(bil, _onsite_vec(onsite, n))
end

function ExchangeModel(bilinear::AbstractMatrix{<:SMatrix{3,3}}; onsite = nothing)
    bil = Matrix{SMatrix{3,3,Float64,9}}(bilinear)
    return ExchangeModel(bil, _onsite_vec(onsite, size(bil, 1)))
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
