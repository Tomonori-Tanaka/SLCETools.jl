# Per-atom MFA distribution viz — the per-atom single-site coefficients at one τ and a
# Julia-side normalized density (for verification).
#
# The single-site mean-field distribution on atom `a` is `P(e) ∝ exp(−V_a(e))` with the
# finite tesseral expansion `V_a(e) = Σ_{l≥1,m} c_a[lm_index(l,m)]·Z_lm(e)` (the engine).
# Because the *exponent* `V_a` — not `P` itself — is what has a finite harmonic expansion,
# exporting the coefficient vectors `c_a` is exact (no cutoff) and tiny. The shared render
# grid and basis matrix are in `grid.jl`; the JSON document in `serialize.jl`.

"""
    SiteDistributionField

Per-atom single-site distribution coefficients at one reduced temperature `tau`. Each
`coeffs[a]` is a `_site_potential`-ready tesseral vector of length `(lmax+1)²`; the
distribution on atom `a` is `P(e) ∝ exp(−Σ_k coeffs[a][k]·Z_k(e))`. `m[a]` is the
self-consistent magnetization `m_a = ⟨e·ê_a⟩` (equal to `mfa_sublattice_m(sampler, tau)[a]`
by construction). `reference[:, a]` is the rigid axis `ê_a`.
"""
struct SiteDistributionField
    reference::Matrix{Float64}        # 3×n unit ê_a
    coeffs::Vector{Vector{Float64}}   # per-atom c_a, each length (lmax+1)²
    m::Vector{Float64}                # per-atom magnetization m_a ∈ [−1, 1]
    tau::Float64
    lmax::Int
end

# Rendering-only clamp for the fully ordered limit (τ < _MFA_MIN_TAU ⇒ κ = Inf): a finite
# but very sharp vMF concentration, so exp(−V) is a finite peak rather than Inf/NaN.
const _MFA_KAPPA_CAP = 3 / _MFA_MIN_TAU

"""
    mfa_site_coefficients(sampler, tau) -> SiteDistributionField

The per-atom single-site coefficient vectors `c_a` and magnetizations `m_a` at reduced
temperature `tau`, dispatching exactly as [`sample`](@ref) / [`mfa_sublattice_m`](@ref):
the Metropolis path (tensorial / multipole) returns the solver's coefficients directly,
while the closed-form vMF path (single global / isotropic exchange) converts each
concentration `κ_a` along `ê_a` into the equivalent `l = 1` coefficients of `V_a = −κ_a(ê_a·e)`.
The fully ordered limit (`κ_a = Inf`) is clamped to a large finite concentration for rendering.
"""
function mfa_site_coefficients(s::MFASampler, tau::Real)::SiteDistributionField
    τ = Float64(tau)
    n = _natoms(s)
    ref = copy(s.reference)
    if _needs_metropolis(s)
        cs, m = _coeffs_and_m(s, τ)
        lmax = isqrt(length(cs[1])) - 1
        return SiteDistributionField(ref, cs, m, τ, lmax)
    end
    _, κ, m = _mfa_state(s, τ)
    ehat = _ehat(s)
    coeffs = Vector{Vector{Float64}}(undef, n)
    @inbounds for a = 1:n
        κa = min(isfinite(κ[a]) ? κ[a] : _MFA_KAPPA_CAP, _MFA_KAPPA_CAP)
        c = zeros(Float64, 4)                            # lmax = 1 ⇒ (1+1)² = 4
        _l1_coeffs!(c, -κa * ehat[a])                    # V_a = −κ_a(ê_a·e) = g·e, g = −κ_a ê_a
        coeffs[a] = c
    end
    return SiteDistributionField(ref, coeffs, m, τ, 1)
end

"""
    site_probabilities(field, grid) -> Matrix{Float64}    # n × npoints

The normalized single-site density on the shared grid: row `a` is `p_a(e_i)` with
`Σ_i p_a(e_i)·grid.weight == 1`. Evaluated through `_site_potential` (the same harmonic
kernel the exported basis matrix reproduces), so it doubles as the verification that the
viewer's `exp(−Z·c_a)` path is correct. The per-row `max(−V)` shift keeps the sharply
peaked (low-τ / ordered) limit finite.
"""
function site_probabilities(field::SiteDistributionField, grid::SphereGrid)::Matrix{Float64}
    n = length(field.coeffs)
    npts = length(grid.dirs)
    P = Matrix{Float64}(undef, n, npts)
    @inbounds for a = 1:n
        c = field.coeffs[a]
        vmin = Inf
        for i = 1:npts
            v = _site_potential(c, grid.dirs[i])
            P[a, i] = v
            vmin = min(vmin, v)
        end
        total = 0.0
        for i = 1:npts
            p = exp(-(P[a, i] - vmin))
            P[a, i] = p
            total += p
        end
        inv_norm = 1.0 / (total * grid.weight)
        for i = 1:npts
            P[a, i] *= inv_norm
        end
    end
    return P
end
