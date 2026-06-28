# Mean-field sampler — P0: the single-site engine (see `docs/specs/mfa-sampling.md`).
#
# Draws a unit spin direction from the single-site Boltzmann distribution
#
#     P(e) ∝ exp(−V(e)),    V(e) = Σ_{l≥1, m} c[idx(l,m)]·Z_lm(e),
#
# where `c` is the already-β-scaled generalized-field coefficient vector indexed by
# `Harmonics.lm_index(l, m)` (length `(lmax+1)²`; the `l = 0` entry, a constant shift,
# is ignored). Mean-field decoupling of any SCE term — any body order, any `l` — folds
# into such a `V`, so one engine covers every case (spec §1.1).
#
# Provides: the potential `_site_potential`; a closed-form von Mises–Fisher draw
# `sample_vmf` (the `l = 1`-only fast path, via `_l1_field`); a general single-site
# Metropolis draw `sample_site_metropolis`; and a deterministic spherical quadrature
# (`sphere_quadrature` / `multipole_average`) for the multipole averages `⟨Z_lm⟩` the
# self-consistency (P1+) needs. RNG is always an explicit `AbstractRNG`.

# --- small geometry helpers ------------------------------------------------------

# A uniform random unit vector on S². (`v ≈ 0` has probability zero; left unguarded.)
function _random_unit(rng::AbstractRNG)::SVector{3,Float64}
    v = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
    return v / norm(v)
end

# The lmax implied by a field of length `(lmax+1)²`; errors loudly on a non-square length.
function _field_lmax(c::AbstractVector{<:Real})::Int
    lmax = isqrt(length(c)) - 1
    (lmax + 1)^2 == length(c) || throw(ArgumentError(
        "field length $(length(c)) is not a perfect square (lmax+1)²"))
    return lmax
end

# Two orthonormal tangent vectors spanning the plane orthogonal to the unit `n`.
function _orthonormal_basis(n::SVector{3,Float64})
    a = abs(n[1]) < 0.9 ? SVector{3,Float64}(1, 0, 0) : SVector{3,Float64}(0, 1, 0)
    e1 = a - dot(a, n) * n
    e1 = e1 / norm(e1)
    return e1, cross(n, e1)
end

# Rotate unit `e` about unit `axis` by angle `θ` (Rodrigues). Volume-preserving and
# symmetric in (e, e′), which the Metropolis proposal relies on for detailed balance.
function _rotate(e::SVector{3,Float64}, axis::SVector{3,Float64}, θ::Real)::SVector{3,Float64}
    c, s = cos(θ), sin(θ)
    return e * c + cross(axis, e) * s + axis * (dot(axis, e) * (1 - c))
end

# --- the single-site potential ---------------------------------------------------

"""
    _site_potential(c, e) -> Float64

The single-site potential `V(e) = Σ_{l≥1, m} c[lm_index(l, m)]·Z_lm(e)`. `c` has length
`(lmax+1)²`; the `l = 0` entry is ignored (a constant shift cancels in `exp(−V)`).
"""
function _site_potential(c::AbstractVector{<:Real}, e::AbstractVector{<:Real})::Float64
    lmax = isqrt(length(c)) - 1
    v = 0.0
    @inbounds for l = 1:lmax
        for m = -l:l
            ci = c[Harmonics.lm_index(l, m)]
            ci == 0 && continue
            v += ci * Harmonics.Zlm(l, m, e)
        end
    end
    return v
end

# The molecular-field vector `g` of the `l = 1` part: `Σ_m c[1,m]·Z_1m(e) = g·e`. Each
# `Z_1m` is a linear form in `e`, so evaluating it at the three axes extracts its
# Cartesian coefficients (`Z_1m(d̂)` = the `d`-component), giving `g` convention-
# consistently from the package's own `Zlm`. (`grad_Zlm` is the *on-sphere* gradient,
# which vanishes at a pole, so it cannot be used here.)
function _l1_field(c::AbstractVector{<:Real})::SVector{3,Float64}
    axes = (SVector{3,Float64}(1, 0, 0), SVector{3,Float64}(0, 1, 0),
            SVector{3,Float64}(0, 0, 1))
    g = zero(MVector{3,Float64})
    @inbounds for m = -1:1
        cm = c[Harmonics.lm_index(1, m)]
        for d = 1:3
            g[d] += cm * Harmonics.Zlm(1, m, axes[d])
        end
    end
    return SVector{3,Float64}(g)
end

# --- closed-form von Mises–Fisher draw (the l = 1 fast path) ----------------------

"""
    sample_vmf(rng, μ, κ) -> SVector{3,Float64}

Draw a unit vector from the von Mises–Fisher distribution with unit mean `μ` and
concentration `κ ≥ 0`, by the exact `p = 3` inverse-CDF construction (Ulrich 1984): the
cosine to `μ` is `w = 1 + κ⁻¹·log(u + (1−u)·e^{−2κ})`, `u ∼ U(0,1)`, with a uniform
azimuth in the tangent plane. For `κ → 0` the draw is isotropic.
"""
function sample_vmf(rng::AbstractRNG, μ::SVector{3,Float64}, κ::Real)::SVector{3,Float64}
    κ < 1.0e-10 && return _random_unit(rng)
    u = rand(rng)
    w = 1.0 + log(u + (1.0 - u) * exp(-2κ)) / κ
    w = clamp(w, -1.0, 1.0)
    s = sqrt(max(0.0, 1.0 - w * w))
    φ = 2π * rand(rng)
    e1, e2 = _orthonormal_basis(μ)
    return w * μ + s * (cos(φ) * e1 + sin(φ) * e2)
end

"""
    sample_vmf_field(rng, c) -> SVector{3,Float64}

Draw from `P(e) ∝ exp(−V(e))` when `V` has only an `l = 1` part: with molecular field
`g = _l1_field(c)`, `P ∝ exp(−g·e)` is `vMF(μ = −ĝ, κ = ‖g‖)`. Higher-`l` entries of `c`
are ignored — call only on an `l = 1`-only field (the engine's vMF fast path).
"""
function sample_vmf_field(rng::AbstractRNG, c::AbstractVector{<:Real})::SVector{3,Float64}
    _field_lmax(c)
    g = _l1_field(c)
    κ = norm(g)
    κ < 1.0e-10 && return _random_unit(rng)
    return sample_vmf(rng, -g / κ, κ)
end

# --- general single-site Metropolis draw -----------------------------------------

# One Metropolis step on the sphere with a symmetric random-rotation proposal: rotate
# `e` about a uniformly random axis by `θ ∼ step·N(0,1)`, accept with `min(1, e^{−ΔV})`.
@inline function _metropolis_step(rng::AbstractRNG, c::AbstractVector{<:Real},
                                  e::SVector{3,Float64}, V::Float64, step::Float64)
    axis = _random_unit(rng)
    e2 = _rotate(e, axis, step * randn(rng))
    V2 = _site_potential(c, e2)
    if V2 <= V || rand(rng) < exp(V - V2)
        return e2, V2
    end
    return e, V
end

"""
    sample_site_metropolis(rng, c, n; step = 0.6, nburn = 200, thin = 10, e_init) -> Vector{SVector{3,Float64}}

Draw `n` (approximately decorrelated) unit directions from `P(e) ∝ exp(−V(e))`,
`V = Σ c·Z`, by single-site Metropolis with a symmetric random-rotation proposal
(`step` = proposal angle scale, radians). A `nburn`-step burn-in precedes the draws and
`thin` steps separate them. Handles any `V` (any `l`, hence DMI / anisotropy / many-body
fields) — the general engine; `sample_vmf_field` is the closed-form `l = 1` fast path.
"""
function sample_site_metropolis(rng::AbstractRNG, c::AbstractVector{<:Real}, n::Integer;
                                step::Real = 0.6, nburn::Integer = 200, thin::Integer = 10,
                                e_init::AbstractVector{<:Real} = _random_unit(rng),
                                )::Vector{SVector{3,Float64}}
    n >= 0 || throw(ArgumentError("n must be ≥ 0; got $n"))
    _field_lmax(c)
    stepf = Float64(step)
    e = SVector{3,Float64}(e_init)
    e = e / norm(e)
    V = _site_potential(c, e)
    for _ = 1:nburn
        e, V = _metropolis_step(rng, c, e, V, stepf)
    end
    out = Vector{SVector{3,Float64}}(undef, n)
    @inbounds for k = 1:n
        for _ = 1:thin
            e, V = _metropolis_step(rng, c, e, V, stepf)
        end
        out[k] = e
    end
    return out
end

# --- spherical quadrature for the multipole averages ⟨Z_lm⟩ ----------------------

"""
    SphereQuadrature

A product quadrature on `S²`: Gauss–Legendre in `z = cosθ` × uniform (trapezoidal) in
azimuth. `dirs` are the node unit vectors, `weights` the solid-angle weights (`Σ = 4π`).
Spectrally accurate for the smooth `Z_lm·exp(−V)` integrands of the self-consistency.
"""
struct SphereQuadrature
    dirs::Vector{SVector{3,Float64}}
    weights::Vector{Float64}
end

# Gauss–Legendre nodes/weights on [−1, 1] via Newton on the Legendre 3-term recurrence
# (no external quadrature dependency).
function _gauss_legendre(n::Int)
    n >= 1 || throw(ArgumentError("n must be ≥ 1; got $n"))
    nodes = Vector{Float64}(undef, n)
    weights = Vector{Float64}(undef, n)
    @inbounds for k = 1:n
        x = cos(π * (k - 0.25) / (n + 0.5))         # Chebyshev-like initial guess
        pn = 1.0
        dpn = 0.0
        for _ = 1:100
            p0, p1 = 1.0, x                          # P₀, P₁
            for j = 2:n
                p0, p1 = p1, ((2j - 1) * x * p1 - (j - 1) * p0) / j
            end
            pn = p1                                  # Pₙ(x);  P_{n-1}(x) = p0
            dpn = n * (x * pn - p0) / (x * x - 1)    # Pₙ′(x)
            dx = -pn / dpn
            x += dx
            abs(dx) <= 1.0e-15 && break
        end
        nodes[k] = x
        weights[k] = 2.0 / ((1.0 - x * x) * dpn * dpn)
    end
    return nodes, weights
end

# Dynamic range Δ = maxV − minV of the single-site potential over the sphere, estimated on
# a coarse Fibonacci grid. Δ is the exponent range of `exp(−V)`, i.e. the peak
# "concentration" that sets how finely the quadrature must resolve a sharply peaked
# (low-temperature, large-field) distribution — sizing by harmonic order alone would
# under-resolve it and bias `⟨Z_lm⟩`.
function _field_scale(c::AbstractVector{<:Real})::Float64
    n = 128
    ga = π * (3 - sqrt(5.0))                          # golden angle
    vmin, vmax = Inf, -Inf
    @inbounds for k = 0:(n - 1)
        z = 1 - 2 * (k + 0.5) / n
        r = sqrt(max(0.0, 1 - z * z))
        φ = ga * k
        v = _site_potential(c, SVector{3,Float64}(r * cos(φ), r * sin(φ), z))
        vmin = min(vmin, v)
        vmax = max(vmax, v)
    end
    return vmax - vmin
end

"""
    sphere_quadrature(lmax; concentration = 0.0, ntheta = 0, nphi = 0) -> SphereQuadrature

Build a product Gauss–Legendre × uniform-azimuth quadrature. The node count defaults to
`2·lmax + 6 + ceil(concentration)` in each angle: the `2·lmax + 6` resolves the harmonics,
and `concentration` (the `exp(−V)` exponent range, e.g. `≈ 2κ` for a vMF field of
concentration `κ`) adds the resolution a sharply peaked field needs. Pass explicit
`ntheta`/`nphi` to override. Use `multipole_average(c, lmax)` to have the size chosen from
`c` automatically.
"""
function sphere_quadrature(lmax::Integer; concentration::Real = 0.0,
                           ntheta::Integer = 0, nphi::Integer = 0)::SphereQuadrature
    # Cap the auto-sized node count: beyond ~this concentration the integrand is essentially
    # a delta and ⟨Z_lm⟩ → its value at the peak, so a finer grid buys nothing but cost — and
    # an uncapped `base ∝ concentration` would explode (base² nodes) for the large molecular
    # fields of the τ → 0 limit. An explicit `ntheta`/`nphi` is honored uncapped.
    base = min(2 * Int(lmax) + 6 + ceil(Int, max(0.0, Float64(concentration))), 256)
    nt = ntheta > 0 ? Int(ntheta) : base
    np = nphi > 0 ? Int(nphi) : base
    z, wz = _gauss_legendre(nt)
    dφ = 2π / np
    dirs = Vector{SVector{3,Float64}}(undef, nt * np)
    weights = Vector{Float64}(undef, nt * np)
    k = 0
    @inbounds for i = 1:nt
        st = sqrt(max(0.0, 1.0 - z[i] * z[i]))
        for j = 1:np
            φ = (j - 1) * dφ
            k += 1
            dirs[k] = SVector{3,Float64}(st * cos(φ), st * sin(φ), z[i])
            weights[k] = wz[i] * dφ                  # Σ = (Σ wz)·2π = 4π
        end
    end
    return SphereQuadrature(dirs, weights)
end

"""
    multipole_average(c, lmax) -> Vector{Float64}
    multipole_average(q, c, lmax) -> Vector{Float64}

The multipole averages `⟨Z_lm⟩ = ∫ Z_lm·e^{−V} dΩ / ∫ e^{−V} dΩ` under `P ∝ exp(−V)`,
`V = Σ c·Z`. Returns a length-`(lmax+1)²` vector indexed by `Harmonics.lm_index`; the
`l = 0` slot holds `⟨Z_00⟩` (the normalization check).

The two-argument form builds a quadrature sized for both `lmax` and the field strength of
`c` (`concentration = _field_scale(c)`) — correct even for a sharply peaked, low-temperature
field. The three-argument form evaluates on a caller-supplied `q` (reuse a precomputed grid,
but size it adequately for the field).
"""
function multipole_average(c::AbstractVector{<:Real}, lmax::Integer)::Vector{Float64}
    _field_lmax(c)
    return multipole_average(sphere_quadrature(lmax; concentration = _field_scale(c)), c, lmax)
end

function multipole_average(q::SphereQuadrature, c::AbstractVector{<:Real},
                           lmax::Integer)::Vector{Float64}
    nlm = (Int(lmax) + 1)^2
    acc = zeros(Float64, nlm)
    norm_z = 0.0
    @inbounds for t in eachindex(q.dirs)
        e = q.dirs[t]
        p = q.weights[t] * exp(-_site_potential(c, e))
        norm_z += p
        acc[1] += p * Harmonics.Zlm(0, 0, e)
        for l = 1:Int(lmax)
            for m = -l:l
                acc[Harmonics.lm_index(l, m)] += p * Harmonics.Zlm(l, m, e)
            end
        end
    end
    acc ./= norm_z
    return acc
end
