# Per-atom MFA probability distributions — coefficient export for the Python viewer.
#
# The single-site mean-field distribution on atom `a` is `P(e) ∝ exp(−V_a(e))` with the
# finite tesseral expansion `V_a(e) = Σ_{l≥1,m} c_a[lm_index(l,m)]·Z_lm(e)` (`site_engine.jl`).
# Because the *exponent* `V_a` — not `P` itself — is what has a finite harmonic expansion,
# exporting the coefficient vectors `c_a` is exact (no cutoff) and tiny. The viewer turns
# them into a per-direction density with a single matrix product against a shared basis
# matrix `Z[i,k] = Z_lm(e_i)` that this module evaluates once (so the harmonic convention
# stays owned by `SCEFitting.Harmonics`; Python never re-implements `Z_lm`).
#
# Provides: `SphereGrid` / `fibonacci_sphere` (the shared render grid), `harmonic_basis`
# (the `Z` matrix), `SiteDistributionField` / `mfa_site_coefficients` (per-atom `c_a` at one
# τ, dispatching exactly as `sample` / `mfa_sublattice_m`), `site_probabilities` (a Julia-side
# normalized density, for verification), and `write_mfa_distributions` (the JSON document).

# --- the shared render grid ------------------------------------------------------

"""
    SphereGrid

A near-uniform set of unit directions on `S²` (golden-angle Fibonacci spiral), shared by
every atom and every τ frame so the viewer triangulates the point cloud once. `weight` is
the single equal-solid-angle quadrature weight `4π/npoints` (no `sinθ` factor — unlike the
Gauss–Legendre `SphereQuadrature`, whose weight is `wz·dφ`; do not mix the two).
"""
struct SphereGrid
    dirs::Vector{SVector{3,Float64}}
    weight::Float64
end

"""
    fibonacci_sphere(npoints = 1000) -> SphereGrid

Build the shared `S²` render grid as a golden-angle Fibonacci spiral (the same construction
as `_field_scale` in `site_engine.jl`): `z = 1 − 2(k+½)/N`, `φ = ga·k`, `ga = π(3−√5)`. The
points are equal solid angle, so the quadrature weight is the constant `4π/N`.
"""
function fibonacci_sphere(npoints::Integer = 1000)::SphereGrid
    npoints >= 1 || throw(ArgumentError("npoints must be ≥ 1; got $npoints"))
    n = Int(npoints)
    ga = π * (3 - sqrt(5.0))                              # golden angle
    dirs = Vector{SVector{3,Float64}}(undef, n)
    @inbounds for k = 0:(n - 1)
        z = 1 - 2 * (k + 0.5) / n
        r = sqrt(max(0.0, 1 - z * z))
        φ = ga * k
        dirs[k + 1] = SVector{3,Float64}(r * cos(φ), r * sin(φ), z)
    end
    return SphereGrid(dirs, 4π / n)
end

"""
    harmonic_basis(grid, lmax) -> Matrix{Float64}    # npoints × (lmax+1)²

The shared tesseral-harmonic basis matrix `Z[i, k] = Z_lm(grid.dirs[i])`, with column `k`
indexed by `Harmonics.lm_index(l, m)`. The `l = 0` column (a constant shift `_site_potential`
ignores) is left zero, so `Z[i, :]·c == _site_potential(c, grid.dirs[i])` exactly for any
coefficient vector `c` of length `(lmax+1)²`. The viewer recovers `V_a = Z·c_a` with one
matrix product, never touching the harmonics.
"""
function harmonic_basis(grid::SphereGrid, lmax::Integer)::Matrix{Float64}
    lmax >= 0 || throw(ArgumentError("lmax must be ≥ 0; got $lmax"))
    nlm = (Int(lmax) + 1)^2
    npts = length(grid.dirs)
    Z = zeros(Float64, npts, nlm)
    @inbounds for i = 1:npts
        e = grid.dirs[i]
        for l = 1:Int(lmax)                              # column 1 (l=0) stays 0
            for m = -l:l
                Z[i, Harmonics.lm_index(l, m)] = Harmonics.Zlm(l, m, e)
            end
        end
    end
    return Z
end

# --- per-atom coefficients at one reduced temperature ---------------------------

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

# --- serialization (format-agnostic document + a thin JSON backend) -------------

# Normalize -0.0 → 0.0 for reproducible bytes (mirrors SCEFitting's persist.jl `_jnum`).
_jnum(x::Real)::Float64 = (y = Float64(x); y == 0.0 ? 0.0 : y)

# Build the self-describing document (a Dict tree, serializer-agnostic and unit-testable).
function _distribution_doc(sampler::MFASampler, crystal::Crystal;
                           taus::AbstractVector{<:Real}, npoints::Integer)
    n = _natoms(sampler)
    n == n_atoms(crystal) || throw(ArgumentError(
        "sampler atom count $n ≠ crystal atom count $(n_atoms(crystal))"))
    isempty(taus) && throw(ArgumentError("taus must be non-empty"))

    grid = fibonacci_sphere(npoints)
    fields = [mfa_site_coefficients(sampler, τ) for τ in taus]
    lmax = fields[1].lmax
    all(f.lmax == lmax for f in fields) ||
        throw(ArgumentError("lmax varies across τ; expected a single fixed lmax"))
    Z = harmonic_basis(grid, lmax)
    nlm = (lmax + 1)^2

    A = crystal.lattice.vectors                          # columns are lattice vectors
    latt = [[_jnum(A[1, j]), _jnum(A[2, j]), _jnum(A[3, j])] for j = 1:3]
    R = A * crystal.frac_positions                       # 3×n Cartesian positions (Å)
    pos = [[_jnum(R[1, a]), _jnum(R[2, a]), _jnum(R[3, a])] for a = 1:n]
    ref = sampler.reference
    refs = [[_jnum(ref[1, a]), _jnum(ref[2, a]), _jnum(ref[3, a])] for a = 1:n]

    dirs = [[_jnum(d[1]), _jnum(d[2]), _jnum(d[3])] for d in grid.dirs]
    Zrows = [[_jnum(Z[i, k]) for k = 1:nlm] for i = 1:length(grid.dirs)]

    frames = Vector{Dict{String,Any}}(undef, length(taus))
    for (t, τ) in enumerate(taus)
        f = fields[t]
        frames[t] = Dict{String,Any}(
            "tau" => _jnum(τ),
            "m" => [_jnum(x) for x in f.m],
            "coeffs" => [[_jnum(c) for c in f.coeffs[a]] for a = 1:n])
    end

    return Dict{String,Any}(
        "schema" => "scetools/mfa-distributions",
        "version" => 1,
        "lattice_vectors" => latt,
        "positions_cartesian" => pos,
        "species" => collect(crystal.species),
        "species_labels" => collect(crystal.species_labels),
        "reference" => refs,
        "lmax" => lmax,
        "T_MF" => _jnum(mfa_temperature_scale(sampler)),
        "grid" => Dict{String,Any}(
            "kind" => "fibonacci",
            "npoints" => length(grid.dirs),
            "weight" => _jnum(grid.weight),
            "directions" => dirs,
            "Z" => Zrows),
        "temperatures" => [_jnum(τ) for τ in taus],
        "frames" => frames)
end

# A minimal JSON emitter for the fixed, shallow schema (Float64 / Int / String / Bool /
# array / Dict only) — keeps the package's zero-extra-dependency property. Keys are sorted
# for reproducible bytes.
function _emit_json(io::IO, x)
    if x isa AbstractString
        print(io, '"')
        for ch in x
            ch == '"' ? print(io, "\\\"") :
            ch == '\\' ? print(io, "\\\\") : print(io, ch)
        end
        print(io, '"')
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa Real
        y = Float64(x)
        isfinite(y) || throw(ArgumentError("non-finite value $y cannot be JSON-encoded"))
        print(io, y)
    elseif x isa AbstractDict
        print(io, '{')
        for (i, k) in enumerate(sort(collect(keys(x))))
            i > 1 && print(io, ',')
            _emit_json(io, String(k))
            print(io, ':')
            _emit_json(io, x[k])
        end
        print(io, '}')
    elseif x isa AbstractVector
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ',')
            _emit_json(io, v)
        end
        print(io, ']')
    else
        throw(ArgumentError("unsupported JSON type $(typeof(x))"))
    end
    return io
end

"""
    write_mfa_distributions(path, sampler, crystal; taus, npoints = 1000) -> path

Write the per-atom MFA probability distributions over `taus` to a single self-describing
JSON file (schema `scetools/mfa-distributions`, version 1) for the Python viewer. The file
carries the structure (lattice vectors as rows, Cartesian positions, species, reference
axes), the shared Fibonacci render grid and its tesseral basis matrix `Z`, and one frame
per τ holding the per-atom magnetizations `m_a` and coefficient vectors `c_a`. The viewer
recovers each atom's density as `exp(−Z·c_a)` (normalized). Requires
`size(sampler.reference, 2) == n_atoms(crystal)`.
"""
function write_mfa_distributions(path::AbstractString, sampler::MFASampler, crystal::Crystal;
                                 taus::AbstractVector{<:Real}, npoints::Integer = 1000)
    doc = _distribution_doc(sampler, crystal; taus = taus, npoints = npoints)
    open(path, "w") do io
        _emit_json(io, doc)
        print(io, '\n')
    end
    return path
end
