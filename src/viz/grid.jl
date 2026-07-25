# Per-atom MFA distribution viz — the shared render grid and its tesseral basis matrix.
#
# The viewer turns each atom's coefficient vector `c_a` into a per-direction density with a
# single matrix product against a shared basis matrix `Z[i,k] = Z_lm(e_i)` that this module
# evaluates once (so the harmonic convention stays owned by `SLCE.Harmonics`; Python
# never re-implements `Z_lm`). The companion compute (`mfa_site_coefficients`,
# `site_probabilities`) lives in `distributions.jl`, the JSON document in `serialize.jl`.

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
as `field_scale` in the engine): `z = 1 − 2(k+½)/N`, `φ = ga·k`, `ga = π(3−√5)`. The
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
indexed by `Harmonics.lm_index(l, m)`. The `l = 0` column (a constant shift `site_potential`
ignores) is left zero, so `Z[i, :]·c == site_potential(c, grid.dirs[i])` exactly for any
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
                Z[i, Harmonics.lm_index(l, m)] = Harmonics.Zlm_unsafe(l, m, e)   # grid is unit
            end
        end
    end
    return Z
end
