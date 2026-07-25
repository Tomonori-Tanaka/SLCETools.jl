# Per-atom MFA probability distribution export (src/viz/distributions.jl). Validates the
# shared Fibonacci grid, the discrete normalization of `site_probabilities`, that the
# exported basis matrix `Z·c` reproduces `site_potential` exactly (the viewer's render
# path), the isotropic vMF closed form, the `m_a` consistency with `mfa_sublattice_m`, the
# tensorial (Bingham) path, and the JSON document shape / round-trip.

using Test
using SLCE
using SLCETools
import JSON
using SLCETools: fibonacci_sphere, harmonic_basis, site_probabilities   # public, unexported
using SLCETools.VASP: read_poscar
using LinearAlgebra
using StaticArrays

const MV = SLCETools

# A 2-atom Fe POSCAR matching the (1,2) sublattice samplers below.
function _fe2_crystal()
    dir = mktempdir()
    write(joinpath(dir, "POSCAR"),
          "Fe2\n1.0\n 2.5 0 0\n 0 2.5 0\n 0 0 2.5\nFe\n2\nDirect\n 0 0 0\n 0.5 0.5 0.5\n")
    return read_poscar(joinpath(dir, "POSCAR"))
end

@testset "MFA distribution export" begin
    @testset "fibonacci_sphere: unit dirs, 4π weight closure" begin
        grid = fibonacci_sphere(1000)
        @test length(grid.dirs) == 1000
        @test all(d -> abs(norm(d) - 1) < 1e-12, grid.dirs)
        @test grid.weight * 1000 ≈ 4π
        @test_throws ArgumentError fibonacci_sphere(0)
    end

    @testset "harmonic_basis: Z·c == site_potential, l=0 column is zero" begin
        grid = fibonacci_sphere(500)
        Z = harmonic_basis(grid, 2)
        @test size(Z) == (500, 9)
        @test all(Z[:, MV.Harmonics.lm_index(0, 0)] .== 0)   # l=0 column zeroed
        c = collect(1.0:9.0)                                  # arbitrary coefficient vector
        V = Z * c                                             # BLAS accumulation order differs
        @test maximum(abs(V[i] - MV.site_potential(c, grid.dirs[i])) for i = 1:500) < 1e-13
    end

    @testset "site_probabilities: discrete normalization Σ p·weight = 1" begin
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        grid = fibonacci_sphere(2000)
        for τ in (0.2, 0.5, 0.9)
            P = site_probabilities(mfa_site_coefficients(s, τ), grid)
            for a in axes(P, 1)
                @test sum(@view P[a, :]) * grid.weight ≈ 1 atol = 1e-12
            end
        end
    end

    @testset "isotropic path: vMF closed form and m_a consistency" begin
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        τ = 0.5
        field = mfa_site_coefficients(s, τ)
        @test field.lmax == 1
        @test field.m == mfa_sublattice_m(s, τ)               # same solver, exact

        grid = fibonacci_sphere(4000)
        P = site_probabilities(field, grid)
        _, κ, _ = MV._mfa_state(s, τ)
        ehat = SVector(0.0, 0.0, 1.0)
        # analytic vMF density κ/(4π sinh κ)·exp(κ ê·e) on the same grid points
        analytic = [κ[1] / (4π * sinh(κ[1])) * exp(κ[1] * dot(ehat, grid.dirs[i]))
                    for i = 1:length(grid.dirs)]
        @test maximum(abs.(@view(P[1, :]) .- analytic)) < 1e-5
        # grid estimate of m_a = ⟨e·ê_a⟩ matches within grid error
        mest = sum(P[1, i] * grid.weight * dot(ehat, grid.dirs[i]) for i = 1:length(grid.dirs))
        @test mest ≈ field.m[1] atol = 1e-4
    end

    @testset "tensorial path: lmax 2, coeffs match solver, non-axisymmetric" begin
        A = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -2)  # easy-axis along z
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]; onsite = [A, A]);
                       reference = Float64[0 0; 0 0; 1 1])
        τ = 0.5
        field = mfa_site_coefficients(s, τ)
        @test field.lmax == 2
        cs, m = MV._coeffs_and_m(s, τ)
        @test field.coeffs == cs
        @test field.m == m
        # the l=2 part is present (single-ion anisotropy ⇒ non-vMF)
        @test any(abs(field.coeffs[1][k]) > 1e-8 for k = 5:9)
    end

    @testset "ordered limit is clamped (finite, normalized)" begin
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        field = mfa_site_coefficients(s, 1e-7)                # τ < _MFA_MIN_TAU ⇒ κ = Inf
        @test all(isfinite, field.coeffs[1])
        grid = fibonacci_sphere(2000)
        P = site_probabilities(field, grid)
        @test all(isfinite, P)
        @test sum(@view P[1, :]) * grid.weight ≈ 1 atol = 1e-12
    end

    @testset "document builder: keys, shapes, conventions" begin
        crystal = _fe2_crystal()
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        taus = [0.3, 0.6, 0.9]
        doc = MV._distribution_doc(s, crystal; taus = taus, npoints = 300)

        @test doc["schema"] == "scetools/mfa-distributions"
        @test doc["version"] == 1
        @test doc["lmax"] == 1
        @test length(doc["lattice_vectors"]) == 3
        @test doc["lattice_vectors"][1] == [2.5, 0.0, 0.0]    # rows are lattice vectors (columns of A)
        @test length(doc["positions_cartesian"]) == 2
        @test doc["positions_cartesian"][2] ≈ [1.25, 1.25, 1.25]   # 0.5·2.5 along each axis
        @test doc["species_labels"] == ["Fe"]
        @test length(doc["reference"]) == 2
        @test doc["reference"][1] == [0.0, 0.0, 1.0]          # per-atom rows [x,y,z]
        @test doc["grid"]["npoints"] == 300
        @test length(doc["grid"]["directions"]) == 300
        @test length(doc["grid"]["Z"]) == 300
        @test length(doc["grid"]["Z"][1]) == 4               # (lmax+1)² = 4
        @test length(doc["frames"]) == 3
        @test doc["temperatures"] == taus
        fr = doc["frames"][2]
        @test fr["tau"] == 0.6
        @test length(fr["coeffs"]) == 2
        @test length(fr["coeffs"][1]) == 4
        @test length(fr["m"]) == 2

        @test_throws ArgumentError MV._distribution_doc(s, crystal; taus = Float64[], npoints = 100)
    end

    @testset "write_mfa_distributions: round-trips through JSON" begin
        crystal = _fe2_crystal()
        s = MFASampler(ExchangeModel([0.0 -1.0; -1.0 0.0]); reference = Float64[0 0; 0 0; 1 1])
        path = joinpath(mktempdir(), "fe2_mfa.json")
        out = write_mfa_distributions(path, s, crystal; taus = [0.4, 0.8], npoints = 200)
        @test out == path
        @test isfile(path)
        txt = read(path, String)
        # a real JSON parser must accept the hand-rolled emitter's output, and the parsed
        # document must carry the schema fields (occursin/startswith alone cannot catch a
        # malformed emitter)
        doc = JSON.parse(txt)
        @test doc["schema"] == "scetools/mfa-distributions"
        @test doc["version"] == 1
        @test length(doc["frames"]) == 2
        @test doc["temperatures"] == [0.4, 0.8]
        @test length(doc["frames"][1]["coeffs"]) == 2
        @test all(isfinite, reduce(vcat, doc["frames"][1]["coeffs"]))
        # the basis-matrix recovery V = Z·c must reproduce site_potential for the viewer
        field = mfa_site_coefficients(s, 0.4)
        grid = fibonacci_sphere(200)
        Z = harmonic_basis(grid, field.lmax)
        V = Z * field.coeffs[1]
        @test maximum(abs(V[i] - MV.site_potential(field.coeffs[1], grid.dirs[i]))
                      for i = 1:200) < 1e-13
    end

    @testset "_emit_json escapes control characters (RFC 8259)" begin
        buf = IOBuffer()
        MV._emit_json(buf, "a\"b\\c\nd\te")
        @test String(take!(buf)) == "\"a\\\"b\\\\c\\nd\\te\""
        buf = IOBuffer()
        MV._emit_json(buf, "x\x01y")                  # a bare control char → 
        @test String(take!(buf)) == "\"x\\u0001y\""
    end
    @testset "_emit_json / document guards reject bad input loudly" begin
        @test_throws ArgumentError MV._emit_json(IOBuffer(), NaN)      # non-finite number
        @test_throws ArgumentError MV._emit_json(IOBuffer(), Inf)
        @test_throws ArgumentError MV._emit_json(IOBuffer(), :sym)     # unsupported type
        # crystal / sampler atom-count mismatch is refused at the document builder
        crystal = _fe2_crystal()                                        # 2 atoms
        s1 = MFASampler(Float64[0; 0; 1;;])                             # 1 atom
        @test_throws ArgumentError MV._distribution_doc(s1, crystal; taus = [0.5],
                                                        npoints = 100)
    end
end
