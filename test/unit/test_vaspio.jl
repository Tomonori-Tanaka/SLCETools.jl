using Test
using SCEFitting
using SCETools
using SCETools.VASP: read_poscar, write_poscar, Oszicar
using LinearAlgebra
using StaticArrays

_write(dir, name, s) = (p = joinpath(dir, name); write(p, s); p)

# A constrained-noncollinear OSZICAR with two atoms, parametrized moments/fields.
function _oszicar_text(; energy_free = "-.84314080E+02", energy_zero = "-.84200000E+02",
                       mw = [(1.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
                       mint = [(1.1, 0.0, 0.0), (0.0, 0.0, 2.1)],
                       field = [(0.0, 0.02, 0.0), (0.03, 0.0, 0.0)])
    io = IOBuffer()
    println(io, " ion                    MW_int                       M_int")
    for i = 1:length(mw)
        println(io, "   $i  $(mw[i][1]) $(mw[i][2]) $(mw[i][3])  $(mint[i][1]) $(mint[i][2]) $(mint[i][3])")
    end
    println(io, " lambda*MW_perp")
    for i = 1:length(field)
        println(io, "   $i  $(field[i][1]) $(field[i][2]) $(field[i][3])")
    end
    println(io, "   1 F= $energy_free E0= $energy_zero  d E =-.272117E-06")
    return String(take!(io))
end

@testset "VASP I/O" begin
    dir = mktempdir()

    @testset "POSCAR — VASP5 Direct" begin
        p = _write(dir, "POSCAR5",
            "FePt\n1.0\n 3.0 0.0 0.0\n 0.0 3.0 0.0\n 0.0 0.0 4.0\nFe Pt\n1 1\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n")
        c = read_poscar(p)
        @test n_atoms(c) == 2
        @test c.species == [1, 2]
        @test c.species_labels == ["Fe", "Pt"]
        @test c.lattice.vectors == SMatrix{3,3,Float64}([3 0 0; 0 3 0; 0 0 4])  # columns = vectors
        @test c.frac_positions[:, 1] ≈ [0.0, 0.0, 0.0]
        @test c.frac_positions[:, 2] ≈ [0.5, 0.5, 0.5]
    end

    @testset "POSCAR — Cartesian converts to fractional" begin
        p = _write(dir, "POSCAR_cart",
            "c\n1.0\n 3.0 0.0 0.0\n 0.0 3.0 0.0\n 0.0 0.0 3.0\nFe\n1\nCartesian\n 1.5 1.5 1.5\n")
        c = read_poscar(p)
        @test c.frac_positions[:, 1] ≈ [0.5, 0.5, 0.5]
    end

    @testset "POSCAR — VASP4 (no symbol line) synthesizes labels" begin
        p = _write(dir, "POSCAR4",
            "no symbols\n1.0\n 3.0 0.0 0.0\n 0.0 3.0 0.0\n 0.0 0.0 3.0\n2 1\nDirect\n 0 0 0\n 0.5 0.5 0.5\n 0.25 0.25 0.25\n")
        c = read_poscar(p)
        @test c.species_labels == ["X1", "X2"]
        @test c.species == [1, 1, 2]
    end

    @testset "POSCAR — negative scale = target volume" begin
        p = _write(dir, "POSCAR_vol",
            "v\n-27.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\nFe\n1\nDirect\n 0 0 0\n")
        c = read_poscar(p)
        @test abs(det(c.lattice.vectors)) ≈ 27.0      # cell volume set to |scale|
        @test c.lattice.vectors == SMatrix{3,3,Float64}(3.0 * I)
    end

    @testset "POSCAR — Selective dynamics line is skipped" begin
        p = _write(dir, "POSCAR_sd",
            "sd\n1.0\n 3 0 0\n 0 3 0\n 0 0 3\nFe\n1\nSelective dynamics\nDirect\n 0.1 0.2 0.3 T T F\n")
        c = read_poscar(p)
        @test c.frac_positions[:, 1] ≈ [0.1, 0.2, 0.3]   # flags after xyz are ignored
    end

    @testset "POSCAR — write/read round-trip" begin
        c = Crystal(Lattice([3.1 0.2 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    [0.0 0.5 0.1; 0.0 0.5 0.7; 0.0 0.5 0.2], [1, 2, 1], ["Fe", "Pt"])
        p = joinpath(dir, "POSCAR_rt")
        write_poscar(p, c)
        c2 = read_poscar(p)
        @test c2.lattice.vectors ≈ c.lattice.vectors
        # atoms are grouped by species on write; compare as sets per species
        @test sort(c2.species) == sort(c.species)
        @test c2.species_labels == c.species_labels
        # the two Fe and one Pt fractional coords survive (modulo grouping order)
        orig = Set(eachcol(c.frac_positions))
        @test all(any(p2 ≈ p1 for p1 in orig) for p2 in eachcol(c2.frac_positions))
    end

    @testset "POSCAR — errors" begin
        @test_throws ArgumentError read_poscar(joinpath(dir, "nonexistent"))
        @test_throws ArgumentError read_poscar(_write(dir, "short", "too\nshort\n"))
    end

    @testset "OSZICAR — energy, moments, field, torque" begin
        p = _write(dir, "OSZICAR1", _oszicar_text())
        d = read_configs(Oszicar(p))[1]
        @test d isa SpinDatum
        @test d.energy ≈ -84.314080                       # :free → the F= value
        @test d.magmoms ≈ [1.0, 2.0]                      # ‖MW_int‖
        @test d.directions[:, 1] ≈ [1.0, 0.0, 0.0]
        @test d.directions[:, 2] ≈ [0.0, 0.0, 1.0]
        @test d.field[:, 1] ≈ [0.0, 0.02, 0.0]
        @test d.field[:, 2] ≈ [0.03, 0.0, 0.0]
        # torque target τ = m × B  (physical / Landau–Lifshitz torque)
        @test d.torques[:, 1] ≈ cross(SVector(1.0, 0, 0), SVector(0.0, 0.02, 0))
        @test d.torques[:, 2] ≈ cross(SVector(0.0, 0, 2), SVector(0.03, 0, 0))
    end

    @testset "OSZICAR — energy_kind and mint" begin
        p = _write(dir, "OSZICAR2", _oszicar_text())
        @test read_configs(Oszicar(p; energy_kind = :sigma0))[1].energy ≈ -84.200000
        dmint = read_configs(Oszicar(p; mint = true))[1]
        @test dmint.magmoms ≈ [1.1, 2.1]                  # M_int columns
    end

    @testset "OSZICAR — SAXIS rotates moments/fields into the Cartesian frame" begin
        p = _write(dir, "OSZICAR3", _oszicar_text())
        d0 = read_configs(Oszicar(p))[1]                                  # saxis = ẑ → identity
        dx = read_configs(Oszicar(p; saxis = [1.0, 0.0, 0.0]))[1]
        R = SMatrix{3,3,Float64}([0 0 1; 0 1 0; -1 0 0])                  # Rz(0)·Ry(π/2)
        @test dx.directions[:, 1] ≈ R * d0.directions[:, 1]
        @test dx.field[:, 2] ≈ R * d0.field[:, 2]
        @test dx.torques[:, 1] ≈ cross(SVector(dx.magmoms[1] * dx.directions[:, 1]...),
                                       SVector(dx.field[:, 1]...))
    end

    @testset "OSZICAR — multiple files, missing field, errors" begin
        p1 = _write(dir, "OSZICAR_a", _oszicar_text())
        p2 = _write(dir, "OSZICAR_b", _oszicar_text(energy_free = "-.10000000E+02"))
        ds = read_configs(Oszicar([p1, p2]))
        @test length(ds) == 2
        @test ds[2].energy ≈ -10.0
        # no constraint block → zero field → zero torque
        nofield = " ion   MW_int   M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n   1 F= -.5E+01 E0= -.5E+01 d E = 0\n"
        dn = read_configs(Oszicar(_write(dir, "OSZICAR_nf", nofield)))[1]
        @test all(iszero, dn.field)
        @test all(iszero, dn.torques)
        @test_throws ArgumentError read_configs(Oszicar(joinpath(dir, "nope")))
        @test_throws ArgumentError read_configs(Oszicar(_write(dir, "noE",
            " ion MW_int M_int\n   1 1.0 0.0 0.0 1.0 0.0 0.0\n")))   # no F= line
    end

    @testset "SpinDatum — direct construction, zero-moment placeholder" begin
        moments = [1.0 0.0; 0.0 0.0; 0.0 0.0]              # atom 2 has zero moment
        field = [0.0 0.0; 0.5 0.0; 0.0 0.0]
        d = SpinDatum(-1.0, moments, field)
        @test d.magmoms ≈ [1.0, 0.0]
        @test d.directions[:, 2] ≈ [0.0, 0.0, 1.0]        # placeholder ẑ for the null moment
        @test d.torques[:, 2] ≈ [0.0, 0.0, 0.0]
        @test_throws ArgumentError SpinDatum(0.0, [1.0; 2.0;;], [1.0; 2.0;;])   # not 3×n
    end

    @testset "DFT-source seam → SCEDataset (code-agnostic)" begin
        c = read_poscar(_write(dir, "POSCAR_ds",
            "FeFe\n1.0\n 3 0 0\n 0 3 0\n 0 0 3\nFe\n2\nDirect\n 0 0 0\n 0.5 0 0\n"))
        basis = SCEBasis(c, BasisSpec(; nbody = 2, cutoff = 2.0, lmax = [2], isotropy = false))
        src = Oszicar([_write(dir, "OSZICAR_d1", _oszicar_text()),
                       _write(dir, "OSZICAR_d2", _oszicar_text(energy_free = "-.80000000E+02"))])
        data = read_configs(src)
        ds = SCEDataset(basis, src)                         # straight from the source
        @test has_torque(ds)
        @test ds.y_E ≈ [d.energy for d in data]
        @test ds.configs[1] == data[1].directions           # configs are the spin directions
        ds_e = SCEDataset(basis, data; use_torque = false)
        @test !has_torque(ds_e)
        @test ds_e.y_E ≈ ds.y_E
    end

    @testset "edge cases and the zero-torque guard" begin
        # multi-ionic-step OSZICAR: the last block / last F= win
        multi = " ion MW_int M_int\n  1 9 0 0  9 0 0\n  2 0 0 9  0 0 9\n" *
                " lambda*MW_perp\n  1 0 9 0\n  2 9 0 0\n  1 F= -.10E+02 E0= -.10E+02 d E=0\n" *
                _oszicar_text()
        dlast = read_configs(Oszicar(_write(dir, "OSZICAR_multi", multi)))[1]
        @test dlast.energy ≈ -84.314080            # the final step, not -10
        @test dlast.magmoms ≈ [1.0, 2.0]

        # F= present but no M_int block → error
        @test_throws ArgumentError read_configs(Oszicar(_write(dir, "OSZICAR_noM",
            "   1 F= -.5E+01 E0= -.5E+01 d E = 0\n")))

        # POSCAR truncated right after 'Selective dynamics', and an invalid coord mode
        @test_throws ArgumentError read_poscar(_write(dir, "POSCAR_trunc",
            "x\n1.0\n 3 0 0\n 0 3 0\n 0 0 3\nFe\n1\nSelective dynamics\n"))
        @test_throws ArgumentError read_poscar(_write(dir, "POSCAR_badmode",
            "x\n1.0\n 3 0 0\n 0 3 0\n 0 0 3\nFe\n1\nBanana\n 0 0 0\n"))

        # write(Cartesian) → read round-trip
        c = Crystal(Lattice([3.0 0 0; 0 3 0; 0 0 3]), [0.1 0.5; 0.2 0.5; 0.3 0.5], [1, 1], ["Fe"])
        pc = joinpath(dir, "POSCAR_cart_rt")
        write_poscar(pc, c; cartesian = true)
        @test read_poscar(pc).frac_positions ≈ c.frac_positions

        # unconstrained data (zero field) → zero torque → use_torque=true is rejected
        basis = SCEBasis(c, BasisSpec(; nbody = 2, cutoff = 2.0, lmax = [1], isotropy = true))
        uc = read_configs(Oszicar(_write(dir, "OSZICAR_uc",
            _oszicar_text(field = [(0.0, 0, 0), (0.0, 0, 0)]))))
        @test all(t -> all(iszero, t), [d.torques for d in uc])
        @test_throws ArgumentError SCEDataset(basis, uc; use_torque = true)
        @test !has_torque(SCEDataset(basis, uc; use_torque = false))
    end
    @testset "Oszicar rejects an unknown energy_kind" begin
        @test_throws ArgumentError Oszicar(["nonexistent"]; energy_kind = :bogus)
    end
end
