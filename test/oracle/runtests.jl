# Oracle: the from-scratch VASP POSCAR / OSZICAR parsers in `SLCETools.VASP` agree bit-for-bit
# with Magesty.jl's proven parsers (synthetic files, since real VASP outputs are not vendored).
# Default SAXIS = ẑ ⇒ no rotation, matching Magesty's raw parse. A separate environment carries
# the pinned Magesty.jl that the core / aux suites deliberately omit.
#
# Run:  julia --project=test/oracle test/oracle/runtests.jl

using Test
using SLCE: read_configs
using SLCETools
using Magesty

@testset "SLCETools.jl oracle" begin
    @testset "VASP POSCAR / OSZICAR parsing vs Magesty" begin
        dir = mktempdir()

        poscar = joinpath(dir, "POSCAR")
        write(poscar,
            "FePt\n1.0\n 3.1 0.2 0.0\n 0.0 3.0 0.0\n 0.0 0.0 4.0\nFe Pt\n1 1\nDirect\n 0.1 0.2 0.3\n 0.5 0.5 0.5\n")
        mr = SLCETools.VASP.read_poscar(poscar)
        mg = Magesty.VaspIO.parse_poscar(poscar)
        for i = 1:3
            @test mr.lattice.vectors[:, i] ≈ mg.lattice_vectors[i]   # column i = lattice vector i
        end
        for a = 1:size(mr.frac_positions, 2)
            @test mr.frac_positions[:, a] ≈ mg.positions[a]
        end
        kd = reduce(vcat, [fill(i, n) for (i, n) in enumerate(mg.numbers)])
        @test mr.species == kd
        @test mr.species_labels == mg.element_symbols

        osz = joinpath(dir, "OSZICAR")
        write(osz,
            " ion   MW_int   M_int\n   1  1.0 0.2 0.0  1.1 0.2 0.0\n   2  0.0 0.0 2.0  0.0 0.0 2.1\n" *
            " lambda*MW_perp\n   1  0.0 0.02 0.0\n   2  0.03 0.0 0.0\n" *
            "   1 F= -.84314080E+02 E0= -.84200000E+02  d E = 0\n")
        mrd = read_configs(SLCETools.VASP.Oszicar(osz))[1]
        mgd = Magesty.VaspIO.parse_oszicar(osz; energy_kind = "f", mint = false)
        @test mrd.energy ≈ mgd.energy
        for i = 1:length(mrd.magmoms)
            @test mrd.magmoms[i] .* mrd.directions[:, i] ≈ mgd.magmom[i, :]   # moment vector
            @test mrd.field[:, i] ≈ mgd.magfield[i, :]
        end
    end
end
