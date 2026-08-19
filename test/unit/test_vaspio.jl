using Test
using SLCE
using SLCETools
using SLCETools.VASP: read_poscar, write_poscar, Oszicar
using LinearAlgebra
using StaticArrays

_write(dir, name, s) = (p = joinpath(dir, name); write(p, s); p)

# A constrained-noncollinear OSZICAR with two atoms, parametrized moments/fields.
function _oszicar_text(; energy_free = "-.84314080E+02", energy_zero = "-.84200000E+02",
                       mw = [(1.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
                       mint = [(1.1, 0.0, 0.0), (0.0, 0.0, 2.1)],
                       field = [(0.0, 0.02, 0.0), (0.03, 0.0, 0.0)],
                       ep = nothing)
    io = IOBuffer()
    ep === nothing || println(io, " E_p =  $ep  lambda =  0.200E+01")
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
        @test d isa TrainingDatum
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

    @testset "OSZICAR — constraint penalty E_p is subtracted (audit #20)" begin
        # Oracle: hand arithmetic on hand-written file contents — the parser never sees
        # the expected values below in any form it could echo back.
        # F = -84.314080, E0 = -84.200000, E_p = 5e-4 (below the 1e-3 default: no warn).
        p = _write(dir, "OSZICAR_ep", _oszicar_text(; ep = "0.50000E-03"))
        d = @test_logs read_configs(Oszicar(p))[1]
        @test d.energy ≈ -84.314080 - 0.0005              # F − E_p, by hand
        @test read_configs(Oszicar(p; energy_kind = :sigma0))[1].energy ≈
              -84.200000 - 0.0005                         # E0 carries the same one copy

        # Above the threshold: energy still corrected, and the deviation warning fires.
        pw = _write(dir, "OSZICAR_ep_warn", _oszicar_text(; ep = "0.25000E-01"))
        dw = @test_logs (:warn, r"E_p") match_mode = :any read_configs(Oszicar(pw))[1]
        @test dw.energy ≈ -84.314080 - 0.025
        # ep_warn = Inf silences without changing the correction …
        di = @test_logs read_configs(Oszicar(pw; ep_warn = Inf))[1]
        @test di.energy ≈ -84.314080 - 0.025
        # … and a tighter threshold flags the small penalty too.
        @test_logs (:warn, r"E_p") match_mode = :any read_configs(Oszicar(p; ep_warn = 1e-4))
        @test_throws ArgumentError Oszicar(p; ep_warn = -1.0)

        # Multi-step file: the LAST E_p (converged step) wins, like the F=/block rules.
        # the summary lines carry 8 tokens (`d E = 0`) so they terminate the 7-column
        # moment block, as in real OSZICAR output
        ms = " E_p =  0.90000E+00  lambda =  0.200E+01\n" *
             " ion MW_int M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n" *
             "   1 F= -.90E+01 E0= -.90E+01 d E = 0\n" *
             " E_p =  0.12000E-01  lambda =  0.200E+01\n" *
             " ion MW_int M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n" *
             "   1 F= -.10E+02 E0= -.10E+02 d E = 0\n"
        dm = @test_logs (:warn, r"E_p") match_mode = :any read_configs(
            Oszicar(_write(dir, "OSZICAR_ep_ms", ms)))[1]
        @test dm.energy ≈ -10.0 - 0.012                   # last F − last E_p, by hand

        # No E_p line (unconstrained run): the energy is left untouched.
        pu = _write(dir, "OSZICAR_ep_none", _oszicar_text())
        @test read_configs(Oszicar(pu))[1].energy ≈ -84.314080
    end

    @testset "OSZICAR — E_p is paired with its own step (review 2026-08-11)" begin
        # A truncated tail: step 1 completes (E_p, F=); the job dies after printing
        # step 2's first E_p but before step 2's own F=. The dangling E_p belongs to
        # no accepted energy and must not be subtracted from step 1's (the pre-review
        # reader computed -9 - 9.9 here).
        tail = " E_p =  0.90000E-02  lambda =  0.200E+01\n" *
               " ion MW_int M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n" *
               "   1 F= -.90E+01 E0= -.90E+01 d E = 0\n" *
               " E_p =  0.99000E+01  lambda =  0.200E+01\n"
        dt = @test_logs (:warn, r"E_p") match_mode = :any read_configs(
            Oszicar(_write(dir, "OSZICAR_ep_tail", tail)))[1]
        @test dt.energy ≈ -9.0 - 0.009                    # step 1's own E_p, by hand

        # A final step with no E_p lines of its own (constraint released): nothing is
        # subtracted — the earlier step's penalty must not leak forward either.
        rel = " E_p =  0.90000E+00  lambda =  0.200E+01\n" *
              " ion MW_int M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n" *
              "   1 F= -.90E+01 E0= -.90E+01 d E = 0\n" *
              " ion MW_int M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n" *
              "   1 F= -.10E+02 E0= -.10E+02 d E = 0\n"
        dr = @test_logs read_configs(Oszicar(_write(dir, "OSZICAR_ep_rel", rel)))[1]
        @test dr.energy ≈ -10.0                           # untouched, and no warning
    end

    @testset "OSZICAR — an empty final constraint block wins as zero (review 2026-08-11)" begin
        # Step 1 carries a nonzero constrained field; step 2's `lambda*MW_perp` header
        # prints but the block has no rows (constraint released, or truncation).
        # "The last block wins": the field must read computed-and-zero — the
        # pre-review reader kept step 1's stale values silently.
        two = _oszicar_text() *
              " ion                    MW_int                       M_int\n" *
              "   1  1.0 0.0 0.0  1.1 0.0 0.0\n   2  0.0 0.0 2.0  0.0 0.0 2.1\n" *
              " lambda*MW_perp\n" *
              "   1 F= -.90E+01 E0= -.90E+01 d E = 0\n"
        pth = _write(dir, "OSZICAR_empty_last_block", two)
        f = @test_logs (:warn, r"final lambda\*MW_perp") match_mode = :any (
            SLCETools.VASP._oszicar_field(pth, 2))
        @test f == zeros(3, 2)
        # a single empty block among none-with-rows keeps the original warning
        one = " ion MW_int M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n" *
              " lambda*MW_perp\n" *
              "   1 F= -.90E+01 E0= -.90E+01 d E = 0\n"
        f1 = @test_logs (:warn, r"no parseable field rows") match_mode = :any (
            SLCETools.VASP._oszicar_field(_write(dir, "OSZICAR_only_empty_block", one), 1))
        @test f1 == zeros(3, 1)
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
        # A GENERIC-azimuth anchor (α ≠ 0): every other SAXIS in the suite has
        # sy = 0, so the Rz(α) branch would otherwise never meet an independent
        # value — the writer/reader round-trips share `_saxis_rotation` and prove
        # only R·Rᵀ = I. Hand-derived from the VASP manual's m_xyz(m') formula
        # (its matrix IS Rz(α)·Ry(β), α = atan(sy, sx), β = atan(√(sx²+sy²), sz)):
        # saxis = (0, 1, 0) ⇒ α = β = π/2 ⇒
        #   Rz(π/2)·Ry(π/2) = [0 −1 0; 1 0 0; 0 0 1]·[0 0 1; 0 1 0; −1 0 0]
        #                   = [0 −1 0; 0 0 1; −1 0 0]      (ẑ′ ↦ ŷ, as it must)
        dy = read_configs(Oszicar(p; saxis = [0.0, 1.0, 0.0]))[1]
        Ry90z90 = SMatrix{3,3,Float64}([0 -1 0; 0 0 1; -1 0 0])
        @test dy.directions[:, 1] ≈ Ry90z90 * d0.directions[:, 1]
        @test dy.field[:, 2] ≈ Ry90z90 * d0.field[:, 2]
    end

    @testset "OSZICAR — multiple files, missing field, errors" begin
        p1 = _write(dir, "OSZICAR_a", _oszicar_text())
        p2 = _write(dir, "OSZICAR_b", _oszicar_text(energy_free = "-.10000000E+02"))
        ds = read_configs(Oszicar([p1, p2]))
        @test length(ds) == 2
        @test ds[2].energy ≈ -10.0
        # no constraint block at all → the field was NOT computed: nothing, not a
        # fabricated zero matrix (torque rows must not be admitted for this file)
        nofield = " ion   MW_int   M_int\n   1  1.0 0.0 0.0  1.0 0.0 0.0\n   1 F= -.5E+01 E0= -.5E+01 d E = 0\n"
        dn = read_configs(Oszicar(_write(dir, "OSZICAR_nf", nofield)))[1]
        @test dn.field === nothing
        @test dn.torques === nothing
        @test !dn.provenance.torque_qualified && !dn.provenance.constrained
        @test_throws ArgumentError read_configs(Oszicar(joinpath(dir, "nope")))
        @test_throws ArgumentError read_configs(Oszicar(_write(dir, "noE",
            " ion MW_int M_int\n   1 1.0 0.0 0.0 1.0 0.0 0.0\n")))   # no F= line
    end

    @testset "spin_datum — direct construction, zero-moment placeholder" begin
        moments = [1.0 0.0; 0.0 0.0; 0.0 0.0]              # atom 2 has zero moment
        field = [0.0 0.0; 0.5 0.0; 0.0 0.0]
        d = spin_datum(-1.0, moments, field)
        @test d.magmoms ≈ [1.0, 0.0]
        @test d.directions[:, 2] ≈ [0.0, 0.0, 1.0]        # placeholder ẑ for the null moment
        @test d.torques[:, 2] ≈ [0.0, 0.0, 0.0]
        @test_throws ArgumentError spin_datum(0.0, [1.0; 2.0;;], [1.0; 2.0;;])   # not 3×n
    end

    @testset "DFT-source seam → SLCEDataset (code-agnostic)" begin
        c = read_poscar(_write(dir, "POSCAR_ds",
            "FeFe\n1.0\n 3 0 0\n 0 3 0\n 0 0 3\nFe\n2\nDirect\n 0 0 0\n 0.5 0 0\n"))
        basis = SLCEBasis(c, BasisSpec(; nbody = 2, cutoff = 2.0, lmax = [2], soc = true))
        src = Oszicar([_write(dir, "OSZICAR_d1", _oszicar_text()),
                       _write(dir, "OSZICAR_d2", _oszicar_text(energy_free = "-.80000000E+02"))])
        data = read_configs(src)
        ds = SLCEDataset(basis, src)                         # straight from the source
        @test has_torque(ds)
        @test ds.y_E ≈ [d.energy for d in data]
        @test ds.configs[1] == data[1].directions           # configs are the spin directions
        ds_e = SLCEDataset(basis, data; use_torque = false)
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

        # an explicit all-zero constraint block → field present but zero → the
        # torque rows are unqualified, so use_torque=true is rejected
        basis = SLCEBasis(c, BasisSpec(; nbody = 2, cutoff = 2.0, lmax = [1], soc = false))
        uc = read_configs(Oszicar(_write(dir, "OSZICAR_uc",
            _oszicar_text(field = [(0.0, 0, 0), (0.0, 0, 0)]))))
        @test all(t -> all(iszero, t), [d.torques for d in uc])
        @test_throws ArgumentError SLCEDataset(basis, uc; use_torque = true)
        @test !has_torque(SLCEDataset(basis, uc; use_torque = false))
    end

    @testset "Oszicar stamps setup_id into the provenance" begin
        p = _write(dir, "OSZICAR_sid", _oszicar_text())
        d = read_configs(Oszicar(p; setup_id = "vasp-ncl-soc"))[1]
        @test d.provenance.setup_id == "vasp-ncl-soc"
        @test d.provenance.constrained && d.provenance.torque_qualified
        @test read_configs(Oszicar(p))[1].provenance.setup_id === nothing
    end

    @testset "Oszicar rejects an unknown energy_kind" begin
        @test_throws ArgumentError Oszicar(["nonexistent"]; energy_kind = :bogus)
    end
end

@testset "oszicar_to_extxyz — the VASP → extxyz generator" begin
    using SLCETools.VASP: oszicar_to_extxyz
    dir = mktempdir()
    poscar = _write(dir, "POSCAR_gen",
        "FePt\n1.0\n 3.0 0.0 0.0\n 0.0 3.0 0.0\n 0.0 0.0 4.0\nFe Pt\n1 1\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n")
    incar4 = _write(dir, "INCAR_gen4",
        "I_CONSTRAINED_M = 4\nLSORBIT = .FALSE.\nLAMBDA = 10\n" *
        "M_CONSTR = 1.0 0.0 0.0  0.0 0.0 2.0\n")
    osz = [_write(dir, "gen_$(i).oszicar", _oszicar_text(; ep = "0.50000E-03"))
           for i = 1:3]

    @testset "mode 4: metadata read from the INCAR, values survive bitwise" begin
        out = joinpath(dir, "gen4.extxyz")
        data = oszicar_to_extxyz(out, osz, poscar; incar = incar4, setup_id = "t")
        @test length(data) == 3
        @test data[1].constraint_mode == 4          # from I_CONSTRAINED_M
        @test data[1].provenance.soc == false       # from LSORBIT
        @test data[1].provenance.setup_id == "t"
        # M_CONSTR magnitudes are normalized away: axes are unit columns
        @test data[1].constraint_axes[:, 1] ≈ [1.0, 0.0, 0.0]
        @test data[1].constraint_axes[:, 2] ≈ [0.0, 0.0, 1.0]
        # E_p is subtracted exactly as the Oszicar reader does
        ref = read_configs(Oszicar(osz[1]))[1]
        @test data[1].energy == ref.energy
        @test data[1].field == ref.field
        # the file round-trips the bare channels bitwise
        back = read_extxyz(out)
        @test length(back) == 3
        @test back[1].moments_bare == data[1].moments_bare
        @test back[1].moments_bare[:, 1] ≈ [1.1, 0.0, 0.0]      # M_int, not MW
        @test back[1].field == data[1].field
        @test back[1].constraint_axes == data[1].constraint_axes
        @test back[1].constraint_mode == 4
        # source records the digest, field_sign the convention
        info = split(readlines(out)[2])
        @test any(startswith(t, "source=oszicar:3:sha256:") for t in info)
        @test any(t == "field_sign=vasp:lambda*MW_perp" for t in info)
    end

    @testset "declared mode cross-checked against the INCAR" begin
        @test_throws ArgumentError oszicar_to_extxyz(joinpath(dir, "x.extxyz"), osz,
                                                     poscar; incar = incar4,
                                                     constraint_mode = 1)
        # matching declaration passes
        data = oszicar_to_extxyz(joinpath(dir, "x.extxyz"), osz, poscar;
                                incar = incar4, constraint_mode = 4)
        @test data[1].constraint_mode == 4
    end

    @testset "mode 1 requires the INCAR (axes are not reconstructible)" begin
        @test_throws ArgumentError oszicar_to_extxyz(joinpath(dir, "y.extxyz"), osz,
                                                     poscar; constraint_mode = 1)
    end

    @testset "generation-time sign gate: a flipped bare moment never becomes a file" begin
        incar1 = _write(dir, "INCAR_gen1",
            "I_CONSTRAINED_M = 1\nLSORBIT = .FALSE.\n" *
            "M_CONSTR = 1.0 0.0 0.0  0.0 0.0 2.0\n")
        # mint flipped against mw on atom 1: y = ê_c·M < 0 while ê_MW·ê_c > 0
        bad = _write(dir, "gen_bad.oszicar",
                     _oszicar_text(; mint = [(-1.1, 0.0, 0.0), (0.0, 0.0, 2.1)]))
        out = joinpath(dir, "never.extxyz")
        err = try
            oszicar_to_extxyz(out, [bad], poscar; incar = incar1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("sign-consistency", err.msg)
        @test !isfile(out)                          # verified at birth, or not born
        # the consistent sibling generates fine under the same INCAR
        data = oszicar_to_extxyz(out, [osz[1]], poscar; incar = incar1)
        @test data[1].constraint_mode == 1
    end

    @testset "M_CONSTR length mismatch is loud" begin
        short = _write(dir, "INCAR_short",
            "I_CONSTRAINED_M = 4\nM_CONSTR = 1.0 0.0 0.0\n")
        @test_throws ArgumentError oszicar_to_extxyz(joinpath(dir, "z.extxyz"), osz,
                                                     poscar; incar = short)
    end

    @testset "SAXIS rotation matches the Oszicar reader" begin
        out = joinpath(dir, "sax.extxyz")
        sax = [1.0, 1.0, 0.5]
        data = oszicar_to_extxyz(out, osz, poscar; incar = incar4, saxis = sax)
        ref = read_configs(Oszicar(osz; saxis = sax))
        @test data[1].directions ≈ ref[1].directions
        @test data[1].field == ref[1].field
    end
end
