# VASP input writing (src/io/vasp.jl, module SCETools.VASP): sampled spin directions →
# constrained-noncollinear INCAR / full input sets. Gates: MAGMOM = magmom · direction with the
# right formatting and atom order (matching write_poscar's species grouping), template
# passthrough, the magnitude sources, the SAXIS frame round-trip, and the constraint block.

using Test
using MagestyRebuild
using SCETools
using LinearAlgebra
using Random

const V = SCETools.VASP

# Read a tag's flat float list back from a written INCAR (honours the `n*v` repeat syntax).
function incar_floats(path, key)
    for line in eachline(path)
        bare = strip(replace(line, r"[#!].*$" => ""))
        m = match(r"^\s*([A-Za-z_0-9]+)\s*=(.*)$", bare)
        m !== nothing && uppercase(m.captures[1]) == key && return V._parse_floats(m.captures[2])
    end
    return nothing
end

incar_has(path, key) = incar_floats(path, key) !== nothing ||
    any(startswith(strip(replace(l, r"[#!].*$" => "")), uppercase(key) * " ") ||
        occursin(Regex("^\\s*" * key * "\\s*="), strip(l)) for l in eachline(path))

_unit(v) = v / norm(v)

@testset "VASP input writing" begin
    rng = MersenneTwister(7)
    # A 3-atom cell, two species in a non-trivial order so write_poscar must regroup.
    lat = Lattice([4.0 0 0; 0 4.0 0; 0 0 4.0])
    cr  = Crystal(lat, [0.0 0.25 0.5; 0.0 0.25 0.5; 0.0 0.25 0.5], [2, 1, 2], ["A", "B"])
    # model-order directions (unit columns)
    cfg = mapreduce(_ -> _unit(randn(rng, 3)), hcat, 1:3)

    @testset "write_incar: MAGMOM = magmom · direction, %.9f, double-spaced" begin
        p = tempname()
        V.write_incar(p, cfg; magmoms = [3.0, 1.0, 3.0], constrain = false)
        mag = incar_floats(p, "MAGMOM")
        @test length(mag) == 9
        M = reshape(mag, 3, 3)
        for a = 1:3
            @test M[:, a] ≈ [3.0, 1.0, 3.0][a] .* cfg[:, a] atol = 1e-9
        end
        @test incar_floats(p, "M_CONSTR") === nothing       # constrain = false ⇒ no M_CONSTR
        @test occursin(r"-?\d+\.\d{9}\b", read(p, String))  # nine-decimal formatting
    end

    @testset "constrain ⇒ M_CONSTR equals MAGMOM + I_CONSTRAINED_M / LAMBDA" begin
        p = tempname()
        V.write_incar(p, cfg; magmoms = 2.5, constrain = true, i_constrained_m = 1, lambda = 5.0)
        @test incar_floats(p, "MAGMOM") ≈ incar_floats(p, "M_CONSTR")
        @test incar_floats(p, "I_CONSTRAINED_M") == [1.0]
        @test incar_floats(p, "LAMBDA") == [5.0]
        @test occursin("LNONCOLLINEAR = .TRUE.", read(p, String))
        # scalar magmoms ⇒ uniform magnitude
        @test reshape(incar_floats(p, "MAGMOM"), 3, 3)[:, 1] ≈ 2.5 .* cfg[:, 1] atol = 1e-9
    end

    @testset "template passthrough: other tags kept, magnitudes from template MAGMOM" begin
        base = """
        SYSTEM = my run
        ENCUT = 520
        ISMEAR = -5
        LNONCOLLINEAR = .TRUE.
        I_CONSTRAINED_M = 1
        RWIGS = 1.30 1.30
        LAMBDA = 10.0
        MAGMOM = 0.0 0.0 3.0  0.0 0.0 1.0  0.0 0.0 3.0
        """
        p = tempname()
        V.write_incar(p, cfg; base = base, constrain = true)
        txt = read(p, String)
        @test occursin("ENCUT = 520", txt)                  # preserved verbatim
        @test occursin("RWIGS = 1.30 1.30", txt)
        @test occursin("I_CONSTRAINED_M = 1", txt)
        # magnitudes are the per-atom norms of the template MAGMOM (3, 1, 3)
        M = reshape(incar_floats(p, "MAGMOM"), 3, 3)
        for a = 1:3
            @test norm(M[:, a]) ≈ [3.0, 1.0, 3.0][a] atol = 1e-9
            @test M[:, a] ≈ [3.0, 1.0, 3.0][a] .* cfg[:, a] atol = 1e-9
        end
        @test incar_floats(p, "M_CONSTR") ≈ incar_floats(p, "MAGMOM")   # overwritten, not duplicated
        @test count(l -> occursin("MAGMOM", l), collect(eachline(p))) == 1
    end

    @testset "template without I_CONSTRAINED_M warns under constrain" begin
        base = "ENCUT = 400\nMAGMOM = 0 0 2  0 0 2  0 0 2\n"
        p = tempname()
        @test_logs (:warn,) match_mode = :any V.write_incar(p, cfg; base = base, constrain = true)
    end

    @testset "SAXIS frame: write rotates out, the reader's rotation recovers the moments" begin
        p = tempname()
        saxis = (1.0, 0.0, 1.0)
        V.write_incar(p, cfg; magmoms = [2.0, 2.0, 2.0], constrain = false, saxis = saxis)
        Mframe = reshape(incar_floats(p, "MAGMOM"), 3, 3)
        R = V._saxis_rotation(saxis)                          # reader: SAXIS frame → Cartesian
        for a = 1:3
            @test (R * Mframe[:, a]) ≈ 2.0 .* cfg[:, a] atol = 1e-9
        end
        @test V._parse_floats(match(r"SAXIS\s*=(.*)", read(p, String)).captures[1]) ≈ [1, 0, 1]
    end

    @testset "SAXIS taken from the template frame (no kwarg needed)" begin
        base = "SAXIS = 1.0 0.0 1.0\nMAGMOM = 0 0 2  0 0 2  0 0 2\nI_CONSTRAINED_M = 1\n"
        p = tempname()
        V.write_incar(p, cfg; base = base, constrain = false)   # no saxis kwarg → use template's
        Mframe = reshape(incar_floats(p, "MAGMOM"), 3, 3)
        R = V._saxis_rotation((1.0, 0.0, 1.0))
        for a = 1:3
            @test (R * Mframe[:, a]) ≈ 2.0 .* cfg[:, a] atol = 1e-9   # magnitude 2 from template
        end
        @test count(l -> occursin("SAXIS", l), collect(eachline(p))) == 1   # not duplicated
    end

    @testset "write_inputs: POSCAR + INCAR consistent, atom order grouped by species" begin
        dir = mktempdir()
        V.write_inputs(dir, cr, cfg; magmoms = Dict("A" => 3.0, "B" => 1.0), constrain = true)
        @test isfile(joinpath(dir, "POSCAR"))
        @test isfile(joinpath(dir, "INCAR"))

        # POSCAR is regrouped by species label order (A = species 1? no: labels ["A","B"],
        # species [2,1,2] ⇒ B (species-index 1) is atom 2; A (species-index 2) is atoms 1,3).
        reloaded = V.read_poscar(joinpath(dir, "POSCAR"))
        perm = V._poscar_order(cr)                            # POSCAR atom k ← model atom perm[k]
        permmag = [Dict("A" => 3.0, "B" => 1.0)[cr.species_labels[cr.species[a]]] for a in perm]
        M = reshape(incar_floats(joinpath(dir, "INCAR"), "MAGMOM"), 3, num_atoms(cr))
        for k = 1:num_atoms(cr)
            @test M[:, k] ≈ permmag[k] .* cfg[:, perm[k]] atol = 1e-9
        end
    end

    @testset "magnitude sources agree (per-atom vector vs per-species map)" begin
        d1 = mktempdir(); d2 = mktempdir()
        # species [2,1,2] over labels ["A","B"] ⇒ atoms are B, A, B in model order.
        permatom = [Dict("A" => 3.0, "B" => 1.0)[cr.species_labels[cr.species[a]]] for a = 1:3]
        V.write_inputs(d1, cr, cfg; magmoms = permatom, constrain = false)             # per-atom
        V.write_inputs(d2, cr, cfg; magmoms = Dict("A" => 3.0, "B" => 1.0), constrain = false)  # per-species
        @test incar_floats(joinpath(d1, "INCAR"), "MAGMOM") ≈
              incar_floats(joinpath(d2, "INCAR"), "MAGMOM")
    end

    @testset "sweep: one subdirectory per configuration" begin
        root = mktempdir()
        configs = [mapreduce(_ -> _unit(randn(rng, 3)), hcat, 1:3) for _ = 1:4]
        dirs = V.write_inputs(root, cr, configs; magmoms = Dict("A" => 3.0, "B" => 1.0), prefix = "s")
        @test length(dirs) == 4
        @test all(isdir, dirs)
        @test all(d -> isfile(joinpath(d, "POSCAR")) && isfile(joinpath(d, "INCAR")), dirs)
        @test basename(dirs[1]) == "s-001"
    end

    @testset "errors" begin
        p = tempname()
        @test_throws ArgumentError V.write_incar(p, cfg)            # no magmoms, no template
        @test_throws ArgumentError V.write_incar(p, cfg; magmoms = [1.0, 2.0])   # wrong length
        @test_throws ArgumentError V.write_incar(p, cfg; magmoms = Dict("A" => 1.0))  # needs crystal
        @test_throws ArgumentError V.write_incar(p, cfg; magmoms = [1.0, -2.0, 1.0])  # negative magnitude
        @test_throws ArgumentError V.write_incar(p, cfg; magmoms = 1.0, base = "/no/such/INCAR.template")  # bad path
        @test_throws ArgumentError V.write_inputs(mktempdir(), cr, cfg;
                                                  magmoms = Dict("A" => 3.0))    # missing "B"
        @test_throws ArgumentError V.write_inputs(mktempdir(), cr, cfg[:, 1:2];
                                                  magmoms = 1.0)                 # wrong atom count
    end
end
