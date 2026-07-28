# VASP → SLCE end-to-end through the code-agnostic DFT-source seam: read a structure
# from a POSCAR, read constrained-noncollinear OSZICARs into `TrainingDatum`s, and fit.
# The point: the SLCE pipeline only ever sees `TrainingDatum` / `SLCEDataset`, so the
# originating DFT code is irrelevant — a different code is just a different source.
#
# (The POSCAR/OSZICAR files here are written synthetically as stand-ins for real VASP
# output, so the fit quality is not meaningful; the flow and the I/O are.)
#
# Run:  julia --project=examples examples/vasp_dft_source.jl

using SLCE
using SLCETools
using SLCETools.VASP: read_poscar, Oszicar
using LinearAlgebra
using Random

dir = mktempdir()

# --- a structure file (stands in for a real POSCAR/CONTCAR) ----------------------
write(joinpath(dir, "POSCAR"),
      "Fe2\n1.0\n 2.5 0 0\n 0 2.5 0\n 0 0 2.5\nFe\n2\nDirect\n 0 0 0\n 0.5 0.5 0.5\n")
crystal = read_poscar(joinpath(dir, "POSCAR"))
println("read_poscar → ", n_atoms(crystal), " atoms, species ", crystal.species_labels)

# --- a few constrained-noncollinear OSZICARs (stand-ins for real runs) -----------
function write_oszicar(path; energy, moments, field)
    open(path, "w") do io
        println(io, " ion   MW_int   M_int")
        for i in axes(moments, 2)
            m = moments[:, i]
            println(io, "   $i  $(m[1]) $(m[2]) $(m[3])  $(m[1]) $(m[2]) $(m[3])")
        end
        println(io, " lambda*MW_perp")
        for i in axes(field, 2)
            b = field[:, i]
            println(io, "   $i  $(b[1]) $(b[2]) $(b[3])")
        end
        println(io, "   1 F= $energy E0= $energy  d E = 0")
    end
end

rng = MersenneTwister(7)
randdir() = (v = randn(rng, 3); v / norm(v))
paths = String[]
for k = 1:8
    moments = reduce(hcat, (randdir() for _ = 1:2))     # 3×2 unit moments
    field = 0.01 .* randn(rng, 3, 2)                    # small constraining field
    p = joinpath(dir, "OSZICAR_$k")
    write_oszicar(p; energy = -80.0 - 0.01k, moments = moments, field = field)
    push!(paths, p)
end

# --- the code-agnostic seam: VASP source → TrainingDatum → dataset → fit ---------
src = Oszicar(paths)                                     # an AbstractDFTSource
data = read_configs(src)                                 # Vector{TrainingDatum}
println("read_configs → ", length(data), " configurations")
d = data[1]
println("  config 1: energy = ", d.energy, " eV, |m| = ", round.(d.magmoms; digits = 3))
println("  torque target τ_a = m_a×B_a (eV), atom 1 = ", round.(d.torques[:, 1]; digits = 5))

basis = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 2.5, lmax = [1], soc = false))
dataset = SLCEDataset(basis, src)                         # source → dataset (read under the hood)
f = fit(SLCEFit, dataset, OLS(); torque_weight = 0.3)
println("\nfit completed on ", nobs(f), " configs, ", n_salcs(basis), " coefficient(s)")
println("✓ the SLCE pipeline saw only TrainingDatum / SLCEDataset — the VASP origin never entered it")
