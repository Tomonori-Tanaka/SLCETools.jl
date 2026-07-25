# Per-atom MFA probability distributions → JSON for the interactive Python sphere viewer.
# Reads a structure from a POSCAR, builds a 2-atom antiferromagnetic exchange model, and
# writes the per-atom `exp(−V_a)` coefficients over a τ sweep to `mfa_distributions.json`.
# Then visualize:
#
#     julia --project=examples examples/mfa_distributions.jl [tau_min tau_max ntau [npoints]]
#     python viz/mfa_viewer.py mfa_distributions.json   # opens an interactive browser figure
#
# The τ sweep and the sphere-grid resolution are optional command-line arguments
# (defaults: 0.2 0.95 8 2562). Pass --help for the argument list.
#
# The viewer draws, at each atom, a sphere coloured by the orientation probability and
# deformed into the distribution's lobes, with a temperature slider over the sweep: as τ
# rises the single-site distribution visibly broadens from a tight cap (ordered) toward a
# near-uniform sphere (disordered). The tensorial / multipole samplers (single-ion
# anisotropy, full SCE multipoles) export through the very same `write_mfa_distributions`
# — only the per-atom coefficient length (lmax) grows.

using SLCE
using SLCETools
using SLCETools.VASP: read_poscar

# --- optional CLI args: tau_min tau_max ntau [npoints] ---------------------------
if !isempty(ARGS) && ARGS[1] in ("-h", "--help")
    println("""
    Usage: julia --project=examples examples/mfa_distributions.jl [tau_min tau_max ntau [npoints]]
      tau_min, tau_max  reduced-temperature sweep endpoints in (0, 1]   (default 0.2 0.95)
      ntau              number of τ points                              (default 8)
      npoints           sphere-grid points per atom                     (default 2562)""")
    exit(0)
end
tau_min = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.2
tau_max = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.95
ntau    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
npoints = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2562
0 < tau_min <= tau_max <= 1 || error("need 0 < tau_min ≤ tau_max ≤ 1; got $tau_min, $tau_max")
ntau >= 1 || error("ntau must be ≥ 1; got $ntau")
npoints >= 1 || error("npoints must be ≥ 1; got $npoints")

dir = mktempdir()

# --- a structure file (stands in for a real POSCAR/CONTCAR) ----------------------
# Two Fe atoms in a 2.5 Å cubic cell: a corner atom and a body-centre atom.
write(joinpath(dir, "POSCAR"),
      "Fe2\n1.0\n 2.5 0 0\n 0 2.5 0\n 0 0 2.5\nFe\n2\nDirect\n 0 0 0\n 0.5 0.5 0.5\n")
crystal = read_poscar(joinpath(dir, "POSCAR"))
println("read_poscar → ", n_atoms(crystal), " atoms, species ", crystal.species_labels)

# --- a hand-built exchange model + reference state -------------------------------
# Antiferromagnetic bilinear coupling (E = J e·e, J > 0) with the reference up/down. The
# single-site distribution is then an axisymmetric vMF cone whose opening angle grows with
# τ — the clearest illustration of the mean-field broadening on the slider.
exch = ExchangeModel([0.0 1.0; 1.0 0.0])
reference = Float64[0 0; 0 0; 1 -1]                    # 3×2, atom 1 up, atom 2 down
sampler = MFASampler(exch; reference = reference)
println("sampler: ", sampler, ",  T_MF = ", mfa_temperature_scale(sampler))

# --- write the per-atom distributions over a temperature sweep -------------------
taus = ntau == 1 ? [tau_min] : range(tau_min, tau_max; length = ntau)
out = joinpath(@__DIR__, "mfa_distributions.json")
write_mfa_distributions(out, sampler, crystal; taus = taus, npoints = npoints)
println("wrote ", out)
println("  τ sweep: ", round.(collect(taus); digits = 3), "  (npoints = ", npoints, ")")
println("  per-atom m at τ=", first(taus), ": ",
        round.(mfa_sublattice_m(sampler, first(taus)); digits = 4))
println("  per-atom m at τ=", last(taus), ": ",
        round.(mfa_sublattice_m(sampler, last(taus)); digits = 4))
println("\nVisualize:  python viz/mfa_viewer.py ", out)
