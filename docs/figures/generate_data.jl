# Generate the data behind the MC-sampling guide figures (docs/src/assets/mc_*.svg).
# Self-contained fixtures only (no external data). Regenerate with
#     julia --project docs/figures/generate_data.jl
#     python3 docs/figures/plot_figures.py
# CSVs land in docs/figures/data/ (tracked, so the plots re-render without Julia).

using SLCE
using SLCETools
using LinearAlgebra
using Printf
using Random
using Statistics: mean, std

const DATA = joinpath(@__DIR__, "data")
mkpath(DATA)

_langevin(x) = coth(x) - 1 / x

# --- fixtures -------------------------------------------------------------------

# The ferromagnetic Heisenberg dimer of the test suite (couples atoms 1–2; 3–4 free).
function dimer_model()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], isotropy = true))
    return SLCEModel(b, 0.0, vcat([-0.02], zeros(n_salcs(b) - 1)))   # negative ⇒ ferro
end

# An 8-atom simple-cubic ferromagnet (2×2×2 sites in one cell): every bilinear SALC at the
# same negative coefficient gives a uniform nearest-neighbor ferro (12 pairs, one J_pair).
function cube_model()
    lat = Lattice(Matrix(4.0 * LinearAlgebra.I(3)))
    frac = reduce(hcat, [[i, j, k] ./ 2 for i = 0:1 for j = 0:1 for k = 0:1])
    cr = Crystal(lat, Float64.(frac), ones(Int, 8), ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.2, lmax = [1], isotropy = true))
    return SLCEModel(b, 0.0, fill(-0.02, n_salcs(b)))
end

# --- figure 1: dimer ⟨e₁·e₂⟩ vs k_BT/|J| — exact vs MC vs mean field ---------------

model = dimer_model()
mc = MetropolisSampler(model)
J = ExchangeModel(model).Jiso[1, 2]                     # < 0 (ferro), E = J e₁·e₂
Jabs = abs(J)

open(joinpath(DATA, "fig1_curves.csv"), "w") do io
    println(io, "t,exact,mfa")                          # t = k_B T / |J|
    for t in range(0.02, 1.2; length = 240)
        exact = _langevin(1 / t)                        # ⟨e₁·e₂⟩ = L(β|J|)
        m = thermal_averaged_m(3t)                      # τ = T/T_MF = 3 k_BT/|J|
        println(io, t, ",", exact, ",", m^2)            # MFA factorizes: ⟨e₁·e₂⟩ → m²
    end
end

open(joinpath(DATA, "fig1_mc.csv"), "w") do io
    println(io, "t,corr,stderr")
    for (i, t) in enumerate([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9, 1.1])
        samp = sample(mc, 2000; kT = t * Jabs, burnin = 500, thin = 8,
                      rng = MersenneTwister(100 + i))
        c12 = [dot(c[:, 1], c[:, 2]) for c in samp]
        println(io, t, ",", mean(c12), ",", std(c12) / sqrt(length(c12)))
    end
end
println("fig1 done")

# --- figure 2: annealing trace on the 8-atom cube ----------------------------------

model8 = cube_model()
mc8 = MetropolisSampler(model8)
J8 = abs(ExchangeModel(model8).Jiso[1, 2])
aligned = zeros(3, 8)
aligned[3, :] .= 1.0
E0 = predict_energy(model8, aligned)                    # ground state (j0 = 0)

# The 2×2×2 periodic cell folds each ± neighbor pair onto one bond (z_eff = 3, Perron
# ρ = 3|J_pair|), so the mean-field scale is k_B·T_MF = ρ/3 = |J_pair|. Span well above
# (disordered) to well below (saturated): 8 → 0.3 in units of |J_pair|.
ladder = [8.0, 4.0, 2.0, 1.0, 0.3] .* J8                # k_B T in eV, high → low
samp8 = sample(mc8; kT = ladder, nsamples = 40, burnin = 60, thin = 3,
               rng = MersenneTwister(7))                # random init, warm-started ladder
open(joinpath(DATA, "fig2_trace.csv"), "w") do io
    println(io, "# E0 = ", E0, "  Jabs = ", J8)
    println(io, "idx,t_over_J,energy,acceptance")
    for (k, (kt, E, a)) in enumerate(zip(samp8.kT, samp8.energy, samp8.acceptance))
        println(io, k, ",", kt / J8, ",", E, ",", a)
    end
end
println("fig2 done")

# --- figure 3: the absolute-orientation zero mode and `randomize` ------------------

# Low temperature, ordered start along +z: without `randomize` the configuration's mean
# axis only creeps away from ẑ (the global orientation is a zero mode the local updates
# diffuse at a rate ∝ 1/n_atoms — hence a 64-site cell, where the creep is slow on the
# length of the run), with it the stored copies are Haar-uniform. Track cosθ = n̂·ẑ.
function cube64_model()
    lat = Lattice(Matrix(8.0 * LinearAlgebra.I(3)))
    frac = reduce(hcat, [[i, j, k] ./ 4 for i = 0:3 for j = 0:3 for k = 0:3])
    cr = Crystal(lat, Float64.(frac), ones(Int, 64), ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.2, lmax = [1], isotropy = true))
    return SLCEModel(b, 0.0, fill(-0.02, n_salcs(b)))
end
model64 = cube64_model()
mc64 = MetropolisSampler(model64)
J64 = maximum(abs, ExchangeModel(model64).Jiso)
init = zeros(3, 64)
init[3, :] .= 1.0
axis_z(c) = normalize(vec(mean(c; dims = 2)))[3]
kw = (; kT = 0.1 * J64, burnin = 50, thin = 2, init = init)
off = sample(mc64, 200; kw..., rng = MersenneTwister(21))
on = sample(mc64, 200; kw..., rng = MersenneTwister(21), randomize = true)
open(joinpath(DATA, "fig3_axis.csv"), "w") do io
    println(io, "idx,cos_off,cos_on")
    for k = 1:200
        println(io, k, ",", axis_z(off.configs[k]), ",", axis_z(on.configs[k]))
    end
end
println("fig3 done")
