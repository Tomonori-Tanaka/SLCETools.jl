# Getting started

```@meta
CurrentModule = SCETools
```

## Installation

SCETools.jl is an exploratory package and is not registered. It depends on the SCE fitting
core [SCEFitting.jl](https://github.com/Tomonori-Tanaka/SCEFitting.jl); add both
from their local paths (or git URLs) in the package manager:

```julia
using Pkg
Pkg.develop(path = "/path/to/SCEFitting.jl")   # the SCE fitting core
Pkg.develop(path = "/path/to/SCETools.jl")
```

Both packages have only lightweight dependencies.

## Sample a Heisenberg dimer from a fitted model

The shortest meaningful end-to-end: build a tiny fitted model with SCEFitting, then draw
finite-temperature configurations from its mean field. Here is a 4-atom chain whose
nearest-neighbour pair (atoms 1–2) carries a ferromagnetic Heisenberg coupling; atoms 3–4 are
uncoupled.

```@example gs
using SCEFitting, SCETools
using LinearAlgebra, Random, Statistics

# A 4-atom chain along z; nearest-neighbour 2-body isotropic (Heisenberg) basis.
lat   = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
frac  = [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75]
chain = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])
basis = SCEBasis(chain, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], isotropy = true))

# A fitted model: the first SALC (the nearest-neighbour Heisenberg bond) carries a
# negative coefficient ⇒ ferromagnetic along the reference; the other SALCs stay zero.
# (In practice `jphi` comes from `fit`; here we just set it.)
model = SCEPredictor(basis, 0.0, vcat([-0.02], zeros(n_salcs(basis) - 1)))
nothing # hide
```

Build a sampler about a ferromagnetic reference (all spins along ``+z``) and read off the
mean-field temperature scale and the per-atom magnetizations:

```@example gs
ref = Float64[0 0 0 0; 0 0 0 0; 1 1 1 1]      # 3 × n_atoms, unit columns
s   = MFASampler(model; reference = ref)

mfa_temperature_scale(s)                       # T_MF, set by the bilinear Perron eigenvalue
```

```@example gs
# Per-atom magnetization m_a(τ) = ⟨e_a · ê_a⟩ at a few reduced temperatures.
# The coupled pair orders together; the free spins (3,4) stay disordered.
[(τ, round.(mfa_sublattice_m(s, τ); digits = 3)) for τ in (0.3, 0.6, 0.9)]
```

Now draw configurations. [`sample`](@ref) returns an [`MFASample`](@ref) carrying the
configurations and the parallel labels `.tau` / `.m`:

```@example gs
rng  = MersenneTwister(2026)
samp = sample(s, 200; tau = 0.5, rng = rng)

n_drawn   = length(samp.configs)               # 200 configurations, each 3 × 4
first_cfg = size(samp.configs[1])
mz_pair   = mean(mean(@view c[3, 1:2]) for c in samp.configs)   # ⟨z⟩ on the coupled pair
(; n_drawn, first_cfg, mz_pair = round(mz_pair; digits = 3))
```

The drawn pair magnetization tracks the self-consistent `mfa_sublattice_m(s, 0.5)` above
(within sampling noise).

## The single global sampler

With no exchange model, [`MFASampler(reference)`](@ref) is the single-global-magnetization
sampler: every spin shares one Langevin magnetization ``m(\tau)``, and ``T_{\mathrm{MF}} = 1``
(only the reduced temperature is physical). The magnetization ↔ temperature map is exposed
directly:

```@example gs
g = MFASampler(ref)                            # global isotropic sampler

m_at = thermal_averaged_m(0.5)                 # m(τ = 0.5) from the Langevin self-consistency
τ_of = tau_from_magnetization(0.8)             # the inverse: τ giving m = 0.8
(; m_at = round(m_at; digits = 4), τ_of = round(τ_of; digits = 4))
```

## Where to go next

- [Sampling](guide/sampling.md) — the `sample` verb in full, the fidelity ladder, and the
  `MFASample` output.
- [Exchange models](guide/exchange_models.md) — build an `ExchangeModel` by hand (Heisenberg,
  DMI, anisotropic, single-ion) or extract one from a fitted `SCEPredictor`.
- [Theory](theory/mfa.md) — the mean-field decoupling and the single-site distributions.
