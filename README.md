# SCETools.jl

Auxiliary tooling around the spin-cluster-expansion (SCE) fitting core
[`MagestyRebuild.jl`](../Magesty_rebuild.jl): utilities that **consume** a fitted
`SCEModel` rather than build one.

- **Mean-field (MFA) spin-configuration sampling** *(available)* — draw physically
  representative finite-temperature spin configurations from the single-site mean field of
  a fitted model (or a hand-built exchange model) at a controlled reduced temperature
  `τ = T/T_MF`. Four constructions of increasing fidelity, one `sample` verb:
  - `MFASampler(reference)` — a single global isotropic sampler (Langevin / von Mises–Fisher);
  - `MFASampler(ExchangeModel(...); reference)` — multi-sublattice isotropic and tensorial
    (Heisenberg / DMI / anisotropic exchange + single-ion);
  - `MFASampler(model::SCEModel; reference)` — the full multipole sampler from a fitted
    model (all clusters and `l`, higher-order / many-body).
- **Active learning** *(planned)* — an efficient model-construction loop that proposes
  configurations (via the sampler), labels them with DFT, and refits the SCE model. See
  `SPEC.md`.

## Relationship to the ecosystem

This package re-founds the sampling / active-learning layer on the clean
`MagestyRebuild` rebuild. The older `SpinClusterMC.jl` (Monte Carlo) and `ActiveSCE.jl`
(active learning) packages remain in use against the original `Magesty.jl` and are not
targeted here.

## Install (development)

Both packages are unregistered; develop the core by path:

```julia
using Pkg
Pkg.develop(path="../Magesty_rebuild.jl")   # the SCE fitting core
```

## Usage

```julia
using MagestyRebuild, SCETools

model = …                                   # a fitted SCEModel (see MagestyRebuild)
ref   = …                                   # 3 × n_atoms reference directions (unit columns)
s     = MFASampler(model; reference = ref)  # keep every channel (bilinear … many-body)
samp  = sample(s, 0.6)                       # configurations at τ = 0.6
```

The sampler reads the fitted Hamiltonian only through `MagestyRebuild`'s public surface
(`multipole_terms`, `bilinear_terms`, `MagestyRebuild.Harmonics`), so it is insulated from
the core's SALC-basis internals.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'             # unit + Aqua
TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'  # + JET
```
