# SCETools.jl

Auxiliary tooling around the spin-cluster-expansion (SCE) fitting core
[`SCEFitting.jl`](../SCEFitting.jl): utilities that **consume** a fitted
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
- **VASP I/O** *(available)* — `SCETools.VASP` is the concrete VASP adapter (the fitting core
  keeps only the abstract DFT-data seam): **read** training data (`read_poscar`, `Oszicar` →
  `SpinDatum`) and **write** constrained-noncollinear inputs from sampled configurations
  (`write_incar` / `write_inputs`). Read and write share one frame / format convention.
- **Active learning** *(planned)* — an efficient model-construction loop that proposes
  configurations (via the sampler), labels them with DFT, and refits the SCE model. See
  `SPEC.md`.

## Relationship to the ecosystem

This package re-founds the sampling / active-learning layer on the clean
`SCEFitting` rebuild. The older `SpinClusterMC.jl` (Monte Carlo) and `ActiveSCE.jl`
(active learning) packages remain in use against the original `Magesty.jl` and are not
targeted here.

## Install (development)

Both packages are unregistered; develop the core by path:

```julia
using Pkg
Pkg.develop(path="../SCEFitting.jl")   # the SCE fitting core
```

## Usage

```julia
using SCEFitting, SCETools

model = …                                   # a fitted SCEModel (see SCEFitting)
ref   = …                                   # 3 × n_atoms reference directions (unit columns)
s     = MFASampler(model; reference = ref)  # keep every channel (bilinear … many-body)
samp  = sample(s, 0.6)                       # configurations at τ = 0.6
```

The sampler reads the fitted Hamiltonian only through `SCEFitting`'s public surface
(`multipole_terms`, `bilinear_terms`, `SCEFitting.Harmonics`), so it is insulated from
the core's SALC-basis internals.

## Documentation

A Documenter site (Home, Getting started, Guide, Theory, API) lives under `docs/`:

```bash
make -C docs serve      # build, then serve at http://localhost:8000 with live reload
make -C docs build      # build the static HTML into docs/build/
```

The build is local-only (no published remote yet); `SCEFitting` must sit at
`../SCEFitting.jl`.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'             # unit + Aqua
TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'  # + JET
```
