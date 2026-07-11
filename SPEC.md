# SPEC — SCETools.jl

Auxiliary tooling around the SCE fitting core `SCEFitting.jl`. Components **consume** a
fitted `SCEPredictor` (or a hand-built exchange model); they never build or fit one. This file
records the realized architecture and the planned active-learning layer.

## Dependency boundary

`SCETools` depends on `SCEFitting` and reads a fitted model **only** through its public
surface:

- `multipole_terms(model) :: Vector{MultipoleTerm}` — the flat, code-neutral per-term view
  (raw `jϕ` coefficient, `body`, `atoms`, `shifts`, `ls`, `folded`).
- `bilinear_terms(model) :: (; pairs, onsites, skipped)` — the bilinear (`ls=[1,1]`) /
  single-ion (`ls=[2]`) channels as Cartesian `3×3` matrices.
- `n_atoms(model)` and the tesseral submodule `SCEFitting.Harmonics` (`Zlm`, `lm_index`).

It never touches the SALC-basis internals (`model.basis.salc_basis.salcs`, `SALCMember`,
`SALCTerm`). The development dependency is a path-dev (`Pkg.develop(path="../SCEFitting.jl")`).

## Module layout

```
src/SCETools.jl              # module: imports + includes + the two export tiers
src/mfa/
  engine.jl                  # module MeanFieldEngine: single-site potential, vMF / Metropolis
                             #   draws, sphere quadrature — pure on-sphere math, no SCE coupling;
                             #   exports site_potential, sample_vmf, sample_vmf_field,
                             #   sample_site_metropolis, SphereQuadrature, sphere_quadrature,
                             #   field_scale, multipole_average (the `_`-prefixed helpers stay private)
  types.jl                   # the carrier / sampler structs + invariant-enforcing inner ctors:
                             #   AbstractSampler, ExchangeModel, MultipoleModel, MFASampler{S}, MFASample
  exchange.jl                # ExchangeModel construction + the longitudinal molecular-field analysis
  selfconsistency.jl         # the Langevin closed forms + the three mean-field solvers (P2/P3/P4)
  sampler.jl                 # the MFASampler constructors + the `sample` verb + dispatch
  bridge.jl                  # ExchangeModel / MultipoleModel / MFASampler from a fitted SCEPredictor
                             #   (+ `_scaled_multipole_terms`, the package's single (4π)^(N/2) site)
src/mc/
  metropolis.jl              # MetropolisSampler / MCSample: single-spin Metropolis on the joint
                             #   Boltzmann distribution of the fitted SCE (training cell, absolute
                             #   k_B·T) — its own directory as a future extraction seam
src/io/
  vasp.jl                    # module SCETools.VASP: the VASP adapter — read (read_poscar /
                             #   Oszicar) + write (write_poscar / write_incar / write_inputs)
src/viz/
  grid.jl                    # the shared Fibonacci render grid + the tesseral basis matrix Z
  distributions.jl           # per-atom single-site coefficients at one τ + a verification density
  serialize.jl               # the self-describing JSON document (dependency-free emitter)
```

The active-learning layer (see below) is planned but unimplemented; no placeholder
directory exists yet — it will be created with its first file.

`test/oracle/` is a separate environment (pinned Magesty.jl) that cross-checks the VASP
POSCAR / OSZICAR parsers bit-for-bit against Magesty; run with
`julia --project=test/oracle test/oracle/runtests.jl`.

## Public API (sampling)

The export surface is tiered (mirroring `SCEFitting`): a lean exported workflow plus a
*public but unexported* tier reached by qualification (`SCETools.<name>`), declared with
the Julia `public` keyword so the tier is machine-checkable (`Base.ispublic`, Aqua).

- **Exported** — `AbstractSampler`, `MFASampler`, `MFASample`, `MetropolisSampler`,
  `MCSample`, `ExchangeModel`, `sample`;
  the helpers `mfa_temperature_scale`, `mfa_sublattice_m`, `thermal_averaged_m`,
  `tau_from_magnetization`; and the viz output `SiteDistributionField`,
  `mfa_site_coefficients`, `write_mfa_distributions`.
- **Public, unexported** (`public`) — `MultipoleModel` (the full-multipole digest; usually
  built via `MFASampler(model)`); the viz plumbing `SphereGrid` / `fibonacci_sphere` /
  `harmonic_basis` / `site_probabilities`; and the submodules `MeanFieldEngine` (engine
  primitives: `site_potential`, `sample_vmf`, `sample_vmf_field`, `sample_site_metropolis`,
  `SphereQuadrature`, `sphere_quadrature`, `field_scale`, `multipole_average`) and `VASP`.

`sample(sampler, n; tau, …)` → `MFASample` (configurations, per-atom magnetizations);
`tau` (or `m`) is keyword-only, and a collection sweep drops the positional `n` in favor
of `nsamples`.
`MFASampler{S}` is parametric on its coupling source (`Nothing` / `ExchangeModel` /
`MultipoleModel`) so the draw-path dispatch is type-stable.

Construction fidelity ladder: `MFASampler(reference)` (single global isotropic) →
`MFASampler(ExchangeModel(...); reference)` (multi-sublattice isotropic / tensorial) →
`MFASampler(model::SCEPredictor; reference)` (full multipole, all clusters and `l`).

`MetropolisSampler(model::SCEPredictor; reference = nothing)` is the joint-Boltzmann
sibling behind the same `sample` verb: single-spin Metropolis on the training cell at an
**absolute** `temperature = k_B·T` (model energy units — no `τ`, no `l=1` Perron scale,
so any body order works) → `MCSample` (configs + parallel `temperature` / `energy` /
`acceptance`). Scope is configuration sampling only; supercell tiling and thermodynamic
observables (`m(T)`, `T_c`) are explicitly deferred — see `docs/specs/mc-sampling.md`.

## Public API (VASP I/O — `SCETools.VASP`)

The concrete VASP adapter the fitting core leaves out (the core owns only the abstract DFT-data
seam). Both directions, with one shared frame / format convention so a write → read round-trip
is the identity:

- **read** — `read_poscar(path) -> Crystal`; `Oszicar(paths; saxis, energy_kind, mint)` (an
  `AbstractDFTSource`, consumed by `SCEFitting.read_configs` / `SCEDataset`). Produces
  training data.
- **write** — `write_poscar(path, crystal; …)`; `write_incar(path, directions; magmoms, base,
  constrain, saxis, …)`; `write_inputs(dir | rootdir, crystal, config | configs; …)` (a POSCAR +
  INCAR input set / sweep, atom order matched). Moment magnitudes come from a per-atom vector, a
  scalar, a per-species map, or a template INCAR's MAGMOM.

See `docs/specs/mfa-sampling.md` for the design (decisions D1–D5, phases P0–P4) and the
physical conventions (`τ = T/T_MF`, `T_MF = ρ/3`, mean-field decoupling, vMF / Bingham).

## Planned — active-learning layer (a future `src/active_learning/`)

Not yet implemented (no placeholder directory — created with its first file); design intent
recorded so the package is laid out for it.

An efficient SCE model-construction loop, closing the sample → label → refit cycle:

1. **Propose** candidate configurations with the MFA sampler at a chosen `τ` (cheap,
   physically representative), optionally targeted by an acquisition criterion.
2. **Label** them with DFT (write inputs via `SCETools.VASP`, run an external solver,
   read back energies / torques as `SpinDatum`).
3. **Refit** the SCE model with `SCEFitting.fit` / `refit` on the augmented dataset.
4. **Assess** uncertainty / acquisition and iterate until a target is met.

The existing `ActiveSCE.jl` (GroupARD posterior, acquisition policies, finite-T campaign
scaffold) is a **read-only design reference**; this layer does not depend on it (it targets
the legacy `Magesty.jl`). Convenience helpers (configuration format conversion, sampler
diagnostics) may also live in this package as they arise.
