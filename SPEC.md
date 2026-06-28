# SPEC — SCETools.jl

Auxiliary tooling around the SCE fitting core `MagestyRebuild.jl`. Components **consume** a
fitted `SCEModel` (or a hand-built exchange model); they never build or fit one. This file
records the realized architecture and the planned active-learning layer.

## Dependency boundary

`SCETools` depends on `MagestyRebuild` and reads a fitted model **only** through its public
surface:

- `multipole_terms(model) :: Vector{MultipoleTerm}` — the flat, code-neutral per-term view
  (raw `jϕ` coefficient, `body`, `atoms`, `shifts`, `ls`, `folded`).
- `bilinear_terms(model) :: (; pairs, onsites, skipped)` — the bilinear (`ls=[1,1]`) /
  single-ion (`ls=[2]`) channels as Cartesian `3×3` matrices.
- `num_atoms(model)` and the tesseral submodule `MagestyRebuild.Harmonics` (`Zlm`, `lm_index`).

It never touches the SALC-basis internals (`model.basis.salcs.salcs`, `SALCMember`,
`SALCTerm`). The development dependency is a path-dev (`Pkg.develop(path="../Magesty_rebuild.jl")`).

## Module layout

```
src/SCETools.jl              # module: imports + includes + exports
src/sampling/
  site_engine.jl             # P0: single-site potential, vMF / Metropolis draws, quadrature
  exchange.jl                # P2/P3: ExchangeModel + MultipoleField carriers, MFA self-consistency
  mfa_sampler.jl             # P1–P4: MFASampler / MFASample, the `sample` verb, dispatch
  sce_bridge.jl              # ExchangeModel / MultipoleField / MFASampler from a fitted SCEModel
src/io/
  vasp.jl                    # module SCETools.VASP: the VASP adapter — read (read_poscar /
                             #   Oszicar) + write (write_poscar / write_incar / write_inputs)
src/active_learning/         # planned (see below); empty until implemented
```

`test/oracle/` is a separate environment (pinned Magesty.jl) that cross-checks the VASP
POSCAR / OSZICAR parsers bit-for-bit against Magesty; run with
`julia --project=test/oracle test/oracle/runtests.jl`.

## Public API (sampling)

- Types: `AbstractSampler`, `MFASampler`, `MFASample`, `ExchangeModel`, `MultipoleField`.
- Verb: `sample(sampler, τ; …)` → `MFASample` (configurations, per-atom magnetizations).
- Helpers: `mfa_temperature_scale`, `mfa_sublattice_m`, `thermal_averaged_m`,
  `tau_from_magnetization`.

Construction fidelity ladder: `MFASampler(reference)` (single global isotropic) →
`MFASampler(ExchangeModel(...); reference)` (multi-sublattice isotropic / tensorial) →
`MFASampler(model::SCEModel; reference)` (full multipole, all clusters and `l`).

## Public API (VASP I/O — `SCETools.VASP`)

The concrete VASP adapter the fitting core leaves out (the core owns only the abstract DFT-data
seam). Both directions, with one shared frame / format convention so a write → read round-trip
is the identity:

- **read** — `read_poscar(path) -> Crystal`; `Oszicar(paths; saxis, energy_kind, mint)` (an
  `AbstractDFTSource`, consumed by `MagestyRebuild.read_configs` / `SCEDataset`). Produces
  training data.
- **write** — `write_poscar(path, crystal; …)`; `write_incar(path, directions; magmoms, base,
  constrain, saxis, …)`; `write_inputs(dir | rootdir, crystal, config | configs; …)` (a POSCAR +
  INCAR input set / sweep, atom order matched). Moment magnitudes come from a per-atom vector, a
  scalar, a per-species map, or a template INCAR's MAGMOM.

See `docs/specs/mfa-sampling.md` for the design (decisions D1–D5, phases P0–P4) and the
physical conventions (`τ = T/T_MF`, `T_MF = ρ/3`, mean-field decoupling, vMF / Bingham).

## Planned — active-learning layer (`src/active_learning/`)

Not yet implemented; design intent recorded so the package is laid out for it.

An efficient SCE model-construction loop, closing the sample → label → refit cycle:

1. **Propose** candidate configurations with the MFA sampler at a chosen `τ` (cheap,
   physically representative), optionally targeted by an acquisition criterion.
2. **Label** them with DFT (write inputs via `MagestyRebuild.VASP`, run an external solver,
   read back energies / torques as `SpinDatum`).
3. **Refit** the SCE model with `MagestyRebuild.fit` / `refit` on the augmented dataset.
4. **Assess** uncertainty / acquisition and iterate until a target is met.

The existing `ActiveSCE.jl` (GroupARD posterior, acquisition policies, finite-T campaign
scaffold) is a **read-only design reference**; this layer does not depend on it (it targets
the legacy `Magesty.jl`). Convenience helpers (configuration format conversion, sampler
diagnostics) may also live in this package as they arise.
