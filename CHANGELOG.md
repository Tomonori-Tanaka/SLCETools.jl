# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this package predates a tagged
release, so everything lives under *Unreleased*.

## [Unreleased]

### Changed — VASP **reader** moved in from MagestyRebuild (all VASP I/O now here)

The concrete VASP reader (`read_poscar`, `write_poscar`, `Oszicar` → `SpinDatum` via the
`MagestyRebuild.read_configs` seam) moved out of the fitting core into `SCETools.VASP`, joining
the writer so all VASP I/O lives in one place. The fitting core now keeps only the abstract
DFT-data seam (`AbstractDFTSource` / `SpinDatum` / `SCEDataset`). Reading VASP training data is
now `using SCETools; SCETools.VASP.read_poscar` / `Oszicar`. The unit tests
(`test/unit/test_vaspio.jl`), the `examples/vasp_dft_source.jl` end-to-end example, and the
VASP-vs-Magesty oracle cross-check (`test/oracle/`) came with it. Adds the `StaticArrays`
dependency usage in the VASP module (already a dependency).

### Added — VASP input writing (`SCETools.VASP`)

Turn sampled spin configurations into constrained-noncollinear VASP inputs (the
active-learning "label" step). Part of the `SCETools.VASP` adapter (alongside the reader above).

- **`write_incar(path, directions; magmoms, base, constrain, saxis, …)`** — write one INCAR.
  The MAGMOM (and, under `constrain`, M_CONSTR) is `magnitude · direction` per atom, `%.9f`.
  A `base` template (path or raw text) is preserved verbatim except for MAGMOM / M_CONSTR;
  without one, a minimal noncollinear INCAR is written.
- **`write_inputs(dir, crystal, config; …)` / `write_inputs(rootdir, crystal, configs; …)`** —
  write a full input set (POSCAR via `MagestyRebuild.VASP.write_poscar` + a matching INCAR), or
  a sweep with one subdirectory per configuration. The INCAR's atom order is regrouped by
  species to match the POSCAR, so the two files are always consistent.
- **Moment magnitudes** (μ_B — the sampler only gives directions) come from a per-atom vector, a
  scalar, a per-species `label => magnitude` map, or the `base` template's MAGMOM norms.
- **SAXIS** — moments are written in the global Cartesian frame by default; a non-default axis
  (from the `saxis` kwarg or the template) writes them in the SAXIS frame, the inverse of the
  reader's `Rz(α)Ry(β)`, so a write → read round-trip is the identity.
- Guards: non-negative magnitudes, atom-count / direction-norm checks, and a typo'd template
  path errors instead of being embedded as a stray line. Adds the `Printf` stdlib dependency.

### Added — initial package: the mean-field spin-configuration sampler

Extracted from `MagestyRebuild.jl` (where it was developed as phases P0–P4) into this
auxiliary package, which depends on the core for the fitted model and its public
introspection surface (`multipole_terms`, `bilinear_terms`, `MagestyRebuild.Harmonics`).

- **P0 — single-site engine** (`sampling/site_engine.jl`): the single-site tesseral
  potential `H_a(e) = Σ h^{lm} Z_lm(e)`, the von Mises–Fisher draw (Ulrich/Wood inverse-CDF
  for `p = 3`), a Metropolis sampler for non-vMF (Bingham / higher-multipole) shapes, and a
  field-aware Gauss–Legendre × azimuth sphere quadrature for the multipole averages `⟨Z_lm⟩`.
- **P1 — single global isotropic sampler** (`MFASampler(reference)`): the Langevin
  self-consistency `m = L(3m/τ)` (`L(κ) = coth κ − 1/κ`), with `thermal_averaged_m` /
  `tau_from_magnetization` and a closed-form vMF draw.
- **P2 — multi-sublattice isotropic** (`ExchangeModel(Jiso; onsite)`): the molecular-field
  matrix `A[a,b] = −Jiso[a,b](ê_a·ê_b)`, its Perron scale `T_MF = ρ/3`, the coupled
  `m_a = L(3(Ā m)_a/τ)` solved by depth-1 Anderson acceleration (critical slowing as τ→1⁻),
  and per-atom vMF concentrations.
- **P3 — tensorial exchange + single-ion** (`ExchangeModel(bilinear; onsite)`): the full
  bilinear tensor `S_ab` (Heisenberg + DMI + anisotropic) and single-ion `A_a`, the
  single-site potential `V_a(e) = β(e·g_a + e' A_a e)` whose `l=2` Bingham factor the vMF
  cannot represent, solved as `m_a = ⟨e·ê_a⟩` by quadrature and drawn by Metropolis.
- **P4 — full multipole / many-body** (`MultipoleField` / `MFASampler(model::SCEModel)`):
  the mean field over **all** SCE clusters and harmonic orders, with the generalized
  molecular field `h_a^{lm} = Σ_φ jφ·(4π)^(N/2)·folded·∏_{b≠a} ⟨Z_b⟩` and per-atom multipole
  averages `⟨Z_lm⟩_a` iterated to self-consistency.
- **`sce_bridge.jl`** builds `ExchangeModel(model)` / `MultipoleField(model)` /
  `MFASampler(model)` from a fitted SCE **through the core's public introspection**
  (`multipole_terms` / `bilinear_terms`), not its SALC-basis internals.
- Validated by exact reduction to the single-global Langevin curve for a pure-bilinear
  model, coupling-scale invariance, easy-axis / easy-plane / DMI shapes, the
  Metropolis↔quadrature `⟨Z2m⟩` agreement, and the many-body factorization check
  `V_a/β = ⟨E | e_a⟩` to machine precision. Aqua + JET clean.
