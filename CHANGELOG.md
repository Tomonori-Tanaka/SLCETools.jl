# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this package predates a tagged
release, so everything lives under *Unreleased*.

## [Unreleased]

### Changed — review-driven refactor (Tier 0 fixes, Tier 1 hot path, Tier 2 structure)

A four-axis review (numerical, generalist, performance, ideal-package) drove a tiered
refactor. Breaking changes are confined to internals and one renamed type; the headline
`MFASampler` / `sample` / `ExchangeModel` / `write_mfa_distributions` / `SCETools.VASP`
path is unchanged. Unit count 377 → 500; Aqua + JET green.

**Tier 0 — correctness / edge cases** (none affecting normal-operation results):
- Ordered limit (`τ → 0`): the saturated order parameter `m → 1` is now gated on a nonzero
  *l=1* molecular field, not on single-ion presence. A purely single-ion (even-l) atom has
  an `e → −e` symmetric distribution, so `⟨e·ê_a⟩ → 0` — reporting `m = 1` was wrong.
- `sample_vmf` returns `μ` directly for `κ = Inf`, avoiding a NaN when `rand()` returns 0.0.
- The JSON emitter escapes `\n \r \t` and all control characters (RFC 8259).
- Each drawn config gets its own `m` vector (no aliasing across co-τ draws); `MFASample`
  gains `firstindex` / `lastindex`. Dropped the unused `Diagonal` / `det` imports.

**Tier 1 — bit-identical hot path** (verified against the `V_a/β = ⟨E|e_a⟩` machine-precision
gate and the Langevin reduction):
- `multipole_average` tabulates `Z_lm(e)` once per quadrature node (was evaluated twice).
- Internal unit-direction paths use `Harmonics.Zlm_unsafe`, skipping the per-call norm check.
- The many-body `_site_coeffs_all!` dispatches each term through a rank-specialized barrier.

**Tier 2 — public API / structure** (breaking):
- **Export tiering**: a lean exported workflow plus a *public but unexported* tier
  (`MultipoleModel`, `SphereGrid`, `fibonacci_sphere`, `harmonic_basis`, `site_probabilities`,
  and the `SCETools.MeanFieldEngine` kernels), reached by qualification.
- **`MultipoleField` → `MultipoleModel`** (a coupling model, the full-fidelity sibling of
  `ExchangeModel`, not a field); `Base.@deprecate_binding` keeps the old name one minor version.
- **`MFASampler{S}`** is now parametric on its coupling source (single `source::S` field,
  `S ∈ {Nothing, ExchangeModel, MultipoleModel}`), replacing the two `Union{Nothing,…}` fields;
  the draw-path dispatch is type-stable (no `::` assertions).
- **Inner constructors** enforce invariants and are the only build path (`ExchangeModel`,
  `MultipoleModel`, `MFASampler`, `MFASample`); the bridge copies the core's introspection
  arrays so a `MultipoleModel` never aliases the fitted model's internals.
- **File reorg**: `sampling/` → `mfa/` with the single-site engine promoted to a
  `module MeanFieldEngine` (`mfa/engine.jl`); the 475-line `exchange.jl` split into
  `types.jl` / `exchange.jl` / `selfconsistency.jl`; `viz/distributions.jl` split into
  `grid.jl` / `distributions.jl` / `serialize.jl`. Removed the empty `active_learning/`
  placeholder (it returns with its first file).

### Changed — follow the `SCEFitting` public-API rename

Track `SCEFitting`'s breaking public-API rename: the bridge and VASP adapter now use
`SCEPredictor` (was `SCEModel`), `n_atoms` (was `num_atoms`), `n_salcs` (was `nsalc`), and the
`SCEBasis.salc_basis` field (was `.salcs`). No change to `SCETools`' own exported names — a
caller passing a fitted model into `MFASampler` / `ExchangeModel` / `MultipoleField` is
unaffected; only the internal import names moved.

### Changed — fitting-core dependency renamed `MagestyRebuild` → `SCEFitting`

The fitting core this package builds on was renamed from `MagestyRebuild`
(`Magesty_rebuild.jl`) to **`SCEFitting`** (`SCEFitting.jl`), unifying the naming under a
shared `SCE*` family. The UUID is unchanged, so this is purely a name update: `using
MagestyRebuild` becomes `using SCEFitting`, the path-dev points at `../SCEFitting.jl`, and the
`SCEFitting.Harmonics` / `multipole_terms` / `bilinear_terms` introspection surface keeps the
same shape. The legacy `Magesty.jl` package used by the `test/oracle/` cross-check is
unaffected and keeps its name.

### Changed — VASP **reader** moved in from SCEFitting (all VASP I/O now here)

The concrete VASP reader (`read_poscar`, `write_poscar`, `Oszicar` → `SpinDatum` via the
`SCEFitting.read_configs` seam) moved out of the fitting core into `SCETools.VASP`, joining
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
  write a full input set (POSCAR via `SCEFitting.VASP.write_poscar` + a matching INCAR), or
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

Extracted from `SCEFitting.jl` (where it was developed as phases P0–P4) into this
auxiliary package, which depends on the core for the fitted model and its public
introspection surface (`multipole_terms`, `bilinear_terms`, `SCEFitting.Harmonics`).

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
- **P4 — full multipole / many-body** (`MultipoleField` / `MFASampler(model::SCEPredictor)`):
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
