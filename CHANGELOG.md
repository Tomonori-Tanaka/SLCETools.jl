# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this package predates a tagged
release, so everything lives under *Unreleased*.

## [Unreleased]

### Changed — sync to SLCE's `TrainingDatum` data layer (BREAKING upstream)

- The OSZICAR reader follows SLCE's `SpinDatum`-type → `TrainingDatum` replacement:
  `read_configs(::Oszicar) -> Vector{TrainingDatum}`. **Absent ≠ zero**: an OSZICAR
  with no `lambda*MW_perp` block now yields `field = torques = nothing` (the field
  was not computed — previously a fabricated zero-filled field, which would have
  claimed an observed `τ = 0` and admitted false torque rows once mixed datasets
  became legal); a present block with zero rows still means "computed, those atoms
  unconstrained". Provenance is stamped per datum (`constrained`/`torque_qualified`
  derived from the field block; new `Oszicar(...; setup_id = ...)` keyword labels
  the computational setup for SLCE's one-setup-per-dataset invariant).
- Test/example sync for SLCE's `isotropy` → `soc` keyword inversion
  (`isotropy = true` ⇒ `soc = false` and vice versa; 14 sites).

### Changed — BREAKING: package renamed SCETools.jl → SLCETools.jl (SLCE family, M0)

- The whole family is renamed to the **spin–lattice cluster expansion (SLCE)**
  stem per `docs/specs/spin-lattice-ce-design.md` §2 (SLCE.jl /
  SLCEMonteCarlo.jl / SLCEDynamics.jl / SLCETools.jl). Package + module name
  changed; **UUID kept** (path-dev Manifests stay resolvable). Old model /
  checkpoint artifacts are unaffected (persistence schemas carry versions,
  not package names).

### Changed — quadrature memoization in the Anderson self-consistency

- The tensorial and full-multipole self-consistency loops
  (`_tensor_state`/`_multipole_state`) no longer rebuild a `SphereQuadrature`
  (Gauss–Legendre solve included) per atom per Anderson iteration: a new
  memoizing `multipole_average(cache::AbstractDict{Int,SphereQuadrature}, c,
  lmax)` method reuses grids keyed on the auto-selected node count
  (`_quadrature_size`, now the single size definition). The cached grid is a
  pure function of that key, so results are **bit-identical** to the two-arg
  form (pinned by a unit test); the caches are caller-owned locals — no global
  state. Sampling near critical slowing speeds up ~1.2–1.7× on the bench
  fixture.

### Fixed — review-pass hardening (whole-package review, 2026-07-18)

- `MetropolisSampler`'s rotation proposal now projects the rotated spin back
  onto the unit sphere each move: compounded Rodrigues rotations previously
  random-walked the column norm off unity (~ε·√n_accepted — harmless except in
  very long chains). Statistically invisible; the antipodal-flip branch was
  already exact. Sampled streams shift at machine epsilon relative to earlier
  runs.
- `write_incar`: removed the never-wired `_species`/`_labels` keywords — a
  per-species magmom `Dict` needs the crystal and is resolved by
  `write_inputs`, as the error message already directed. No caller change.
- `SPEC.md`: added the missing `KB_EV` to the public-unexported tier;
  documented the `ExchangeModel`-skips vs `MultipoleModel`-errors asymmetry
  for (unreachable today) repeated-atom self-bonds in `bridge.jl`.
- Tests synced to SCEFitting's canonical (v4) SALC members: `multipole_terms`
  now emits **one** member per physical bond (both directed contributions
  pre-summed), so the dimer fixtures expect 1 term instead of 2. The
  machine-precision energy gates were already passing — the numerics were
  never affected, only the term-count expectations.

### Added — Metropolis Monte-Carlo sampler (`MetropolisSampler`)

- **`MetropolisSampler(model::SCEPredictor; reference = nothing)` + `sample(...) ->
  MCSample`** (`src/mc/metropolis.jl`): single-spin Metropolis on the **joint** Boltzmann
  distribution of the fitted SCE over its training cell — the correlated sibling of the
  single-site mean-field `MFASampler`, behind the same `sample` verb. The control is
  absolute — exactly one of `temperature` (**kelvin**, converted with the public
  `SCETools.KB_EV`; assumes an eV-fitted model) or `kT` (`k_B·T` in the model's energy
  units, for theory/test runs and non-eV models); no reduced `τ`, no `l=1` Perron scale,
  so any body order works — including models without a bilinear channel. Distinct
  keyword names so a kelvin value can never be silently read as an energy.
  Each attempt contracts the fitted terms against the current neighbor harmonics
  (`ΔE = c_a·ΔZ`, exact for any body order); β enters only in the accept step. Keywords:
  `burnin` / `thin` (sweeps), `step`, `rng`, `init`, and `randomize` (one Haar rotation
  per stored copy — for an isotropic model still exact Boltzmann with uniform absolute
  orientation, e.g. anisotropy training data). A multi-temperature call warm-starts each
  next temperature (high→low = annealing). `MCSample` carries `configs` with parallel
  `kT` / `temperature` [K] / `energy` (that stored config's SCE energy, `j0` excluded) /
  `acceptance` diagnostics. Design record: `docs/specs/mc-sampling.md`; guide:
  `docs/src/guide/mc_sampling.md`. Supercell tiling and thermodynamic observables are
  explicitly deferred.
- The `(4π)^(N/2)`-scaled term digest is factored out of `MultipoleModel(model)` into
  `_scaled_multipole_terms` (`mfa/bridge.jl`) and shared by both consumers — still the
  package's single scale-application site (pure refactor; the P4 suite is the gate).
- Gates (`test/unit/test_mc_sampler.jl`): machine-precision local↔global energy
  consistency against `predict_energy`; the exact two-spin correlation `⟨e₁·e₂⟩ = −L(βJ)`
  (and the energy diagnostic against `J·⟨e₁·e₂⟩`); the single-site Langevin limit;
  `randomize` isotropy invariance (energy + Gram matrix); byte-identical seeded runs;
  guards.
- Guide figures (`docs/src/assets/mc_*.svg`, generated by `docs/figures/`): the dimer
  exact-vs-MC-vs-mean-field correlation, an annealing energy/acceptance trace, and the
  `randomize` orientation zero-mode illustration.

### Fixed — Metropolis lobe-trapping bias (the flip proposal)

- **The single-site Metropolis engine gained an antipodal-flip proposal** (`e → −e` with
  probability 0.2, else the random rotation; both components symmetric, so detailed
  balance holds). A rotation-only chain started at `+ê_a` could not cross the equator
  barrier of a strongly bimodal potential — an `e ↔ −e` symmetric single-ion double-well,
  exactly the `m → 0` regime near/above `T_MF` or a single-ion-only atom — so the drawn
  ensemble sampled one lobe: every odd multipole was silently biased (⟨e·ê⟩ ≈ +1 where
  the label said `m ≈ 0`) while the even `⟨Z_2m⟩` cross-checks kept passing. On an
  asymmetric (l=1-dominated) well the flip is simply rejected with weight `e^{−ΔV}`, so
  ordered-regime draws are unaffected. New regression gates: a deep double-well engine
  test (⟨e_z⟩ ≈ 0, ⟨Z_20⟩ ↔ quadrature) and a sampled single-ion-only atom whose ensemble
  now matches its own `m ≈ 0` label. Metropolis RNG streams change; seeded runs remain
  reproducible.
- **The viewer's mean-moment arrows now update with the τ slider** (`viz/mfa_viewer.py`):
  the animation frames only re-sent the density meshes, so the arrows stayed frozen at the
  first frame's `m` for every τ. Each frame now rebuilds every arrow-size group from that
  frame's per-atom `|m|` (visibility is left to the arrow-size slider). The README notes
  the arrow is a magnitude-only indicator (signed `m` is in the JSON; read the sign from
  the sphere colouring).

### Changed (breaking) — pre-registration API polish

- **The deprecated `MultipoleField` binding is removed** (unreleased package; the cheap
  window is now). Use `MultipoleModel`.
- **`MeanFieldEngine` renames**: `_site_potential` → `site_potential`, `_field_scale` →
  `field_scale` — they were de-facto public (viz layer, docs, tests) while `_`-prefixed,
  and the submodule exported `_` names (a private-marker / export contradiction). The
  engine now exports only the public primitives; the remaining `_` internals are imported
  by the parent via explicit qualification.
- The public-but-unexported tier is now declared with the Julia **`public` keyword**
  (`MultipoleModel`, the viz plumbing, `MeanFieldEngine`, `VASP`), machine-checkable via
  `Base.ispublic` / Aqua.
- Tests / docs build synthetic predictors with the core's new public
  `SCEPredictor(basis, j0, jphi)` — the last SALC-internal touch (`basis.salc_basis.keys`)
  is gone. Fixtures that passed a **short** `jphi` through the old unvalidated 4-argument
  constructor (relying on the other SALCs being implicitly ignored) now pass explicit
  zero-padded vectors.

### Added — guards, shared constants, registration infra

- Negative reduced temperatures are rejected loudly (`sample`, `mfa_sublattice_m`,
  `thermal_averaged_m`) instead of silently aliasing the fully ordered limit.
- `_multipole_state` clamps its Anderson iterate to the lmax-implied bound
  `√((2·lmax+1)/4π)` (a fixed ±1.5 would clip a legitimate `⟨Z_l0⟩` for l ≥ 14) and clamps
  the final `m` to `[−1, 1]`; the *absence* of an upper-τ shortcut in the tensorial /
  multipole solvers is now documented as deliberate (single-ion order persists above the
  exchange `T_MF`).
- The tesseral constants are bound to the core's single definition
  (`SCEFitting.Harmonics.N1/A2/B2`), so the forward mapping here and the core's inverse
  cannot drift.
- Test hardening: `fixed`/`uniform`/`randomize` pinned on the Metropolis path; the JSON
  export is parsed by a real JSON parser (new test-only JSON dep); `write_incar`'s `extra`
  kwarg (incl. `.TRUE.`/`.FALSE.`), zero-SAXIS, template-MAGMOM mismatch, `Oszicar`
  `energy_kind`, `ExchangeModel` Hermiticity/onsite-length, `MultipoleModel`-sampler
  dimension guards, an m-collection sweep, and a real rotation check for `randomize`.
- `LICENSE` (MIT) and a CI workflow (tests + Aqua + JET on Ubuntu/macOS, with the
  path-dev `SCEFitting` checked out as a sibling).

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
