# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Auxiliary tooling around the SCE fitting core
[`SCEFitting.jl`](../SCEFitting.jl): utilities that **consume** a fitted
`SCEPredictor` rather than build one. The first component is the **mean-field (MFA)
spin-configuration sampler** — draw physically representative finite-temperature spin
configurations from the single-site mean field of a fitted model (or a hand-built
exchange model) at a controlled reduced temperature `τ = T/T_MF`. Its joint-Boltzmann
sibling is the **Metropolis MC sampler** (`src/mc/metropolis.jl`, `MetropolisSampler`):
correlated configurations on the training cell at an **absolute** temperature —
`temperature` [K, via `KB_EV`, assumes an eV model] or `kT` [model energy units], exactly
one (deliberately *not* `τ`; the model need not have a bilinear channel). Future
components (active learning, configuration / diagnostic helpers) live alongside. Priority:
numerical correctness and reproducibility, and the same physical conventions as the
core (this package never re-derives them — it reads the fitted Hamiltonian through the
core's public surface).

This package depends on `SCEFitting` and reads a fitted model **only** through its
public introspection surface — `multipole_terms`, `bilinear_terms`, `n_atoms(model)`,
and the tesseral spherical-harmonic submodule `SCEFitting.Harmonics` (`Zlm`,
`lm_index`) — never the SALC-basis internals (`model.basis.salc_basis.salcs`, `SALCMember`,
`SALCTerm`). During development the dependency is a path-dev:
`Pkg.develop(path="../SCEFitting.jl")`.

## Numerical / physics conventions

Inherited from the core (`SCEFitting`'s `CLAUDE.md`); the ones this package leans on:

- **Spin directions are unit vectors**; configuration layout `3 × n_atoms` (rows x,y,z;
  columns atoms). The `reference` is the same layout.
- **Real (tesseral) spherical harmonics `Zₗₘ`**, per-site factor `(4π)^(−1/2)`; an N-body
  SCE term carries `(4π)^(N/2)`. `multipole_terms` returns the **raw** fitted `jϕ`; this
  package applies the `(4π)^(body/2)` scale once, in `_scaled_multipole_terms`
  (`mfa/bridge.jl`), shared by `MultipoleModel` and `MetropolisSampler` — never re-apply
  downstream.
- **Reduced temperature `τ = T/T_MF`** with `T_MF = ρ/3` from the `l=1` (bilinear) Perron
  eigenvalue `ρ`, `β = 3/(ρτ)`. Scale invariance: only coupling *ratios* matter.
- **Mean-field decoupling** `⟨∏ Z⟩ → ∏⟨Z⟩` (assumes distinct sites per cluster — asserted).
  The single-site distribution is vMF (`l=1`, closed form) or a Bingham / higher-multipole
  shape (`l≥2`, drawn by the Metropolis engine; magnetizations by sphere quadrature).

## Coupled ("linked") code sites — change one, check all

- **`mfa/bridge.jl` ↔ the core's introspection contract** (`SCEFitting`'s
  `sce/introspect.jl`): `_scaled_multipole_terms` consumes `multipole_terms` and applies
  `coef·(4π)^(body/2)` (feeding both `MultipoleModel(model)` and
  `MetropolisSampler(model)`); `ExchangeModel(model)` consumes `bilinear_terms` (the `3×3`
  bilinear / single-ion matrices). If a `MultipoleTerm` field or the scale convention
  changes upstream, this file and `mc/metropolis.jl` move with it. The regression gates
  are the P1–P4 suite (`test/unit/test_{multipole,tensorial,exchange}.jl`: exact
  reduction to the single-global Langevin curve, scale invariance, the many-body
  factorization `V_a/β = ⟨E | e_a⟩` to machine precision) and the MC suite
  (`test/unit/test_mc_sampler.jl`: `ΔE = c_a·ΔZ` ≡ the `predict_energy` difference to
  machine precision, the exact two-spin `⟨e₁·e₂⟩ = −L(βJ)`).
- **`mc/metropolis.jl` ↔ the mean-field kernels**: `_accumulate_site_term!` /
  `_term_energy` are the single-site / full-contraction siblings of
  `selfconsistency.jl`'s `_accumulate_term!` (same `μ = idx − l − 1` mapping and
  rank-specialized barrier, concrete `Z(e_b)` instead of `⟨Z⟩`), and the sweep reuses
  the engine's proposal (`_rotate` + `_METROPOLIS_FLIP_FRACTION`) and the MFA sampler's
  `_random_rotation` / `_normalize_reference`. Change one side and re-check the
  machine-precision local↔global gate in `test_mc_sampler.jl`.
- **`mfa/engine.jl` (`MeanFieldEngine`) ↔ `SCEFitting.Harmonics`** (`Zlm`, `lm_index`): the
  engine primitives (`site_potential`, the vMF / Metropolis draws, the quadrature) evaluate
  tesseral harmonics through the core submodule (bound here by `import SCEFitting.Harmonics`,
  on the unit-direction `Zlm_unsafe` fast path — `Harmonics` is part of the core's declared
  (`public`-keyword) stable surface, `Zlm_unsafe` included). A normalization change upstream
  shifts every multipole average.
- **`mfa/exchange.jl` `_l1_coeffs!` / `_l2_coeffs!`** (field → tesseral coefficients) are the
  *forward* of the core's `_l1_pair_matrix` / `_l2_onsite_matrix` (tesseral → `3×3`, in the
  core's `sce/bilinear.jl`). The tesseral constants `_N1`/`_A2`/`_B2` are **bound to**
  `SCEFitting.Harmonics.N1/A2/B2` (single definition upstream), so the forward and inverse
  conversions cannot drift apart. The bilinear extraction uses the core's (inverse) matrices
  via `bilinear_terms`; do not duplicate that delicate conversion here.
- **`io/vasp.jl` — read ↔ write inverse-consistency** (`SCETools.VASP`, one module holds both):
  (1) **SAXIS frame** — one `_saxis_rotation` (`R = Rz(α)Ry(β)`) serves both; the reader rotates
  SAXIS → Cartesian by `R`, the writer Cartesian → SAXIS by `Rᵀ`, so a write → read round-trip is
  the identity. The INCAR's *declared* SAXIS line and the frame the moments are written in must
  always agree (template SAXIS honoured / overridden together). (2) **Atom order** —
  `_poscar_order` must reproduce `write_poscar`'s species grouping exactly, or `write_inputs`
  silently misassigns moments to atoms. (3) **MAGMOM = magnitude · direction**, M_CONSTR ==
  MAGMOM under `constrain`. (4) The **torque sign / SpinDatum layout** is owned upstream by
  `SCEFitting`'s `dftsource.jl` (`τ_a = m_a × B_a`); the OSZICAR reader must keep producing
  that. Gates: `test/unit/test_vaspio.jl` (read), `test/unit/test_vasp_incar.jl` (write,
  round-trip / order / formatting), `test/oracle/` (parsers vs Magesty bit-for-bit). The sampler
  gives only directions + an order parameter `m_a ∈ [0,1]`, **not** μ_B magnitudes — the write
  magnitudes are an external input.

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |
| `julia --project=test/oracle test/oracle/runtests.jl` | VASP parsers vs pinned Magesty |

The suite (`test/runtests.jl`) dispatches on `TEST_MODE`
(`default`/`all`/`unit`/`aqua`/`jet`). It needs `SCEFitting` available (path-dev).
The oracle suite is a separate environment carrying the pinned `Magesty.jl` the
core suite deliberately omits.

## References

- `SPEC.md` — architecture, primary types, public API, and the planned active-learning layer.
- `docs/specs/mfa-sampling.md` — the mean-field sampler design spec (D1–D5, P0–P4).
- `references/` — supporting literature (notes tracked, PDFs local-only).
