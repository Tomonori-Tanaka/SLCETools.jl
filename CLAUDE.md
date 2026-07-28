# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Auxiliary tooling around the SLCE fitting core
[`SLCE.jl`](../SLCE.jl): utilities that **consume** a fitted
`SLCEModel` rather than build one. The first component is the **mean-field (MFA)
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

This package depends on `SLCE` and reads a fitted model **only** through its
public introspection surface — `spin_multipole_terms`, `bilinear_terms`, `n_atoms(model)`,
and the tesseral spherical-harmonic submodule `SLCE.Harmonics` (`Zlm`,
`lm_index`) — never the SALC-basis internals (`model.basis.salc_basis.salcs`, `SALCMember`,
`SALCTerm`). During development the dependency is a path-dev:
`Pkg.develop(path="../SLCE.jl")`.

## Numerical / physics conventions

Inherited from the core (`SLCE`'s `CLAUDE.md`); the ones this package leans on:

- **Spin directions are unit vectors**; configuration layout `3 × n_atoms` (rows x,y,z;
  columns atoms). The `reference` is the same layout.
- **Real (tesseral) spherical harmonics `Zₗₘ`**, per-site factor `(4π)^(−1/2)`; an N-body
  SLCE term carries `(4π)^(N/2)`. `spin_multipole_terms` returns the **raw** fitted `jϕ`; this
  package applies the `(4π)^(body/2)` scale once, in `_scaled_multipole_terms`
  (`mfa/bridge.jl`), shared by `MultipoleModel` and `MetropolisSampler` — never re-apply
  downstream.
- **Reduced temperature `τ = T/T_MF`** with `T_MF = ρ/3` from the `l=1` (bilinear) Perron
  eigenvalue `ρ`, `β = 3/(ρτ)`. Scale invariance: only coupling *ratios* matter.
- **Mean-field decoupling** `⟨∏ Z⟩ → ∏⟨Z⟩` (assumes distinct sites per cluster — asserted).
  The single-site distribution is vMF (`l=1`, closed form) or a Bingham / higher-multipole
  shape (`l≥2`, drawn by the Metropolis engine; magnetizations by sphere quadrature).

## Coupled ("linked") code sites — change one, check all

- **Names this package borrows rather than owns** (`src/SLCETools.jl` header):
  `n_atoms` is a **method of `SLCE`'s generic** (`import SLCE: n_atoms`, extended for
  `ExchangeModel` / `MultipoleModel`), and `KB_EV` / `resolve_kt` are `using`-ed from
  `SLCE` and merely re-published. Do not re-define them here: this package and
  SLCEMonteCarlo each carried a private, character-for-character-identical `KB_EV`
  until the 2026-07-28 family naming batch — two copies of a unit conversion that
  could drift with neither suite able to see the other. A struct that counts atoms
  spells the field `n_atoms` and gets an `n_atoms(::T)` method, so the accessor reads
  the same on a `Crystal`, an `SLCEModel` and a coupling digest.

- **`mfa/bridge.jl` ↔ the core's introspection contract** (`SLCE`'s
  `slce/introspect.jl`): `_scaled_multipole_terms` consumes `spin_multipole_terms` and applies
  `coef·(4π)^(body/2)` (feeding both `MultipoleModel(model)` and
  `MetropolisSampler(model)`); `ExchangeModel(model)` consumes `bilinear_terms` (the `3×3`
  bilinear / single-ion matrices). If a `SpinMultipoleTerm` field or the scale convention
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
- **`mfa/engine.jl` (`MeanFieldEngine`) ↔ `SLCE.Harmonics`** (`Zlm`, `lm_index`): the
  engine primitives (`site_potential`, the vMF / Metropolis draws, the quadrature) evaluate
  tesseral harmonics through the core submodule (bound here by `import SLCE.Harmonics`,
  on the unit-direction `Zlm_unsafe` fast path — `Harmonics` is part of the core's declared
  (`public`-keyword) stable surface, `Zlm_unsafe` included). A normalization change upstream
  shifts every multipole average.
  Gate: `test_multipole.jl` "many-body factorization: `V_a/β` equals the conditional mean
  energy ⟨E|e_a⟩" — it fences `site_potential` against `SLCE.predict_energy`, so the
  `(4π)^(body/2)` scale and the `μ = idx − l − 1` ↔ `lm_index` correspondence cannot drift.
  Do NOT add a "normalization pin" on the harmonics as the engine consumes them: the engine
  derives its own constant from `Harmonics` (`_l1_field` reads `Zlm` at the three Cartesian
  axes; `test_site_engine.jl` divides by `N10`), so such a pin self-cancels and would be
  green before and after the change it claims to catch. A pure, consistent renormalization
  upstream is *supposed* to leave this gate green — it is harmless-with-refit; the hard
  literals that would flag a reproducibility break live upstream in `test_harmonics.jl`.
- **`mfa/exchange.jl` `_l1_coeffs!` / `_l2_coeffs!`** (field → tesseral coefficients) are the
  *forward* of the core's `_l1_pair_matrix` / `_l2_onsite_matrix` (tesseral → `3×3`, in the
  core's `slce/bilinear.jl`). The tesseral constants `_N1`/`_A2`/`_B2` are **bound to**
  `SLCE.Harmonics.N1/A2/B2` (single definition upstream), so the forward and inverse
  conversions cannot drift apart. The bilinear extraction uses the core's (inverse) matrices
  via `bilinear_terms`; do not duplicate that delicate conversion here.
  Gates: `test_tensorial.jl` "the l=1/l=2 coefficient writers reproduce their forms against
  Zlm" (semantic, against `Zlm` itself) + "single-ion anisotropy is rotationally covariant"
  (end-to-end through `MFASampler`, with a control that must fail). Both are needed and
  both must use a **non-diagonal, non-traceless** `A`: on a diagonal tensor four of the five
  `l=2` branches are identically zero, which is what every other single-ion test in this
  package feeds them — a swapped `axz ↔ ayz` was silent across the whole suite until these
  landed (verified by mutation: 635 pre-existing assertions stayed green). Do NOT "simplify"
  either gate into a round-trip against `_l2_onsite_matrix`: a round-trip is satisfied by any
  pair of mutually consistent but jointly wrong conventions, and in this direction it is not
  even the identity — `_l2_coeffs!` discards the trace and the antisymmetric part.
- **`io/vasp.jl` — read ↔ write inverse-consistency** (`SLCETools.VASP`, one module holds both):
  (1) **SAXIS frame** — one `_saxis_rotation` (`R = Rz(α)Ry(β)`) serves both; the reader rotates
  SAXIS → Cartesian by `R`, the writer Cartesian → SAXIS by `Rᵀ`, so a write → read round-trip is
  the identity. The INCAR's *declared* SAXIS line and the frame the moments are written in must
  always agree (template SAXIS honoured / overridden together). (2) **Atom order** —
  `_poscar_order` must reproduce `write_poscar`'s species grouping exactly, or `write_inputs`
  silently misassigns moments to atoms. (3) **MAGMOM = magnitude · direction**, M_CONSTR ==
  MAGMOM under `constrain`. (4) The **torque sign / TrainingDatum layout** is owned upstream by
  `SLCE`'s `dftsource.jl` (`τ_a = m_a × B_a`); the OSZICAR reader must keep producing
  that. (5) **Absent ≠ zero**: an OSZICAR with no `lambda*MW_perp` block yields
  `field = nothing` (2-arg `spin_datum`, `torque_qualified = false`) — never a fabricated
  zero-filled field, which would claim `τ = 0` was observed and admit false torque rows
  into a co-fit; a present block with zero rows means "computed, those atoms
  unconstrained". `Oszicar(...; setup_id = ...)` stamps the computational-setup label
  (`SLCEDataset` rejects cross-setup mixtures). Gates: `test/unit/test_vaspio.jl` (read),
  `test/unit/test_vasp_incar.jl` (write, round-trip / order / formatting), `test/oracle/`
  (parsers vs Magesty bit-for-bit). The sampler gives only directions + an order parameter
  `m_a ∈ [0,1]`, **not** μ_B magnitudes — the write magnitudes are an external input.

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |
| `julia --project=test/oracle test/oracle/runtests.jl` | VASP parsers vs pinned Magesty |

The suite (`test/runtests.jl`) dispatches on `TEST_MODE`
(`default`/`all`/`unit`/`aqua`/`jet`). It needs `SLCE` available (path-dev).
The oracle suite is a separate environment carrying the pinned `Magesty.jl` the
core suite deliberately omits.

## References

- `SPEC.md` — architecture, primary types, public API, and the planned active-learning layer.
- `docs/specs/mfa-sampling.md` — the mean-field sampler design spec (D1–D5, P0–P4).
- `references/` — supporting literature (notes tracked, PDFs local-only).
