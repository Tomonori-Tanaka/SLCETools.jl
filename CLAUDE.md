# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Auxiliary tooling around the SCE fitting core
[`MagestyRebuild.jl`](../Magesty_rebuild.jl): utilities that **consume** a fitted
`SCEModel` rather than build one. The first component is the **mean-field (MFA)
spin-configuration sampler** — draw physically representative finite-temperature spin
configurations from the single-site mean field of a fitted model (or a hand-built
exchange model) at a controlled reduced temperature `τ = T/T_MF`. Future components
(active learning, configuration / diagnostic helpers) live alongside it. Priority:
numerical correctness and reproducibility, and the same physical conventions as the
core (this package never re-derives them — it reads the fitted Hamiltonian through the
core's public surface).

This package depends on `MagestyRebuild` and reads a fitted model **only** through its
public introspection surface — `multipole_terms`, `bilinear_terms`, `num_atoms(model)`,
and the tesseral spherical-harmonic submodule `MagestyRebuild.Harmonics` (`Zlm`,
`lm_index`) — never the SALC-basis internals (`model.basis.salcs.salcs`, `SALCMember`,
`SALCTerm`). During development the dependency is a path-dev:
`Pkg.develop(path="../Magesty_rebuild.jl")`.

## Numerical / physics conventions

Inherited from the core (`MagestyRebuild`'s `CLAUDE.md`); the ones this package leans on:

- **Spin directions are unit vectors**; configuration layout `3 × n_atoms` (rows x,y,z;
  columns atoms). The `reference` is the same layout.
- **Real (tesseral) spherical harmonics `Zₗₘ`**, per-site factor `(4π)^(−1/2)`; an N-body
  SCE term carries `(4π)^(N/2)`. `multipole_terms` returns the **raw** fitted `jϕ`; this
  package applies the `(4π)^(body/2)` scale once, in `MultipoleField(model)` (`sce_bridge.jl`).
- **Reduced temperature `τ = T/T_MF`** with `T_MF = ρ/3` from the `l=1` (bilinear) Perron
  eigenvalue `ρ`, `β = 3/(ρτ)`. Scale invariance: only coupling *ratios* matter.
- **Mean-field decoupling** `⟨∏ Z⟩ → ∏⟨Z⟩` (assumes distinct sites per cluster — asserted).
  The single-site distribution is vMF (`l=1`, closed form) or a Bingham / higher-multipole
  shape (`l≥2`, drawn by the Metropolis engine; magnetizations by sphere quadrature).

## Coupled ("linked") code sites — change one, check all

- **`sce_bridge.jl` ↔ the core's introspection contract** (`MagestyRebuild`'s
  `sce/introspect.jl`): `MultipoleField(model)` consumes `multipole_terms` and applies
  `coef·(4π)^(body/2)`; `ExchangeModel(model)` consumes `bilinear_terms` (the `3×3`
  bilinear / single-ion matrices). If a `MultipoleTerm` field or the scale convention
  changes upstream, this file moves with it. The regression gate is the P1–P4 suite
  (`test/unit/test_{multipole,tensorial,exchange,mfa_sampler}.jl`): exact reduction to the
  single-global Langevin curve, scale invariance, and the many-body factorization check
  `V_a/β = ⟨E | e_a⟩` to machine precision.
- **`site_engine.jl` ↔ `MagestyRebuild.Harmonics`** (`Zlm`, `lm_index`): the quadrature /
  vMF / Metropolis kernels evaluate tesseral harmonics through the core submodule (bound
  here by `import MagestyRebuild.Harmonics`). A normalization change upstream shifts every
  multipole average.
- **`exchange.jl` `_l1_coeffs!` / `_l2_coeffs!`** (field → tesseral coefficients) are the
  *forward* of the core's `_l1_pair_matrix` / `_l2_onsite_matrix` (tesseral → `3×3`). The
  bilinear extraction uses the core's (inverse) matrices via `bilinear_terms`; do not
  duplicate that delicate conversion here.
- **`io/vasp.jl` (write) ↔ `MagestyRebuild.VASP` / `dftsource.jl` (read)**: the INCAR writer
  must stay inverse-consistent with the reader. (1) **Atom order** — `_poscar_order` must
  reproduce `write_poscar`'s species grouping exactly, or moments are silently misassigned to
  atoms. (2) **SAXIS frame** — the writer rotates Cartesian → SAXIS by `Rᵀ`, the inverse of the
  reader's `Rz(α)Ry(β)` (`_saxis_rotation` must match the reader's α/β); the *declared* SAXIS
  line and the frame the moments are written in must always agree (template SAXIS is honoured /
  overridden together). (3) **MAGMOM = magnitude · direction**, M_CONSTR == MAGMOM under
  `constrain`. The gate is `test/unit/test_vasp_incar.jl` (round-trip, order, formatting). The
  sampler gives only directions + an order parameter `m_a ∈ [0,1]`, **not** μ_B magnitudes — the
  magnitudes are an external input.

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |

The suite (`test/runtests.jl`) dispatches on `TEST_MODE`
(`default`/`all`/`unit`/`aqua`/`jet`). It needs `MagestyRebuild` available (path-dev).

## References

- `SPEC.md` — architecture, primary types, public API, and the planned active-learning layer.
- `docs/specs/mfa-sampling.md` — the mean-field sampler design spec (D1–D5, P0–P4).
- `references/` — supporting literature (notes tracked, PDFs local-only).
