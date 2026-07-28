# Metropolis MC sampling — design spec

> **Naming note (2026-07-28).** This is a dated decision record and is kept as
> written; the names below are the ones the decision was taken under. Renamed
> since, in the family-wide naming batch: `MultipoleTerm` → `SpinMultipoleTerm`. The current spelling is what
> the code, `SPEC.md` and the API reference use.


Status: **implemented** (`src/mc/metropolis.jl`). Companion of the mean-field sampler
spec (`mfa-sampling.md`), which reserved the `AbstractSampler` seam for exactly this
sampler.

## Goal

Draw spin configurations from the **joint Boltzmann distribution**
`P({e}) ∝ exp(−E({e})/k_BT)` of a fitted SCE — correlations included — as a sibling of
the single-site mean-field sampler, behind the same `sample` verb. The intended use is
*configuration sampling* (realistic finite-temperature training/proposal configurations,
e.g. for active learning), not thermodynamic observables.

## Decisions

- **M1 — scope: config sampler only.** No observable API (`m_s(T)`, susceptibility, `T_c`
  estimation) and no supercell tiling. The only diagnostics are what equilibration
  requires: a per-stored-config energy and the Metropolis acceptance fraction, carried on
  the result. Deferred work is listed at the end.
- **M2 — simulation cell = the training cell.** The fitted terms already fold every
  periodic image onto the training cell (`MultipoleTerm.shifts` are not re-expanded), so
  the chain is the `n_atoms`-spin periodic system the model was fitted on — exactly the
  right ensemble for training-cell configuration proposals. A larger cell (tiling the
  terms by their `shifts`) is the natural extension for observables; deferred.
- **M3 — absolute temperature, dual keyword.** The control is absolute (the mean-field
  `τ = T/T_MF` is *not* used: `T_MF` comes from the `l=1` bilinear Perron eigenvalue,
  which need not exist, and true MC needs no linearized scale), under **exactly one** of
  two keywords: `temperature` in **kelvin** (converted with `KB_EV`, the exact CODATA
  `k_B` in eV/K — assumes an eV-fitted model, the package convention) or `kT` = `k_B·T`
  directly in the model's energy units (theory/test runs in coupling units, non-eV
  models). Internally everything is `β = 1/kT`. The two live under *distinct names*
  deliberately: a single keyword serving both would let `temperature = 300` (meant as
  kelvin) be read as 300 eV — a silent infinite-temperature run. `MCSample` carries both
  labels (`kT` always well-defined; `temperature = kT/KB_EV`).
- **M4 — API parity behind `AbstractSampler`.** `MetropolisSampler <: AbstractSampler`,
  sampling through the same two `sample` forms as the mean field (positional `n` at one
  value / keyword collection sweep; the exactly-one `temperature`/`kT` rule mirrors the
  mean field's `tau`/`m`). The result is a new labeled type `MCSample` (`configs` +
  parallel `kT` / `temperature` / `energy` / `acceptance`) — the `MFASample` labels
  (`tau`, per-atom mean-field `m`) do not apply. Same array interface.
- **M5 — β in the accept step only.** Site coefficients `c_a` and every stored energy
  stay in the model's energy units; the acceptance uses `exp(−βΔE)`. (The mean-field
  kernel instead folds β into `c_a` — call sites differ, the contraction is shared.)
- **M6 — the `(4π)^(N/2)` scale is applied exactly once**, in `_scaled_multipole_terms`
  (`mfa/bridge.jl`), the digest shared with `MultipoleModel`. The MC file never
  re-applies it; the machine-precision gate against `predict_energy` pins this.
- **M7 — warm start across a temperature sweep.** The keyword-form chain state carries
  over between consecutive temperatures (fresh `burnin` at each), so a high→low list is
  an annealing run; independent chains = one call per temperature.
- **M8 — `randomize` = one Haar rotation per stored copy** (same keyword and
  `_random_rotation` as the mean-field sampler). The chain itself is never rotated; the
  stored energy is recomputed from the rotated copy so `energy[k]` always matches
  `configs[k]`. For an isotropic model the rotated draws remain exact Boltzmann samples
  while the absolute orientation — a slowly-diffusing zero mode of local updates —
  becomes exactly uniform (anisotropy-training data); for an anisotropic model it is
  data augmentation, not equilibrium sampling.

## Algorithm

One sweep = `n_atoms` sequential single-spin attempts (deterministic site order — a
composition of per-site reversible kernels keeps the Boltzmann distribution stationary
even though the composite kernel is not reversible; no RNG spent on site selection,
bit-reproducible runs; a random-permutation scan is a trivial future toggle).

Per attempt at site `a`:

1. Contract every term containing `a` against the **concrete** neighbor harmonics
   `Z_lm(e_b)` of the current configuration (`_accumulate_site_term!` — the mean-field
   leave-one-out kernel `_accumulate_term!` restricted to the one position of `a`; the
   ctor asserts `allunique(atoms)` per term, so `c_a` is independent of `e_a`).
2. Propose with the engine's symmetric mixture: antipodal flip (probability
   `_METROPOLIS_FLIP_FRACTION`) or a Rodrigues rotation about a uniform axis by
   `step·N(0,1)` radians.
3. `ΔE = c_a · (Z(e′) − Z(e))` — the *exact* total-energy change of the single-spin
   move. Accept with `min(1, exp(−βΔE))`; on acceptance update the config column and the
   cached `Z` row.

Stored-config energy: full recomputation `Σ_terms coef·⟨folded, ∏Z⟩` (no incremental
accumulation ⇒ zero drift), which by the core's introspection contract equals
`predict_energy(model, config) − j0` (`j0` is a constant, irrelevant to the chain).

Defaults mirror the engine: `step = 0.6` rad, `burnin = 200`, `thin = 10` (both in
sweeps). No step auto-adaptation — the `acceptance` diagnostic is the tuning feedback.

## Validation gates (`test/unit/test_mc_sampler.jl`)

1. **Local ↔ global, machine precision**: `c_a` ≡ the `a`-row of the mean-field
   `_site_coeffs_all!` at β=1; `ΔE = c_a·ΔZ` ≡ the `_total_energy` difference ≡ the
   `predict_energy` difference (pins M5/M6 and the `j0` exclusion).
2. **Exact two-spin gate**: ferro dimer, `⟨e₁·e₂⟩ = −L(βJ)` within MC error at
   `|βJ| = 1, 2.5`; `mean(energy) = J·⟨e₁·e₂⟩` validates the energy diagnostic itself.
3. **Single-site Langevin limit**: a hand-built `l=1` field term reproduces
   `⟨e_z⟩ = −L(βh)` — where the mean field is exact, so this doubles as the MFA
   cross-link. (A low-`T` mean-field comparison on an exchange-only cluster is *not*
   valid: exchange alone is rotation-invariant, so the true `⟨e⟩` is zero while the
   mean field breaks the symmetry by its reference.)
4. **`randomize`**: same seed ⇒ identical chains up to the first storage, so the first
   stored pair tests Haar rotation directly — equal energy (isotropy), equal Gram
   matrix, unit columns, and `energy[k] ↔ configs[k]` consistency.
5. Seed reproducibility (byte-identical, incl. sweep + randomize), sweep
   structure/warm-start (annealing lowers the ferro bond energy), guards.

Real-model smoke (Nd₂Fe₁₄B l02 refit, 68 atoms / 4692 terms): annealing 0.15 → 0.01 eV
from a *random* start recovers the ferrimagnetic order (Nd projections onto the mean Fe
axis ≈ −0.85, Fe ≈ +0.95), and the T = 0.01 eV mean energy sits ≈ `N·k_BT` above the
relaxed collinear ground state (equipartition: 2 rotational DOF × ½k_BT per magnetic
spin) — ~0.8 s for 50 stored configs at `burnin = 300`, `thin = 20`.

## Deferred (explicitly out of scope)

- **Supercell tiling** — replicate the terms over an `L₁×L₂×L₃` cell using
  `MultipoleTerm.shifts` (currently folded and unused); needed for finite-size-honest
  observables (`m(T)`, `T_c`).
- **Thermodynamic observables** — sublattice `m(T)`, susceptibility, Binder/`T_c`;
  meaningful only with M2 lifted.
- Random-permutation sweeps, step auto-adaptation, cluster updates / parallel tempering.
