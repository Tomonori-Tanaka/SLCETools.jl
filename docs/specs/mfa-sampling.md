# Spec — Mean-field spin-configuration sampling (`MFASampler`)

**Status:** in progress. Brainstormed and agreed; all design decisions are resolved
(see [Decisions](#8-decisions)). Implementation proceeds in phases P0–P5 (§7).
**P0 (engine), P1 (single global isotropic), P2 (multi-sublattice isotropic), P3
(tensorial exchange + single-ion, noncollinear), and P4 (full multipole / many-body)
are landed**; only P5 (sources & I/O) remains.

**Goal.** Generate physically representative finite-temperature spin
configurations for SCE training, instead of purely random (paramagnetic-limit)
directions. Given a magnetic model and a reference ordered state, draw spins from
the **single-site mean-field (MFA) distribution** at a controlled disorder level,
sweeping from ordered to disordered. This covers the low-energy manifold the model
must be accurate on, and supports an active-learning loop
(fit → sample around the predicted state → DFT → refit).

This generalizes Magesty's `MfaSampling` (single global magnetization, isotropic
von Mises–Fisher draw) to **per-sublattice magnetizations, tensorial exchange
(DMI + anisotropic exchange), single-ion and higher-order/many-body anisotropy, and
noncollinear references** — all through one uniform single-site engine.

---

## 1. Physics background

### 1.1 Mean-field decoupling collapses any SCE term to a single-site potential

The SCE energy is a sum of cluster terms, each a product of per-site tesseral
harmonics over an `N`-site cluster:

```
Φ_φ({e}) ~ Σ folded · ∏_{i=1}^N Z_{l_i}^{m_i}(e_{a_i}),    1 ≤ l_i ≤ lmax.
```

The MFA replaces every site **other than** `a` by its thermal average. Because the
sites in a cluster are distinct, the product factorizes
(`⟨∏ Z⟩ → ∏⟨Z⟩`), and the dependence on `e_a` is always a single
`Z_{l_a}^{m_a}(e_a)`. Summing over every cluster containing `a` gives an effective
**single-site potential that is again a finite tesseral expansion** (`l_a ≤ lmax`):

```
H_a^MF(e_a) = Σ_{l,m} h_a^{lm} · Z_l^m(e_a),

h_a^{lm} = Σ_{φ ∋ a} J_φ · folded · ∏_{b≠a in φ} ⟨Z_{l_b}^{m_b}(e_b)⟩.
```

So **body order and harmonic order do not require any special-casing**: every term
folds into the generalized molecular fields `h_a^{lm}`. This is the central reason
the design uses one single-site sampler for all cases.

### 1.2 Order parameters are the multipole averages `⟨Z_lm⟩`

The self-consistent order parameters generalize from the magnetization to the full
set of multipole averages per site/sublattice:

| `l` | multipole | physics | effect on the single-site distribution shape |
|----:|-----------|---------|----------------------------------------------|
| 1 | dipole = magnetization `m ê` | ferro/antiferro order | von Mises–Fisher (vMF) cone |
| 2 | quadrupole (nematic) | biquadratic exchange, single-ion anisotropy | Bingham factor (elliptical / girdle / bimodal) |
| 3 | octupole | 3-body chirality, … | 3-fold tilt |
| ≥4 | higher | cubic anisotropy, ring exchange, … | faceted / multi-lobed |

The single-site Boltzmann distribution is then

```
P(e_a) ∝ exp[ −β · H_a^MF(e_a) ] = exp[ −β Σ_{lm} h_a^{lm} Z_l^m(e_a) ].
```

`l = 1` fields give a vMF cone; `l = 2` give a Bingham shape (where single-ion
anisotropy and two-ion quadrupolar order **merge into the same effective field**);
`l ≥ 3` give faceted shapes. Closed-form samplers (vMF / Fisher–Bingham–Kent) only
span `l ≤ 2`, so `l ≥ 3` (any 3-body term, biquadratic-and-up, cubic anisotropy)
**requires the general Metropolis engine**.

### 1.3 Which part of the shape is self-consistent

- **Self-consistent (neighbor-derived):** every `h_a^{lm}` that contains a
  neighbor multipole `⟨Z_{l_b}⟩` — the whole vMF cone (width `κ_a` and, in the full
  version, axis) and all dynamically-generated higher multipole fields. As `T` rises
  and neighbors disorder, these collapse.
- **Fixed input (single-ion):** the on-site anisotropy term (`l = 2` single-ion,
  higher-order single-ion) is a temperature-independent on-site potential; its
  absolute strength does not change, but its **relative** weight grows as the
  molecular field collapses. Consequence: near and above the ordering temperature
  the sampler draws an **anisotropy-weighted paramagnet** (easy-axis/plane preference
  persists), not a uniform sphere — more physical than the isotropic high-`T` limit.

### 1.4 Scale invariance — why the absolute `Tc` need not be correct

Parametrize by the **reduced temperature** `τ = T / T_MF`, where `T_MF` is the
model's own mean-field ordering temperature. Scaling all couplings by `λ` scales
`T_MF` by `λ`, and in `β h_a = h_a / k_B T` the `λ` cancels. Hence the per-sublattice
curves `m_a(τ)` (and all multipole curves) depend only on the **ratios** of the
couplings, not their overall scale. An approximate model — a cheap SCE, or an
external Lichtenstein `Jij` off by an overall factor — still gives the correct
**relative** sublattice disordering. This is the headline advantage over a naive
single global `m`.

---

## 2. Architecture

```
AbstractSampler            # dispatch seam; future Metropolis-MC / spin-spiral samplers slot in
 └─ MFASampler             # single-site mean-field sampler (this spec)

ExchangeModel              # neutral bilinear+single-ion carrier (tensorial 𝓙_ij, A_i)
 ├─ from an SCEModel       # isotropic+DMI+anisotropic bilinear via the to_sunny extraction; single-ion via ls=[2]
 └─ from external Jij      # (i, j, R, 𝓙::SMatrix{3,3}) lists (Lichtenstein / TB2J)

sample(sampler; …) -> configs    # the one verb; returns 3×n_atoms unit-direction matrices
```

Three sampler constructions, increasing in fidelity:

1. `MFASampler(seed)` — single global magnetization, direct `τ`/`m`, isotropic vMF.
   No model needed. Magesty-equivalent; validates the engine.
2. `MFASampler(exch::ExchangeModel; reference)` — per-sublattice MFA on a tensorial
   bilinear + single-ion model. Covers external `Jij` and the bilinear truncation of
   an SCE.
3. `MFASampler(model::SCEModel; reference)` — full multipole MFA over **all** SCE
   clusters and harmonic orders (higher-order / many-body). The most faithful; only
   the SCE source carries the higher-`l` structure.

The single-site engine is shared by all three:
- **`:vmf`** — closed-form von Mises–Fisher draw. Fast path when a site's effective
  field is purely `l = 1` (isotropic).
- **`:metropolis`** — single-site Metropolis on `H_a^MF(e_a)`. General: handles
  DMI, anisotropy, and any higher-order single-site potential with no special-casing.

The mean-field self-consistency for `{⟨Z_lm⟩_a}` is solved by deterministic sphere
quadrature (accurate), and the final configuration draws use the engine above.

### Reuse of existing machinery

- **`to_sunny` / `_l1_pair_matrix` / `_l2_onsite_matrix`** already extract the full
  3×3 per-bond bilinear matrix (iso + DMI + anisotropic) and the single-ion matrix
  from an `SCEModel`. `ExchangeModel(model)` is essentially that extraction without
  the Sunny assembly.
- **`evaluate` / `accumulate_grad!`** (the folded-tensor contraction kernels) compute
  `h_a^{lm}` by contracting `folded` against neighbor multipoles `⟨Z_b⟩` while leaving
  site `a`'s axis free — the same kernel structure, neighbor `Z(e_b)` replaced by
  `⟨Z_b⟩`.
- **Symmetry orbits** (`clusters/orbits.jl`) define the sublattice partition together
  with the reference directions.

---

## 3. Public API (sketch)

`sample` always returns an `MFASample` (a labeled result — D1): `.configs` is the
bare `Vector{Matrix{Float64}}`, with parallel per-config labels `.tau` and `.m`
(per-sublattice magnetization at that config's τ). The object is iterable as its
configs.

```julia
# ── single global m (Magesty-equivalent; no model) ───────────────────────────
sampler = MFASampler(seed)                       # seed :: 3 × n_atoms reference directions
samp    = sample(sampler, 200; tau = 0.6, rng = MersenneTwister(0))   # -> MFASample
samp.configs                                     # Vector{Matrix{Float64}}, each 3 × n_atoms
samp.tau, samp.m                                 # per-config labels
samp    = sample(sampler, 200; m = 0.8, rng = rng)                    # by magnetization instead

# ── per-sublattice MFA from a tensorial exchange model ───────────────────────
exch    = ExchangeModel(scemodel)                # iso + DMI + anisotropic bilinear + single-ion
exch    = ExchangeModel(crystal, bonds; onsite)  # external: bonds = (i, j, R, 𝓙::SMatrix{3,3}); onsite = A_i
sampler = MFASampler(exch; reference = seed)     # or reference = "POSCAR" (VASP MAGMOM)
samp    = sample(sampler; tau = range(0.1, 0.95; length = 10), nsamples = 20, rng = rng)

# ── full multipole MFA off the SCE model (higher-order / many-body) ──────────
sampler = MFASampler(scemodel; reference = seed)
samp    = sample(sampler; tau = range(0.1, 0.95; length = 10), nsamples = 20, rng = rng)

# ── feed the existing pipeline (energies/torques come from DFT, out of band) ──
ds = SCEDataset(basis, samp.configs, energies)   # or (…, energies, torques)
f  = fit(SCEFit, ds, Ridge(lambda = 1e-4))

# ── inspection helpers ───────────────────────────────────────────────────────
mfa_sublattice_m(sampler, 0.6)                   # per-sublattice m_a(τ)
mfa_multipoles(sampler, 0.6)                     # ⟨Z_lm⟩ per sublattice (full version)
mfa_temperature_scale(sampler)                   # T_MF, so a caller can convert T = τ·T_MF (D4)
```

Common keywords carried over from Magesty's `mfa_sweep` (keep, naming TBD):
`nsamples`, `fixed` (atoms held at the reference), `uniform` (atoms forced fully
random), `randomize` (per-config global rotation; diversifies the global frame for
anisotropic/SOC training). RNG is an explicit `rng::AbstractRNG` (reproducible,
matching the GLMNet-fold convention) — not the global `rand()` Magesty used.

### Control variable

`τ = T / T_MF` (reduced; scale-free — §1.4) is the **only** temperature control; the
single global sampler also accepts `m` directly. Absolute `T` is intentionally not an
input — the MFA scale is not quantitative (§1.4) — but `mfa_temperature_scale(sampler)`
exposes `T_MF` so a caller can convert `T = τ·T_MF` if they trust the coupling scale (D4).

---

## 4. Algorithm

### 4.1 Self-consistency (per reduced temperature τ)

1. Initialize multipoles from the reference: `⟨Z_lm⟩_a` at full order (`τ = 0`).
2. Build the generalized molecular fields `h_a^{lm}` (§1.1) from the current
   neighbor multipoles and the model couplings.
3. For each site/sublattice, form `H_a^MF(e_a)` and compute the updated multipoles
   `⟨Z_lm⟩_a = ∫_{S²} Z_lm(e) P(e) de` by sphere quadrature.
4. Iterate 2–3 to convergence (damped fixed point).

`T_MF` is obtained from the linearized self-consistency (largest eigenvalue of the
`l = 1` molecular-field matrix), so the user-facing `τ` needs no external `Tc`.

### 4.2 Configuration draws

Per τ, with the converged `h_a^{lm}`: draw `nsamples` configurations, each spin `a`
independently from `P(e_a) ∝ exp(−β H_a^MF(e_a))` via the `:vmf` fast path
(`l = 1`-only site) or `:metropolis` (otherwise). Per-atom magnitudes are irrelevant
(configs are unit directions; magnitude is stored separately in `SpinDatum`).

---

## 5. Sources and I/O

- **SCE model → `ExchangeModel` / direct `SCEModel` sampler.** The “simple SCE near a
  reference state” path. Bilinear extraction reuses `to_sunny`; full multipole uses
  the model directly.
- **External `Jij` → `ExchangeModel`.** A neutral `(i, j, R, 𝓙::SMatrix{3,3})` +
  on-site `A_i` constructor. First external reader target: **TB2J** (Lichtenstein
  tensorial `Jij`; already vendored under `~/Packages`).
- **Reference + sublattices from a VASP input.** A POSCAR (geometry) plus `MAGMOM`
  (the `3 × n_atoms` initial moments) supplies **both** the reference directions and
  the sublattice partition in one file (sublattice = symmetry orbit × reference
  direction). VASP only for now; the reader lives in the `VASP` submodule alongside
  `read_poscar`. Explicit `3 × n_atoms` matrices are always accepted.
- **(Optional) DFT-input writers** for the drawn configs (POSCAR/INCAR per config),
  mirroring Magesty's `sample_mfa_incar`, kept code-agnostic in core + a thin VASP
  adapter.

---

## 6. Conventions and linked sites

- Spins are unit vectors, layout `3 × n_atoms`; configs are `Vector{Matrix{Float64}}`
  (the pipeline's existing config type). The sampler returns exactly this.
- Tesseral harmonics `Z_lm` and the `(4π)^(N/2)` scale are shared with the basis;
  the molecular-field contraction must use the **same** `Z_lm` and folded-tensor
  conventions as `evaluate` (linked site — change one, check both).
- The exchange extraction shares the `to_sunny` matrix formulas and their energy-
  reconstruction gate (`_reconstruct_energy ≈ predict_energy − j0`): if the harmonic
  normalization or the `(4π)^(N/2)` scale moves, both move together.
- Reference equilibrium: the molecular field at the reference should be parallel to
  `ê_a`. The sampler **verifies** this and warns when the supplied reference is not a
  stationary state of the model (the rigid-direction MFA is then inconsistent).
- RNG is explicit and seeded; a fixed seed reproduces a draw within a Julia
  session/version.

---

## 7. Staging (phases)

Each phase is independently validatable; the isotropic special case checks against
Magesty's `MfaSampling`.

- **P0 — engine.** Single-site potential `H^MF(e) = Σ h^{lm} Z_lm(e)`, sphere
  quadrature for `⟨Z_lm⟩`, `:vmf` closed form, `:metropolis` general draw. Unit:
  `:metropolis` reproduces vMF moments for an `l = 1`-only field.
- **P1 — single global, isotropic.** `MFASampler(seed)` with direct `τ`/`m`. Validates
  the Langevin self-consistency and the vMF draw against Magesty.
- **P2 — multi-sublattice, isotropic bilinear.** `ExchangeModel` from an SCE
  (isotropic part) + coupled `m_a(τ)` self-consistency; per-sublattice vMF. Validates
  on a ferrimagnet (distinct `m_a` collapse rates).
- **P3 — tensorial.** Full bilinear tensor (DMI + anisotropic exchange) + single-ion;
  `l = 1, 2` multipoles; Fisher–Bingham shapes via `:metropolis`; **noncollinear
  references**.
- **P4 — higher-order / many-body.** Full multipole MFA over all SCE clusters and `l`;
  many-body factorization through the folded-tensor contraction.
- **P5 — sources & I/O.** VASP reference reader (POSCAR + `MAGMOM`); external `Jij`
  (TB2J); optional DFT-input writers.

---

## 8. Decisions

All resolved. Conservative, exactness-leaning defaults with opt-in escapes/extensions.

- **D1 — Output shape: labeled object.** `sample` returns an `MFASample` with
  `.configs::Vector{Matrix{Float64}}` plus parallel per-config labels `.tau` and `.m`
  (per-sublattice magnetization); the object is iterable as its configs. The bare
  vector is `samp.configs`. Provenance is kept for stratification / diagnostics /
  active-learning weighting at no cost to the simple path.
- **D2 — Direction self-consistency: rigid `ê_a` first, `ê_a(τ)` as an extension.**
  Fix the reference directions at `τ = 0` and solve only magnitudes / multipoles. This
  is **exact for any collinear reference** (the molecular field stays ∥ `ê_a`), and an
  approximation only for noncollinear references with asymmetric sublattice
  disordering — where the full `ê_a(τ)` (cone axis rotates) is a later extension. The
  sampler verifies the reference is a stationary state (molecular field ∥ `ê_a`) and
  warns otherwise.
- **D3 — Multipole cutoff: full `lmax` for the SCE sampler, opt-in cap.** Iterate all
  `l ≤ lmax` multipoles (the quadrature produces them anyway); a `multipole_lmax`
  keyword caps it for speed. Truncating below `lmax` would drop the self-consistent
  feedback of `l ≥ 3` terms (3-body chirality, cubic anisotropy) that are the point of
  the arbitrary-order SCE. The `ExchangeModel` (bilinear) sampler is inherently `l ≤ 2`.
- **D4 — Temperature control: reduced `τ` only, expose `T_MF`.** No absolute-`T` input
  (the MFA scale is not quantitative); `mfa_temperature_scale(sampler)` returns `T_MF`
  so a caller can convert `T = τ·T_MF` themselves if the coupling scale is trustworthy.
- **D5 — External reader: raw constructor always, TB2J first.**
  `ExchangeModel(crystal, bonds; onsite)` (raw `(i,j,R,𝓙)` + `A_i`) is always available;
  a **TB2J** reader (Lichtenstein tensorial `Jij`) lands in P5. Other formats on demand.

---

## 9. Validation plan

- **Engine (P0):** `:metropolis` on an `l = 1` field reproduces the vMF mean resultant
  length `L(κ)` and `⟨cosθ⟩`; sphere quadrature `⟨Z_lm⟩` matches analytic vMF moments.
- **Isotropic self-consistency (P1):** `m(τ)` matches the Langevin/Brillouin curve;
  `τ → 0` returns the reference, `τ → 1⁺` returns uniform; cross-check Magesty.
- **Multi-sublattice (P2):** on a two-sublattice ferrimagnet, `m_a(τ)` collapse rates
  match the analytic coupled MFA; scale-invariance check — scaling all `Jij` leaves
  `m_a(τ)` unchanged.
- **Tensorial (P3):** easy-axis single-ion → cone sharpening / bimodality; easy-plane
  → girdle; the sampled `⟨Z_2m⟩` matches the quadrature self-consistency; DMI on a
  canted reference gives the expected tilt.
- **Higher-order (P4):** a known 3-body / biquadratic model reproduces its analytic
  multipole self-consistency; many-body factorization matches a direct mean-field
  evaluation.
- **End-to-end:** MFA-sampled configs fed through `SCEDataset` → `fit` recover known
  couplings at least as well as random-direction sampling, with better low-`T`
  coverage. `numerical-reviewer` pass on the self-consistency and the molecular-field
  contraction.

---

## 10. Task list (coarse)

- [x] **P0 single-site engine** (`src/sampling/site_engine.jl`): potential eval, vMF
      closed form, symmetric-proposal Metropolis, field-aware Gauss–Legendre × azimuth
      quadrature for `⟨Z_lm⟩`. Tests: all three paths reproduce Langevin `L(κ)`, Metropolis
      matches quadrature on a non-vMF `l=2` field, quadrature auto-sizes to sharp peaks.
- [x] **P1 single global, isotropic** (`src/sampling/mfa_sampler.jl`): `AbstractSampler`
      seam, `MFASampler(reference)`, the `MFASample` labeled output (D1), and the `sample`
      verb (scalar `n` form + collection sweep). Langevin self-consistency `m = L(3m/τ)`
      via self-written bisection (no Roots dep); `τ ↔ m` inversion; `mfa_temperature_scale`
      (D4, `T_MF = 1` in reduced units); `fixed`/`uniform`/`randomize` keywords. Tests:
      self-consistency residual + inverse round-trip, ordered / near-uniform boundary
      limits, drawn configs carry `⟨cosθ⟩ = m(τ)`, reproducibility. Numerically equivalent
      to Magesty's `MfaSampling`.
- [x] **P2 multi-sublattice, isotropic** (`src/sampling/exchange.jl`): `ExchangeModel`
      (symmetric `Jiso[a,b] = Σ_R J_iso(a,b,R)`, from a fitted SCE via the Sunny bilinear
      extraction `tr(M)/3`, or a raw matrix) + the coupled per-atom self-consistency
      `m_a = L(3(Ā m)_a/τ)` with the normalized molecular-field matrix `Ā = A/ρ`,
      `A[a,b] = −Jiso[a,b](ê_a·ê_b)` (folding the reference makes ferro/antiferro/ferri
      ferromagnetic in the magnitudes), `T_MF = ρ/3` from the Perron eigenvalue, per-atom
      vMF `κ_a`. Anderson-accelerated solve (critical slowing near `T_MF`); Perron sign-
      definiteness and reference-stationarity (`h_a ∥ ê_a`) checks. Tests: reduces to the
      single-global Langevin curve for a uniform ferro/antiferromagnet, ferrimagnet gives
      distinct sublattice collapse rates at one `T_MF`, scale invariance (`max|Δm| ≈ 1e-12`),
      free-spin handling, SCE extraction + dropped-channel warnings.
- [x] **P3 tensorial** (`src/sampling/exchange.jl`, `exchange_from_sce.jl`): the full
      bilinear tensor `S_ab` (Heisenberg + DMI + anisotropic) and single-ion `A_a` (`ls=[2]`),
      from a fitted SCE (only higher-order channels dropped) or a raw `bilinear`/`onsite`.
      Single-site potential `V_a(e) = β(e·g_a + e' A_a e)`, `g_a = Σ_b S_ab m_b ê_b`,
      `β = 3/(ρτ)`; longitudinal `A[a,b] = −ê_a' S_ab ê_b` (generalizes P2), `T_MF = ρ/3`.
      The l=2 single-ion gives a **Bingham** single-site shape (closed-form vMF cannot
      represent it), so the self-consistency `m_a = ⟨e·ê_a⟩` uses the quadrature and the
      draw uses the **Metropolis** engine. Tesseral-coefficient conversions (`_l1_coeffs!`/
      `_l2_coeffs!`, exact inverses of the Sunny `_l1_pair_matrix`/`_l2_onsite_matrix`).
      **Noncollinear references** (rigid-axis D2, with a stationarity warning). Tests:
      easy-axis cone sharpening / above-`T_MF` persistence, easy-plane girdle, Metropolis ↔
      quadrature agreement on `⟨Z_2m⟩`, DMI tilt of a collinear reference, the τ → 0 limit.
- [x] **P4 full multipole MFA** (`src/sampling/exchange.jl`, `exchange_from_sce.jl`):
      `MultipoleField` digests **every** SCE cluster term; the generalized molecular field
      `h_a^{lm} = Σ_φ jφ·(4π)^(N/2)·folded · ∏_{b≠a} ⟨Z_{l_b}^{m_b}(e_b)⟩` is built by
      `_site_coeffs_all!` (the `accumulate_grad!` leave-one-out structure with the site-a
      harmonic left symbolic). The order parameters are the **full per-atom multipole
      averages `⟨Z_lm⟩_a`** (`l ≤ lmax`), iterated to self-consistency by quadrature;
      `β = 3/(ρτ)` with `ρ` the `l=1` (bilinear) Perron; the draw is Metropolis.
      `MFASampler(model::SCEModel; reference)` is the user entry. Validated by the exact
      reduction to the single-global Langevin curve for a pure-bilinear model, scale
      invariance, and — the headline — the many-body factorization checked against the
      conditional mean SCE energy `⟨E|e_a⟩` of a biquadratic model (matches to ~MC noise;
      a deterministic cross-check confirms `V_a/β = ⟨E|e_a⟩` to machine precision).
- [ ] P5 VASP reference reader; external `Jij` (TB2J); optional DFT writers.
- [ ] Docs: a Documenter guide page + a tutorial; design-note cross-reference.

Design decisions D1–D5 are settled (§8).
