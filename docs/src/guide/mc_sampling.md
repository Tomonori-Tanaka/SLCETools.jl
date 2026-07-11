# Monte-Carlo sampling

```@meta
CurrentModule = SCETools
```

[`MetropolisSampler`](@ref) is the joint-Boltzmann sibling of the mean-field
[`MFASampler`](@ref): single-spin Metropolis on the fitted SCE Hamiltonian over its
training cell, so the drawn configurations carry the model's true inter-site
correlations. The price is a Markov chain — burn-in, thinning, and an acceptance rate to
watch — instead of closed-form single-site draws.

## Construction and the `sample` verb

```julia
mc = MetropolisSampler(model)                       # model::SCEPredictor
mc = MetropolisSampler(model; reference)            # reference = default chain start

# n configurations at one absolute temperature (k_B·T, model energy units)
samp = sample(mc, 200; temperature = 0.02, rng = MersenneTwister(1))

# an annealing sweep: the chain warm-starts each next temperature
samp = sample(mc; temperature = [0.15, 0.08, 0.04, 0.02], nsamples = 50)
```

Unlike the mean-field sampler there is **no reduced temperature**: `temperature` is the
absolute ``k_B T`` in the model's energy units (eV for an eV-fitted model), so the
sampler works for any body order — including models without a bilinear (``l=1``) channel,
where ``T_{\mathrm{MF}}`` is not even defined. To compare against mean-field results at
reduced ``\tau``, convert with `temperature = τ * mfa_temperature_scale(mfa_sampler)`.

In the collection form the chain state **carries over** between consecutive temperatures
(with a fresh burn-in at each), so ordering high → low is an annealing run — useful for
reaching low-temperature order from a random start. Call once per temperature for
independent chains.

| Keyword | Meaning |
|---|---|
| `burnin` | equilibration sweeps before the first stored configuration (default 200) |
| `thin` | sweeps between stored configurations (default 10) |
| `step` | proposal rotation angle scale, radians (default 0.6) |
| `rng` | an `AbstractRNG` for reproducibility |
| `init` | chain start: a `3 × n_atoms` matrix ▸ the sampler's `reference` ▸ random |
| `randomize` | one uniform random global rotation per **stored** configuration |

One sweep is `n_atoms` single-spin attempts; each attempt contracts the fitted terms
against the current neighbor harmonics, so the acceptance uses the *exact* energy change
of the move (any body order, no linearization).

## The `MCSample` output

[`MCSample`](@ref) mirrors the [`MFASample`](@ref) interface (iterate / index as the
configurations) with MC-native labels:

| Field | Contents |
|---|---|
| `.configs` | `Vector` of `3 × n_atoms` configurations (unit columns) |
| `.temperature` | the ``k_B T`` of each configuration |
| `.energy` | that configuration's SCE energy (model units, `j0` excluded) |
| `.acceptance` | Metropolis accept fraction over the sweeps that produced it |

Use `.energy` to check equilibration (the trace should be trend-free within one
temperature) and `.acceptance` to tune `step`: aim for roughly 0.2–0.6, lowering `step`
at low temperature. The first configuration at each temperature includes its burn-in
window in the acceptance denominator.

## `randomize` and anisotropy training data

Local single-spin updates diffuse the configuration's *absolute* orientation very slowly
(for an isotropic model it is an exact zero mode), so a chain started along ``+z`` stays
near ``+z`` for a long time even when the physics says all orientations are equivalent.
`randomize = true` applies one Haar-uniform global rotation to each **stored copy** (the
chain itself is untouched):

- **isotropic model** — the rotated configurations are still exact Boltzmann samples
  (the energy is invariant; `.energy` is recomputed on the stored copy and machine-equal),
  with the absolute orientation now exactly uniform. This is the right way to generate
  training data for a *later* anisotropic fit: the spin structure gets presented to the
  crystal axes in all orientations.
- **anisotropic model** — the rotation changes the energy, so the result is data
  augmentation, not an equilibrium sample of that model. `.energy` still matches the
  stored (rotated) configuration.

## When to use which sampler

| | [`MFASampler`](@ref) | [`MetropolisSampler`](@ref) |
|---|---|---|
| distribution | single-site mean field (no correlations) | joint Boltzmann (correlated) |
| control | reduced ``\tau = T/T_{\mathrm{MF}}`` (or `m`) | absolute ``k_B T`` |
| cost | closed form / short per-site chains | lattice Markov chain |
| smooth per-atom distribution ``P(\boldsymbol e_a)`` | yes (coefficients) | histogram only |
| needs an ``l=1`` channel | yes (sets ``T_{\mathrm{MF}}``) | no |

Cheap sweeps, smooth per-atom distributions, and ``\tau``-controlled proposals → mean
field. Realistic correlated configurations at a physical temperature → Metropolis.
