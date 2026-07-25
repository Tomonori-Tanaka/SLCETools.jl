# Sampling

```@meta
CurrentModule = SLCETools
```

The mean-field sampler draws spin configurations from the single-site distribution of a
fitted (or hand-built) model at a reduced temperature ``\tau = T/T_{\mathrm{MF}}``. There is
one verb, [`sample`](@ref), and a ladder of [`MFASampler`](@ref) constructions.

## Constructing a sampler

```julia
MFASampler(reference)                              # single global magnetization
MFASampler(ExchangeModel(Jiso); reference)         # multi-sublattice isotropic
MFASampler(ExchangeModel(bilinear; onsite); reference)   # tensorial + single-ion
MFASampler(model::SLCEModel; reference)             # full multipole (all clusters / l)
```

`reference` is a `3 × n_atoms` matrix of **unit-column** reference directions
``\hat{\boldsymbol e}_a`` — the axes the mean field orders about (columns are normalized for
you). Every construction shares the same `sample` interface; they differ only in the
single-site law the spins are drawn from (see [Theory](../theory/mfa.md)):

- **isotropic** channels give a von Mises–Fisher cone — drawn in closed form;
- **anisotropic / single-ion / higher-multipole** channels give a Bingham or
  higher-multipole shape — drawn with a Metropolis engine.

See [Exchange models](exchange_models.md) for building the [`ExchangeModel`](@ref) /
[`MultipoleModel`](@ref) carriers.

## The `sample` verb

Two forms, returning an [`MFASample`](@ref):

```julia
# n configurations at one fixed point (give a temperature OR a target magnetization)
sample(sampler, n; tau = 0.6)
sample(sampler, n; m = 0.8)

# a sweep: nsamples configurations at each of several points
sample(sampler; tau = 0.1:0.1:0.9, nsamples = 50)
sample(sampler; m = [0.2, 0.5, 0.8], nsamples = 50)
```

Either fix the reduced temperature `tau` or the target magnetization `m` (the sampler
inverts ``m \leftrightarrow \tau`` for you). The `m` control is only meaningful for the
**single global** sampler, where one Langevin magnetization maps to a unique ``\tau``; a
coupled (exchange-model / multipole) sampler has distinct per-atom ``m_a(\tau)``, so
control it by `tau` and read the per-atom magnetizations off `samp.m` (or
[`mfa_sublattice_m`](@ref)). Both forms accept:

| Keyword | Meaning |
|---|---|
| `rng` | an `AbstractRNG` for reproducibility (default `default_rng()`) |
| `randomize` | rotate the whole configuration by a uniform random `SO(3)` element (Shoemake) |
| `fixed` | atom indices held exactly at their reference direction |
| `uniform` | atom indices drawn isotropically (infinite temperature) regardless of `tau` |

```julia
samp = sample(s, 200; tau = 0.5, rng = MersenneTwister(1), fixed = [1])
```

## The `MFASample` output

[`MFASample`](@ref) bundles the configurations with parallel per-sample labels:

| Field | Contents |
|---|---|
| `.configs` | `Vector` of `3 × n_atoms` configurations (unit columns) |
| `.tau` | the reduced temperature of each configuration |
| `.m` | the per-atom magnetization vector ``m_a`` used for each configuration |

It iterates and indexes as its configurations, so `for c in samp` and `samp[i]` yield the
`3 × n_atoms` matrices directly, while `samp.tau` / `samp.m` carry the labels (decision D1:
labelled output, ready to hand to a DFT writer).

## Temperature, magnetization, and the scale

For the single global isotropic law the magnetization ↔ reduced-temperature map is exposed
on its own:

```@docs
thermal_averaged_m
tau_from_magnetization
```

For a coupled sampler the per-atom magnetizations and the mean-field temperature scale are:

```@docs
mfa_sublattice_m
mfa_temperature_scale
```

Because everything is expressed in the reduced temperature ``\tau = T/T_{\mathrm{MF}}``, the
sampler is **scale-free**: multiplying every coupling by a constant leaves ``m_a(\tau)``
unchanged — only the coupling *ratios* (and the single-ion strength relative to the exchange)
are physical. The absolute ``T_{\mathrm{MF}}`` is reported by [`mfa_temperature_scale`](@ref)
but never needed to sample.
