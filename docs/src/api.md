# API reference

```@meta
CurrentModule = SCETools
```

Every exported type and function. The headline workflow is `MFASampler(model; reference)`
→ [`sample`](@ref) → [`MFASample`](@ref); see [Sampling](guide/sampling.md) and
[Exchange models](guide/exchange_models.md) for the constructions and helpers (some
docstrings are shown on those pages and indexed below).

```@index
```

## Samplers

```@docs
AbstractSampler
MFASampler
sample
MFASample
```

## Carriers and helpers

The [`ExchangeModel`](@ref) / [`MultipoleField`](@ref) carriers are documented under
[Exchange models](guide/exchange_models.md); the τ ↔ m helpers
([`thermal_averaged_m`](@ref), [`tau_from_magnetization`](@ref), [`mfa_sublattice_m`](@ref),
[`mfa_temperature_scale`](@ref)) under [Sampling](guide/sampling.md).
