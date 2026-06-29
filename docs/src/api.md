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

The [`ExchangeModel`](@ref) / [`MultipoleModel`](@ref) carriers are documented under
[Exchange models](guide/exchange_models.md); the τ ↔ m helpers
([`thermal_averaged_m`](@ref), [`tau_from_magnetization`](@ref), [`mfa_sublattice_m`](@ref),
[`mfa_temperature_scale`](@ref)) under [Sampling](guide/sampling.md).

## Orientation distributions

The per-atom orientation-distribution export (coefficients + a shared harmonic basis, for
the interactive viewer) is documented under
[Orientation distributions & visualization](guide/distributions.md):
[`write_mfa_distributions`](@ref), [`mfa_site_coefficients`](@ref),
[`SiteDistributionField`](@ref), [`fibonacci_sphere`](@ref), [`SphereGrid`](@ref),
[`harmonic_basis`](@ref), and [`site_probabilities`](@ref). The docstrings are shown there.

## VASP I/O

The namespaced `SCETools.VASP` submodule is the concrete VASP adapter: it **reads** DFT
training data ([`read_poscar`](@ref SCETools.VASP.read_poscar),
[`Oszicar`](@ref SCETools.VASP.Oszicar)) and **writes** constrained-noncollinear inputs from
sampled configurations ([`write_inputs`](@ref SCETools.VASP.write_inputs),
[`write_incar`](@ref SCETools.VASP.write_incar),
[`write_poscar`](@ref SCETools.VASP.write_poscar)). See [VASP I/O](guide/vasp.md); the
docstrings are shown there.
