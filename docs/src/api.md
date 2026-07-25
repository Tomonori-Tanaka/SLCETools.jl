# API reference

```@meta
CurrentModule = SLCETools
```

```@docs
SLCETools
```

Every public type and function. The headline workflows are `MFASampler(model; reference)`
→ [`sample`](@ref) → [`MFASample`](@ref) (mean field) and `MetropolisSampler(model)` →
[`sample`](@ref) → [`MCSample`](@ref) (Metropolis MC); see [Sampling](guide/sampling.md),
[Monte-Carlo sampling](guide/mc_sampling.md), and
[Exchange models](guide/exchange_models.md) for the constructions and helpers (some
docstrings are shown on those pages and indexed below).

The surface is two-tiered: the **exported** workflow above, plus a **public,
unexported** tier declared with the Julia `public` keyword and reached by qualification
(`SLCETools.<name>`): [`MultipoleModel`](@ref), the viz plumbing
([`SphereGrid`](@ref), [`fibonacci_sphere`](@ref), [`harmonic_basis`](@ref),
[`site_probabilities`](@ref)), and the submodules `SLCETools.MeanFieldEngine` (the engine
primitives `site_potential`, `sample_vmf`, `sample_vmf_field`, `sample_site_metropolis`,
`SphereQuadrature`, `sphere_quadrature`, `field_scale`, `multipole_average`) and
`SLCETools.VASP`.

```@index
```

## Samplers

```@docs
AbstractSampler
MFASampler
sample
MFASample
MetropolisSampler
MCSample
KB_EV
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

## MeanFieldEngine kernels

The `MeanFieldEngine` submodule (public, unexported — call as
`SLCETools.MeanFieldEngine.site_potential` etc.) is the self-contained single-site
engine: the potential `V(e) = Σ c·Z`, the closed-form vMF draw, the general Metropolis
draw (rotation + antipodal-flip mixture proposal), and the deterministic sphere
quadrature for the multipole averages. Pure on-sphere math with no SCE coupling.

```@docs
MeanFieldEngine.site_potential
MeanFieldEngine.sample_vmf
MeanFieldEngine.sample_vmf_field
MeanFieldEngine.sample_site_metropolis
MeanFieldEngine.SphereQuadrature
MeanFieldEngine.sphere_quadrature
MeanFieldEngine.multipole_average
```

## VASP I/O

The namespaced `SLCETools.VASP` submodule is the concrete VASP adapter: it **reads** DFT
training data ([`read_poscar`](@ref SLCETools.VASP.read_poscar),
[`Oszicar`](@ref SLCETools.VASP.Oszicar)) and **writes** constrained-noncollinear inputs from
sampled configurations ([`write_inputs`](@ref SLCETools.VASP.write_inputs),
[`write_incar`](@ref SLCETools.VASP.write_incar),
[`write_poscar`](@ref SLCETools.VASP.write_poscar)). See [VASP I/O](guide/vasp.md); the
docstrings are shown there.
