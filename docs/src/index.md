# SCETools.jl

```@meta
CurrentModule = SCETools
```

Auxiliary tooling around the spin-cluster-expansion (SCE) fitting core
[MagestyRebuild.jl](https://github.com/Tomonori-Tanaka/Magesty_rebuild.jl) — utilities that
**consume** a fitted `SCEModel` rather than build one.

The first component is the **mean-field (MFA) spin-configuration sampler**: draw physically
representative finite-temperature spin configurations from the single-site mean field of a
fitted model (or a hand-built exchange model) at a controlled reduced temperature
``\tau = T/T_{\mathrm{MF}}``. Such configurations are exactly what you feed back into DFT to
enrich an SCE training set, which is why sampling and (planned) active learning live here
together.

!!! note "Status — companion to an architectural exploration (v0)"
    SCETools depends on MagestyRebuild and reads a fitted model **only** through its public
    introspection surface — [`multipole_terms`](https://github.com/Tomonori-Tanaka/Magesty_rebuild.jl),
    `bilinear_terms`, `num_atoms`, and the tesseral submodule `MagestyRebuild.Harmonics` —
    never its SALC-basis internals. The sampler was developed inside MagestyRebuild (phases
    P0–P4) and extracted here when the core was narrowed to fitting only.

## The sampler at a glance

One verb, [`sample`](@ref); a ladder of [`MFASampler`](@ref) constructions of increasing
fidelity:

| Construction | Physics captured | Single-site law |
|---|---|---|
| `MFASampler(reference)` | a single global magnetization | Langevin / von Mises–Fisher |
| `MFASampler(ExchangeModel(Jiso); reference)` | multi-sublattice isotropic (Heisenberg) exchange | per-atom von Mises–Fisher |
| `MFASampler(ExchangeModel(bilinear; onsite); reference)` | tensorial exchange (DMI + anisotropic) + single-ion | von Mises–Fisher / Bingham (Metropolis) |
| `MFASampler(model::SCEModel; reference)` | **all** SCE clusters and `l` — higher-order / many-body | full multipole (Metropolis) |

The [`ExchangeModel`](@ref) carrier can be built by hand (raw couplings, or a TB2J-style
``J_{ij}`` tensor) or extracted from a fitted `SCEModel`; the full
[`MultipoleField`](@ref) path keeps every channel.

## Documentation

| Page | What's there |
|------|--------------|
| [Getting started](getting_started.md) | Install, then sample a Heisenberg dimer from a fitted model |
| [Guide: sampling](guide/sampling.md) | The `sample` verb, the fidelity ladder, `MFASample`, the τ ↔ m helpers |
| [Guide: exchange models](guide/exchange_models.md) | Building an `ExchangeModel` by hand or from a fitted `SCEModel` |
| [Guide: VASP inputs](guide/dft_inputs.md) | Write constrained-noncollinear INCAR / input sets from sampled configurations |
| [Theory: the mean-field sampler](theory/mfa.md) | The MFA decoupling, reduced temperature, vMF / Bingham single-site laws |
| [API reference](api.md) | Every exported type and function |

## Relationship to the ecosystem

This package re-founds the sampling / active-learning layer on the clean MagestyRebuild
rebuild. The older `SpinClusterMC.jl` (Monte Carlo) and `ActiveSCE.jl` (active learning)
packages remain in use against the original `Magesty.jl` and are not targeted here.

## Planned

`src/active_learning/` is laid out for an efficient model-construction loop — propose
configurations with the sampler, label them with DFT, refit the SCE model, iterate. See the
package `SPEC.md`.
