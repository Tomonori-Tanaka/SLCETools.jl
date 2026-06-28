# Theory: the mean-field sampler

```@meta
CurrentModule = SCETools
```

The sampler generates finite-temperature spin configurations from the **single-site
mean-field** approximation (MFA) of an SCE energy surface. This page sketches why any SCE
model collapses to a single-site tesseral potential, and what distribution the spins are then
drawn from.

## The SCE energy and the mean-field decoupling

An SCE energy is a polynomial in unit spin directions,

```math
E\bigl(\{\hat{\boldsymbol e}_a\}\bigr) = j_0 + \sum_\varphi J_\varphi\,\Phi_\varphi,
\qquad
\Phi_\varphi = (4\pi)^{N/2}\sum_\mu \text{folded}_\mu \prod_{i} Z_{l_i\mu_i}(\hat{\boldsymbol e}_{a_i}),
```

with real tesseral harmonics ``Z_{l m}`` over clusters of ``N`` spins. The mean-field
approximation replaces every spin other than the one of interest by its thermal average and
uses the factorization of independent sites,

```math
\bigl\langle \textstyle\prod_i Z_{l_i\mu_i}\bigr\rangle \;\longrightarrow\; \prod_i \langle Z_{l_i\mu_i}\rangle .
```

Collecting, for a chosen site ``a``, every term's contraction against the *other* sites'
multipole averages leaves a **single-site tesseral potential**

```math
H_a(\hat{\boldsymbol e}) = \sum_{l m} h_a^{l m}\, Z_{l m}(\hat{\boldsymbol e}),
\qquad
h_a^{l m} = \sum_{\varphi \ni a} J_\varphi (4\pi)^{N/2}\,\text{folded}\;\prod_{b\neq a}\langle Z_{l_b\mu_b}(\hat{\boldsymbol e}_b)\rangle .
```

This is the same leave-one-out contraction the SCE gradient kernel performs, with the site-``a``
harmonic left symbolic instead of differentiated. The order parameters are the per-atom
multipole averages ``\langle Z_{l m}\rangle_a``, solved to self-consistency.

## Reduced temperature

The single-site Boltzmann weight is ``P_a(\hat{\boldsymbol e}) \propto \exp[-\beta H_a]``. The
``l = 1`` (bilinear) part defines a molecular-field matrix whose Perron eigenvalue ``\rho`` sets
the mean-field ordering temperature ``T_{\mathrm{MF}} = \rho/3``; writing
``\tau = T/T_{\mathrm{MF}}`` gives ``\beta = 3/(\rho\tau)``. Everything is then expressed in
``\tau``, so the sampler is **scale-free** — scaling all couplings scales ``\rho`` and the field
together and leaves ``m_a(\tau)`` unchanged. Only coupling *ratios* (and the single-ion strength
relative to the exchange) are physical, and the absolute ``T_{\mathrm{MF}}`` is never needed to
sample.

## Single-site distributions

The shape of ``P_a`` depends on which harmonic orders ``H_a`` contains:

- **``l = 1`` only (isotropic exchange).** ``H_a = \hat{\boldsymbol e}\cdot\boldsymbol g_a`` is a
  linear form — a **von Mises–Fisher** cone about the molecular-field direction, with
  concentration ``\kappa_a``. For the single global / multi-sublattice isotropic samplers this
  reduces to the Langevin self-consistency ``m = \mathcal L(3m/\tau)`` with
  ``\mathcal L(\kappa) = \coth\kappa - 1/\kappa``, drawn in closed form (Ulrich/Wood inverse-CDF).
- **``l = 2`` (DMI / anisotropic exchange / single-ion).** ``H_a`` gains a quadratic form
  ``\boldsymbol e' A_a \boldsymbol e`` — a **Bingham** factor the von Mises–Fisher law cannot
  represent. The magnetizations are solved as ``m_a = \langle \hat{\boldsymbol e}\cdot\hat{\boldsymbol e}_a\rangle``
  by sphere quadrature, and configurations are drawn with a Metropolis engine.
- **``l \geq 3`` (higher-order / many-body).** ``H_a`` is a general tesseral potential; the full
  multipole averages ``\langle Z_{l m}\rangle_a`` (``l \leq l_{\max}``) are iterated to
  self-consistency, and the spins are again drawn by Metropolis.

## Self-consistency

The coupled magnetizations (or, for the full multipole path, the flattened
``\langle Z_{l m}\rangle_a``) solve a fixed-point map ``m = G(m)``. Near ``T_{\mathrm{MF}}`` the
contraction rate approaches one (critical slowing), so a damped Picard iteration stalls; the
sampler uses **depth-1 Anderson acceleration** instead, which selects the stable ordered branch
over the trivial disordered solution and converges through the critical region.

## Conventions and references

The reduced-temperature formulation, the rigid reference axes ``\hat{\boldsymbol e}_a``, the SCE
source at full ``l_{\max}``, and the labelled output are design decisions D1–D5 recorded in the
package `docs/specs/mfa-sampling.md`. The physical conventions (unit spins, real tesseral
harmonics with the per-site ``(4\pi)^{-1/2}`` and per-``N`` ``(4\pi)^{N/2}`` factors) are
inherited verbatim from
[SCEFitting.jl](https://github.com/Tomonori-Tanaka/SCEFitting.jl); the sampler reads
them through that package's public introspection surface and never re-derives them.
