"""
    SCETools

Auxiliary tooling around the spin-cluster-expansion (SCE) fitting core
[`SCEFitting`](https://github.com/Tomonori-Tanaka/SCEFitting.jl): utilities that
*consume* a fitted `SCEPredictor`
rather than build one. The first component is the **mean-field (MFA) spin-configuration
sampler** — draw physically representative finite-temperature spin configurations from the
single-site mean field of a fitted model (or a hand-built exchange model) at a controlled
reduced temperature `τ = T/T_MF`. Future components (active learning, configuration /
diagnostic helpers) live alongside it.

The package reads the fitted Hamiltonian only through `SCEFitting`'s public
introspection surface (`multipole_terms`, `bilinear_terms`, and the tesseral-harmonic
submodule `SCEFitting.Harmonics`), never its SALC-basis internals.

See `docs/specs/mfa-sampling.md` for the sampler design.
"""
module SCETools

using LinearAlgebra: norm, I, eigen, Symmetric, dot, cross, mul!
using StaticArrays
using Statistics: mean
using Random: AbstractRNG, default_rng

# The SCE fitting core: the fitted-model type and the public, code-neutral introspection
# surface (so the sampler never reaches into the SALC-basis internals). `Harmonics` is the
# core's tesseral spherical-harmonic kernel, imported so the moved sampler files keep their
# `Harmonics.Zlm` / `Harmonics.lm_index` calls unchanged.
import SCEFitting.Harmonics
using SCEFitting: SCEPredictor, n_atoms, multipole_terms, MultipoleTerm, bilinear_terms,
    Crystal

# --- mean-field spin-configuration sampling (docs/specs/mfa-sampling.md) ---
# P0: the single-site engine submodule (potential, vMF / Metropolis draws, sphere quadrature),
# pure on-sphere math; the parent re-binds the names the rest of the package uses.
include("mfa/engine.jl")
using .MeanFieldEngine: _random_unit, _field_lmax, _site_potential, _l1_field, sample_vmf,
    sample_vmf_field, sample_site_metropolis, SphereQuadrature, sphere_quadrature,
    _field_scale, multipole_average
# types (carriers + sampler) → ExchangeModel construction → the τ self-consistency solvers →
# the MFASampler constructors + the `sample` verb → extraction from a fitted SCE.
include("mfa/types.jl")
include("mfa/exchange.jl")
include("mfa/selfconsistency.jl")
include("mfa/sampler.jl")
include("mfa/bridge.jl")

# VASP input writing: sampled spin configurations → constrained-noncollinear INCAR / input
# sets. Namespaced as `SCETools.VASP` (mirrors the reader-side `SCEFitting.VASP`), so it
# does not grow the top-level export list.
include("io/vasp.jl")

# Per-atom MFA probability distributions → coefficient export for the Python sphere viewer:
# the shared render grid + basis matrix (grid), the per-atom coefficients (distributions), and
# the self-describing JSON document (serialize).
include("viz/grid.jl")
include("viz/distributions.jl")
include("viz/serialize.jl")

# Mean-field spin-configuration sampling (docs/specs/mfa-sampling.md).
export AbstractSampler, MFASampler, MFASample, ExchangeModel, MultipoleModel, sample,
    mfa_temperature_scale, mfa_sublattice_m, thermal_averaged_m, tau_from_magnetization

# Deprecated alias: `MultipoleModel` was named `MultipoleField` before v0.2 (it is a coupling
# model, the full-fidelity sibling of `ExchangeModel`, not a field). Kept one minor version.
Base.@deprecate_binding MultipoleField MultipoleModel

# Per-atom MFA probability distribution export (viz/distributions.jl).
export SiteDistributionField, SphereGrid, fibonacci_sphere, mfa_site_coefficients,
    harmonic_basis, site_probabilities, write_mfa_distributions

end # module SCETools
