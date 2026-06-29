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

# --- Public API (exported) --------------------------------------------------------
# The mean-field sampling workflow a user reaches for. The coupling-model digest, the engine
# primitives, and the viz render plumbing are *public but unexported* — see the block below.

# the sampler, the carrier the user hand-builds, and the `sample` verb
export AbstractSampler, MFASampler, MFASample, ExchangeModel, sample
# reduced-temperature ↔ magnetization helpers
export mfa_temperature_scale, mfa_sublattice_m, thermal_averaged_m, tau_from_magnetization
# per-atom MFA distribution export (the viz output a user calls)
export SiteDistributionField, mfa_site_coefficients, write_mfa_distributions

# Deprecated alias: `MultipoleModel` was named `MultipoleField` before v0.2 (it is a coupling
# model, the full-fidelity sibling of `ExchangeModel`, not a field). Kept one minor version.
Base.@deprecate_binding MultipoleField MultipoleModel

# --- Public, unexported -----------------------------------------------------------
# Reachable as `SCETools.<name>` (and documented), but kept out of the flat `using` namespace.
# The headline workflow (`MFASampler` / `sample` / `ExchangeModel` / `write_mfa_distributions`)
# already drives them. Power users and the test suite reach them by qualification.
#
#   coupling digest : MultipoleModel        (built via `MFASampler(model)`; rarely hand-made)
#   viz plumbing    : SphereGrid, fibonacci_sphere, harmonic_basis, site_probabilities
#   engine kernels  : SCETools.MeanFieldEngine.{sphere_quadrature, multipole_average,
#                     sample_vmf, sample_vmf_field, sample_site_metropolis, SphereQuadrature}

end # module SCETools
