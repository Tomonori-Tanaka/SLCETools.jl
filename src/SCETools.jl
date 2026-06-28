"""
    SCETools

Auxiliary tooling around the spin-cluster-expansion (SCE) fitting core
[`SCEFitting`](https://github.com/Tomonori-Tanaka/SCEFitting.jl): utilities that
*consume* a fitted `SCEModel`
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

using LinearAlgebra: norm, det, I, eigen, Symmetric, Diagonal, dot, cross, mul!
using StaticArrays
using Statistics: mean
using Random: AbstractRNG, default_rng

# The SCE fitting core: the fitted-model type and the public, code-neutral introspection
# surface (so the sampler never reaches into the SALC-basis internals). `Harmonics` is the
# core's tesseral spherical-harmonic kernel, imported so the moved sampler files keep their
# `Harmonics.Zlm` / `Harmonics.lm_index` calls unchanged.
import SCEFitting.Harmonics
using SCEFitting: SCEModel, num_atoms, multipole_terms, MultipoleTerm, bilinear_terms

# --- mean-field spin-configuration sampling (docs/specs/mfa-sampling.md) ---
# P0: the single-site engine (potential, vMF / Metropolis draws, sphere quadrature).
# P2/P3: the ExchangeModel carrier + coupled self-consistency (before MFASampler, which
# references the ExchangeModel type). P1/P2/P3: the `MFASampler` + the `sample` verb.
include("sampling/site_engine.jl")
include("sampling/exchange.jl")
include("sampling/mfa_sampler.jl")
# Extract an ExchangeModel / MultipoleField from a fitted SCE via the core's public
# `multipole_terms` / `bilinear_terms` introspection.
include("sampling/sce_bridge.jl")

# VASP input writing: sampled spin configurations → constrained-noncollinear INCAR / input
# sets. Namespaced as `SCETools.VASP` (mirrors the reader-side `SCEFitting.VASP`), so it
# does not grow the top-level export list.
include("io/vasp.jl")

# Mean-field spin-configuration sampling (docs/specs/mfa-sampling.md).
export AbstractSampler, MFASampler, MFASample, ExchangeModel, MultipoleField, sample,
    mfa_temperature_scale, mfa_sublattice_m, thermal_averaged_m, tau_from_magnetization

end # module SCETools
