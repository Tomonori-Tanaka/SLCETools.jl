"""
    SCETools

Auxiliary tooling around the spin-cluster-expansion (SCE) fitting core
[`SCEFitting`](https://github.com/Tomonori-Tanaka/SCEFitting.jl): utilities that
*consume* a fitted `SCEPredictor`
rather than build one. The first component is the **mean-field (MFA) spin-configuration
sampler** — draw physically representative finite-temperature spin configurations from the
single-site mean field of a fitted model (or a hand-built exchange model) at a controlled
reduced temperature `τ = T/T_MF`. Its joint-Boltzmann sibling is the **Metropolis
Monte-Carlo sampler** (`MetropolisSampler`) — correlated configurations at an absolute
temperature `k_B·T`. Future components (active learning, configuration / diagnostic
helpers) live alongside them.

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
# pure on-sphere math; the parent re-binds the names the rest of the package uses (the
# `_`-prefixed ones are engine internals, imported here by explicit qualification).
include("mfa/engine.jl")
using .MeanFieldEngine: _random_unit, _field_lmax, _l1_field, _rotate,
    _METROPOLIS_FLIP_FRACTION, site_potential, sample_vmf,
    sample_vmf_field, sample_site_metropolis, SphereQuadrature, sphere_quadrature,
    field_scale, multipole_average
# types (carriers + sampler) → ExchangeModel construction → the τ self-consistency solvers →
# the MFASampler constructors + the `sample` verb → extraction from a fitted SCE.
include("mfa/types.jl")
include("mfa/exchange.jl")
include("mfa/selfconsistency.jl")
include("mfa/sampler.jl")
include("mfa/bridge.jl")

# --- Metropolis Monte-Carlo sampling (docs/specs/mc-sampling.md) ---
# The joint-Boltzmann sibling of the mean-field sampler: single-spin Metropolis on the
# training cell at an absolute temperature, reusing the mean-field term digest and the
# engine's proposal primitives. Kept in its own directory as a future extraction seam.
include("mc/metropolis.jl")

# The VASP adapter (`SCETools.VASP`): OSZICAR/POSCAR reading into `SpinDatum`s and
# constrained-noncollinear INCAR / input-set writing. Namespaced as a submodule, so it
# does not grow the top-level export list (a second DFT code would be a sibling).
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

# the samplers, the carrier the user hand-builds, and the `sample` verb
export AbstractSampler, MFASampler, MFASample, ExchangeModel, sample
export MetropolisSampler, MCSample
# reduced-temperature ↔ magnetization helpers
export mfa_temperature_scale, mfa_sublattice_m, thermal_averaged_m, tau_from_magnetization
# per-atom MFA distribution export (the viz output a user calls)
export SiteDistributionField, mfa_site_coefficients, write_mfa_distributions

# --- Public, unexported -----------------------------------------------------------
# Reachable as `SCETools.<name>` (and documented), but kept out of the flat `using`
# namespace. The headline workflow (`MFASampler` / `sample` / `ExchangeModel` /
# `write_mfa_distributions`) already drives them; power users and the test suite reach
# them by qualification. Declared with the `public` keyword so the tier is
# machine-checkable (`Base.ispublic`, Aqua) instead of a comment-only promise.
public MultipoleModel                       # coupling digest (built via `MFASampler(model)`)
public KB_EV                                # Boltzmann constant, eV/K (kelvin ↔ kT)
public SphereGrid, fibonacci_sphere, harmonic_basis, site_probabilities  # viz plumbing
public MeanFieldEngine, VASP                # engine kernels / the VASP adapter submodule

end # module SCETools
