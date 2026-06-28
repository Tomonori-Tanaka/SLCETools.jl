# Exchange models

```@meta
CurrentModule = SCETools
```

[`ExchangeModel`](@ref) is the neutral carrier of the **bilinear** exchange (and single-ion
anisotropy) the mean-field sampler needs. [`MultipoleField`](@ref) is its higher-order
generalization, keeping **all** SCE channels. Both can be built by hand or extracted from a
fitted `SCEModel`.

## Building an `ExchangeModel` by hand

Three constructors, from least to most general:

```julia
# Isotropic (Heisenberg): Jiso[a,b] = Σ_R J_iso(a,b,R), symmetric.
ExchangeModel(Jiso::AbstractMatrix; onsite = nothing)

# Full tensorial: bilinear[a,b] = S_ab (3×3), the field is g_a = Σ_b S_ab ⟨e_b⟩.
# Its symmetric part is Heisenberg + Γ, its antisymmetric part the Dzyaloshinskii–Moriya vector.
ExchangeModel(bilinear::AbstractMatrix{<:SMatrix{3,3}}; onsite = nothing)
```

`onsite`, if given, is a length-`n` vector of `3 × 3` single-ion matrices ``A_a`` contributing
``\boldsymbol e' A_a \boldsymbol e`` (the `ls = [2]` channel). The energy convention is
``E = \sum_{\text{bonds}} \boldsymbol e_a' S_{ab}\,\boldsymbol e_b + \sum_a \boldsymbol e_a' A_a\,\boldsymbol e_a``,
with the directed bond matrices satisfying ``S_{ba} = S_{ab}'``.

```julia
using SCETools, StaticArrays, LinearAlgebra

# A two-sublattice antiferromagnet: J > 0 couples atoms 1 and 2.
J    = 0.01
Jiso = [0.0 J; J 0.0]
afm  = ExchangeModel(Jiso)

# Add a Dzyaloshinskii–Moriya vector D ∥ z on the 1–2 bond (antisymmetric part of S_12),
# and an easy-axis single-ion term on each atom.
D   = SVector(0.0, 0.0, 0.005)
Dx  = SMatrix{3,3}(0,-D[3],D[2], D[3],0,-D[1], -D[2],D[1],0)      # cross-product matrix
S12 = J * I + Dx
bil = [zero(SMatrix{3,3,Float64}) S12; SMatrix{3,3,Float64}(S12') zero(SMatrix{3,3,Float64})]
A   = SMatrix{3,3}(0.0,0,0, 0,0,0, 0,0,-0.01)                     # easy axis along z (e'Ae)
dmi = ExchangeModel(bil; onsite = [A, A])
```

A purely isotropic model (`isotropic = true`) takes the fast closed-form von Mises–Fisher
path; any DMI, anisotropic, or single-ion content switches the sampler to the Bingham /
Metropolis path automatically.

## Extracting from a fitted `SCEModel`

The bilinear and single-ion channels of a fitted model fold into an `ExchangeModel`:

```julia
exch = ExchangeModel(model)        # ls=[1,1] bilinear + ls=[2] single-ion
```

Only those two channels are representable as a bilinear model; any higher-order / higher-`l`
SALCs are dropped and reported via `@warn`. To keep **every** channel, use the full multipole
field:

```julia
mf = MultipoleField(model)         # all clusters and l, higher-order / many-body
s  = MFASampler(model; reference)  # ≡ MFASampler(MultipoleField(model); reference)
```

Both extractions read the fitted model through MagestyRebuild's public introspection surface
(`multipole_terms`, `bilinear_terms`) — never its SALC-basis internals — so they are insulated
from the core's basis representation.

## Reference, stationarity, and temperature scale

The longitudinal molecular-field matrix ``A[a,b] = -\hat{\boldsymbol e}_a' S_{ab}\,\hat{\boldsymbol e}_b``
folds the reference directions in, so any collinear ferro/antiferro/ferrimagnetic order becomes
ferromagnetic in the magnitudes ``m_a``. Its Perron eigenvalue ``\rho`` sets
``T_{\mathrm{MF}} = \rho/3``. The reference should be a stationary state of the exchange model
(the bilinear molecular field parallel to ``\hat{\boldsymbol e}_a``); a non-collinear ground
state forced by DMI / anisotropy is only approximately stationary under the rigid-axis mean
field, and the sampler warns when the reference is not stationary.

## Types

```@docs
ExchangeModel
MultipoleField
```
