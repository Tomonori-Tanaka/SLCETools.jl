# Orientation distributions & visualization

```@meta
CurrentModule = SCETools
```

Besides *drawing* configurations with [`sample`](@ref), SCETools can export the per-atom
**single-site orientation distribution** itself — the probability of each spin pointing in a
given direction — at one or more reduced temperatures ``\tau``, for an interactive 3D
viewer. The single-site law is

```math
P_a(\boldsymbol e) \propto \exp\!\big(-V_a(\boldsymbol e)\big), \qquad
V_a(\boldsymbol e) = \sum_{l\ge 1, m} c_a[lm]\, Z_{lm}(\boldsymbol e),
```

with the tesseral harmonics ``Z_{lm}`` and the self-consistent per-atom coefficients
``c_a`` (see [Theory](../theory/mfa.md)). The heavy physics runs in Julia; rendering is a
standalone Python viewer reading a JSON file, so the two are decoupled.

## Exporting to a file

The one verb is [`write_mfa_distributions`](@ref): give it a sampler, the
`crystal` it was built for, and the temperatures to sweep.

```julia
using SCETools
using SCETools.VASP: read_poscar

crystal = read_poscar("POSCAR")
sampler = MFASampler(model; reference)          # any sampler (isotropic … full multipole)

write_mfa_distributions("mfa_distributions.json", sampler, crystal;
                        taus = range(0.2, 0.95; length = 8), npoints = 2562)
```

Multiple ``\tau`` go into one file (the viewer gets a temperature slider). `npoints` sets
the shared sphere-grid resolution. The sampler's atom count must match the crystal.

### What the file stores — coefficients, not samples

The export stores the **exponent** coefficients ``c_a`` of ``V_a`` (the log-density), *not*
the density ``P`` itself. This matters: ``P = \exp(-V)`` has an infinite harmonic expansion
(it would need a lossy cutoff), whereas ``V`` is exactly finite — `lmax` is 1 for an
isotropic vMF cone, 2 with single-ion anisotropy, or the model's `lmax` for the full
multipole law. So the coefficients are exact and tiny (a handful of numbers per atom).

To turn coefficients into a coloured surface without re-implementing the harmonics in
Python, the file also carries a **shared basis matrix** ``Z[i,k] = Z_{lm}(\boldsymbol e_i)``
evaluated once by Julia on the render grid. The viewer recovers each atom's density with a
single matrix product ``V_a = Z\,c_a`` followed by ``\exp(-V_a)`` — the tesseral-harmonic
convention stays owned by [`SCEFitting.Harmonics`](https://github.com/Tomonori-Tanaka/SCEFitting.jl)
and is never duplicated. The full JSON schema (`scetools/mfa-distributions`, version 1) is
documented in [`viz/README.md`](https://github.com/Tomonori-Tanaka/SCETools.jl/blob/main/viz/README.md).

```@docs
write_mfa_distributions
```

## Visualizing

Install the viewer dependencies (`numpy`, `scipy`, `plotly`) and open the figure:

```bash
pip install -r viz/requirements.txt
python viz/mfa_viewer.py mfa_distributions.json      # writes + opens an .html
```

A self-contained HTML opens in the browser (drag to orbit, scroll to zoom — WebGL, no
native window). Each atom is a sphere **coloured by** ``P_a(\boldsymbol e)``; a green
shaft+head arrow shows the mean moment (length ``\propto`` the order parameter ``m_a``), the
gray box is the unit cell, and a VESTA-style x/y/z arrow triad sits just outside it. Two
sliders control the **temperature** ``\tau`` and the **arrow size**.

Useful flags: `--scale S` bulges the sphere out along the distribution's lobes (default
`0` = a true sphere coloured only by ``P``); `--head-frac F` sets the arrow head fraction
(default `0.5`); `--shared-clim`, `--no-triad`, `--grid-axes`, `--no-cell`, `--no-arrows`,
`--no-open`. See [`viz/README.md`](https://github.com/Tomonori-Tanaka/SCETools.jl/blob/main/viz/README.md).

The runnable demo [`examples/mfa_distributions.jl`](https://github.com/Tomonori-Tanaka/SCETools.jl/blob/main/examples/mfa_distributions.jl)
builds a 2-atom antiferromagnet and writes the sweep; its τ range and grid resolution are
optional command-line arguments:

```bash
julia --project=examples examples/mfa_distributions.jl [tau_min tau_max ntau [npoints]]
python viz/mfa_viewer.py examples/mfa_distributions.json
```

## Building blocks

[`write_mfa_distributions`](@ref) is the one call most users need; these are the pieces it
composes, exported for inspection or a custom exporter.

```@docs
mfa_site_coefficients
SiteDistributionField
fibonacci_sphere
SphereGrid
harmonic_basis
site_probabilities
```
