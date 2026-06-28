# VASP I/O

```@meta
CurrentModule = SCETools
```

`SCETools.VASP` is the concrete VASP adapter for the SCE workflow — the code-specific I/O the
fitting core ([SCEFitting](https://github.com/Tomonori-Tanaka/SCEFitting.jl)) keeps
out of itself (the core owns only the abstract `AbstractDFTSource` / `SpinDatum` /
`SCEDataset` seam). It goes both ways:

- **read** — `read_poscar` (POSCAR → `Crystal`) and `Oszicar` (constrained-noncollinear
  OSZICARs → `SpinDatum`s) produce *training data* for fitting;
- **write** — `write_poscar`, and `write_incar` / `write_inputs` (sampled spin directions →
  constrained-noncollinear INCAR / input sets) produce *DFT jobs* from sampled configurations.

The two directions share one frame / format convention, so a write → read round-trip is the
identity.

## Reading DFT training data

```@example vaspio
using SCEFitting, SCETools
using SCETools.VASP: read_poscar, Oszicar

dir = mktempdir()
write(joinpath(dir, "POSCAR"),
      "Fe2\n1.0\n 2.5 0 0\n 0 2.5 0\n 0 0 2.5\nFe\n2\nDirect\n 0 0 0\n 0.5 0.5 0.5\n")

crystal = read_poscar(joinpath(dir, "POSCAR"))            # → Crystal
(num_atoms(crystal), crystal.species_labels)
```

An [`Oszicar`](@ref SCETools.VASP.Oszicar) wraps one or more constrained-noncollinear OSZICAR
files as an `AbstractDFTSource`; `SCEFitting.read_configs` turns it into `SpinDatum`s
(energy, spin directions, moment magnitudes, constraining field, and the derived torque
target ``\boldsymbol\tau_a = \boldsymbol m_a \times \boldsymbol B_a``), and `SCEDataset` goes
straight from the source to a fit-ready dataset:

```julia
src     = Oszicar(["run1/OSZICAR", "run2/OSZICAR"]; saxis = [0, 0, 1])
dataset = SCEDataset(basis, src)                          # read_configs(src) under the hood
fit(SCEFit, dataset, OLS(); torque_weight = 0.5)
```

Moments and fields are rotated from the `SAXIS` quantization frame into Cartesian by
`Rz(α)·Ry(β)`; pass `energy_kind = :sigma0` for `E0` instead of the `F=` free energy, or
`mint = true` to read the `M_int` columns.

```@docs
SCETools.VASP.read_poscar
SCETools.VASP.Oszicar
```

## Writing constrained-noncollinear inputs

To turn sampled spin configurations into DFT jobs — the active-learning "label" step — write
the INCAR (and optionally a matching POSCAR). A sampler produces unit **directions**; the
moment **magnitudes** (μ_B) come from elsewhere — a template INCAR's existing `MAGMOM` (each
atom's norm), a per-species map, or an explicit per-atom vector.

[`write_inputs`](@ref SCETools.VASP.write_inputs) writes a `POSCAR` (via
[`write_poscar`](@ref SCETools.VASP.write_poscar)) and a matching `INCAR`, with the
`MAGMOM` / `M_CONSTR` atom order regrouped by species to match the POSCAR:

```@example vaspio
cfg = Float64[0 1; 0 0; 1 0]            # atom 1 along +z, atom 2 along +x
out = mktempdir()
SCETools.VASP.write_inputs(out, crystal, cfg; magmoms = Dict("Fe" => 2.2))
print(read(joinpath(out, "INCAR"), String))
```

For a whole sweep, pass the configurations of an `MFASample` (from [`sample`](@ref)) and get
one subdirectory per configuration:

```julia
samp = sample(MFASampler(model; reference = ref), 50; tau = 0.6)
SCETools.VASP.write_inputs("runs", crystal, samp.configs; magmoms = Dict("Fe" => 2.2))
# runs/config-001/{POSCAR,INCAR}, runs/config-002/…, …
```

### Moment magnitudes

`magmoms` accepts (in `write_inputs`): a scalar (uniform), a per-atom vector (crystal atom
order), a per-species `Dict("Fe" => 2.2, …)`, or `nothing` (default) to take each atom's
magnitude from the `base` template's `MAGMOM` norms. The written `MAGMOM` for an atom is
`magnitude · direction`; with `constrain = true` (default) the same vectors go to `M_CONSTR`
for a direction-constrained run.

### Merging into a template INCAR

Keep your tuned INCAR and replace only the magnetic tags — all other tags (`ENCUT`, `RWIGS`,
`I_CONSTRAINED_M`, `LAMBDA`, `SAXIS`, …) pass through verbatim:

```julia
SCETools.VASP.write_incar("INCAR", cfg; base = "INCAR.template")   # magnitudes from the template
```

Without a `base`, a minimal noncollinear INCAR is written (`LNONCOLLINEAR = .TRUE.`, `MAGMOM`,
and — under `constrain` — `I_CONSTRAINED_M` / `LAMBDA` / `M_CONSTR`); add `RWIGS` and the
electronic-structure tags through `base` or `extra`. For a non-default `SAXIS` the moments are
written in the SAXIS frame, the inverse of the reader's rotation, so the round-trip holds.

```@docs
SCETools.VASP.write_inputs
SCETools.VASP.write_incar
SCETools.VASP.write_poscar
```
