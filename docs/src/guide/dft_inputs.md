# VASP inputs

```@meta
CurrentModule = SCETools
```

To turn sampled spin configurations into DFT jobs — the "label" step of an active-learning
loop, or just to enrich an SCE training set — `SCETools.VASP` writes **constrained
noncollinear** VASP inputs from the spin directions. It mirrors the reader-side
`MagestyRebuild.VASP` (POSCAR / OSZICAR), with consistent conventions.

A sampler produces unit spin **directions**; the moment **magnitudes** (μ_B) come from
elsewhere — a template INCAR's existing `MAGMOM` (each atom's norm), a per-species map, or an
explicit per-atom vector.

## Writing a full input set

[`write_inputs`](@ref SCETools.VASP.write_inputs) writes a `POSCAR` (via
`MagestyRebuild.VASP.write_poscar`) and a matching `INCAR`, with the `MAGMOM` / `M_CONSTR` atom
order regrouped by species to match the POSCAR — so the two files are always consistent.

```@example vasp
using MagestyRebuild, SCETools

lat = Lattice([3.0 0 0; 0 3.0 0; 0 0 3.0])
fe  = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
cfg = Float64[0 1; 0 0; 1 0]            # atom 1 along +z, atom 2 along +x

dir = mktempdir()
SCETools.VASP.write_inputs(dir, fe, cfg; magmoms = Dict("Fe" => 2.2))
print(read(joinpath(dir, "INCAR"), String))
```

For a whole sweep, pass the configurations of an `MFASample` (from [`sample`](@ref)) and get one
subdirectory per configuration:

```julia
samp = sample(MFASampler(model; reference = ref), 50; tau = 0.6)
SCETools.VASP.write_inputs("runs", crystal, samp.configs; magmoms = Dict("Fe" => 2.2))
# runs/config-001/{POSCAR,INCAR}, runs/config-002/…, …
```

## Moment magnitudes

`magmoms` accepts (in `write_inputs`):

| `magmoms` | meaning |
|---|---|
| `2.2` | a uniform magnitude on every atom |
| `[2.2, 1.7, …]` | per-atom magnitudes (in the crystal's atom order) |
| `Dict("Fe" => 2.2, "O" => 0.0)` | per-species magnitudes |
| `nothing` (default) | take each atom's magnitude from the `base` template's `MAGMOM` |

The written `MAGMOM` vector for an atom is `magnitude · direction`; with `constrain = true`
(the default) the same vectors are written to `M_CONSTR` for a direction-constrained run.

## Merging into a template INCAR

Keep your tuned INCAR and let the writer replace only the magnetic tags. All other tags
(`ENCUT`, `RWIGS`, `I_CONSTRAINED_M`, `LAMBDA`, `SAXIS`, …) pass through verbatim; the moment
magnitudes default to the template's existing `MAGMOM` norms:

```julia
SCETools.VASP.write_incar("INCAR", cfg; base = "INCAR.template")   # magnitudes from the template
```

Pass `base` as a path or as raw INCAR text. Without a `base`, a minimal noncollinear INCAR is
written (`LNONCOLLINEAR = .TRUE.`, `MAGMOM`, and — under `constrain` — `I_CONSTRAINED_M` /
`LAMBDA` / `M_CONSTR`); add `RWIGS` and the electronic-structure tags through `base` or `extra`
for a runnable constrained calculation.

## Frame (SAXIS)

By default the moments are written in the global Cartesian frame (`SAXIS = [0,0,1]`). For a
non-default `saxis`, the moments are rotated *out* of Cartesian by the inverse of the reader's
`Rz(α)Ry(β)` rotation, so a write → read round-trip through `MagestyRebuild.VASP.Oszicar` is the
identity.

## Atom order

`MAGMOM` / `M_CONSTR` must follow the POSCAR atom order. `write_inputs` handles this for you
(it permutes the moments to the species-grouped order `write_poscar` uses). If you call the
low-level [`write_incar`](@ref SCETools.VASP.write_incar) against a POSCAR you wrote yourself,
make sure the configuration columns are in that POSCAR's atom order.

## Functions

```@docs
SCETools.VASP.write_inputs
SCETools.VASP.write_incar
```
