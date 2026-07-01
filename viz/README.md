# MFA orientation-distribution viewer

Interactive 3D viewer for the per-atom mean-field orientation distributions exported by
SCETools.jl. The heavy physics runs in Julia; this is a standalone Python/Plotly renderer
(it is **not** a dependency of the Julia package). It renders with WebGL in the browser, so
it is portable and avoids the native-VTK `Segmentation fault: 11` some setups hit on rotate.

## Pipeline

1. **Julia** — write the distributions over a temperature sweep to a JSON file:

   ```julia
   using SCETools
   write_mfa_distributions("mfa_distributions.json", sampler, crystal;
                           taus = range(0.2, 0.95; length = 8), npoints = 2562)
   ```

   See [`examples/mfa_distributions.jl`](../examples/mfa_distributions.jl) for a runnable
   end-to-end demo.

2. **Python** — install the viewer deps and open the figure:

   ```bash
   pip install -r viz/requirements.txt
   python viz/mfa_viewer.py mfa_distributions.json      # writes + opens an .html
   ```

   A self-contained HTML opens in the browser; drag to orbit, scroll to zoom. Each atom is
   drawn as a sphere **coloured by** the orientation probability `P(e) ∝ exp(−V_a(e))` (a
   heatmap on a true sphere by default); the green arrow is the mean moment (shaft + head,
   length ∝ `|m_a|`, updated per τ frame), the gray box is the unit cell, and a VESTA-style
   x/y/z arrow triad sits just outside it. The arrow is a **magnitude-only** indicator drawn
   along the fixed reference axis: the JSON carries signed `m`, but an anti-aligned
   sublattice (`m_a < 0`, possible off a non-stationary reference) renders the same as an
   aligned one — read the sign from the sphere colouring, not the arrow.

   Two in-figure sliders: the **τ slider** (temperature sweep) and the **arrow-size slider**
   (0×–5× the moment arrows; 0× hides them).

   Options: `-o out.html`, `--scale S` (bulge the sphere out along the lobes — handy for the
   anisotropy, but the shape then distorts at low τ; default 0 = true sphere), `--arrow-scale
   A`, `--head-frac F` (cone-head fraction, default 0.5; lower = smaller head), `--cmap`,
   `--shared-clim` (one colour scale across all τ), `--no-triad`, `--grid-axes` (boxed tick
   axes instead of the triad), `--no-cell`, `--no-arrows`, `--no-open`.

## File format (`scetools/mfa-distributions`, version 1)

A self-describing JSON document. **Conventions** (the viewer relies on these and does *not*
re-derive them):

- `lattice_vectors` — three rows, each a **lattice vector** `a, b, c` (Å).
- `positions_cartesian` — per-atom `[x, y, z]` already in Cartesian Å (do **not** multiply
  by the lattice again).
- `reference` — per-atom unit axis `ê_a` as a row `[x, y, z]`.
- `grid.directions` — shared unit directions `e_i` on S² (one row each); the render mesh
  vertices.
- `grid.Z` — shared basis matrix `Z[i][k] = Z_lm(e_i)`, columns indexed by the package's
  `lm_index(l, m)` ordering, with the `l = 0` column zeroed. The viewer recovers the
  exponent as `V_a = Z @ coeffs_a` — a plain matrix product, so the tesseral-harmonic
  convention stays owned by Julia (`SCEFitting.Harmonics`) and is never re-implemented here.
- `grid.weight` — the equal-solid-angle quadrature weight `4π / npoints` (no `sinθ`
  factor). Densities are normalized so `Σ_i P_i · weight = 1`.
- `frames[t].coeffs` — per-atom coefficient vectors `c_a` (length `(lmax+1)²`) at
  temperature `temperatures[t]`; `frames[t].m` — per-atom order parameter `m_a ∈ [−1, 1]`.

Because only the small coefficient vectors vary per atom and per τ (the bulky grid and
basis are stored once), the file stays compact even for many atoms and a long τ sweep.
