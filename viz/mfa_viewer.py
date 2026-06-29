#!/usr/bin/env python3
"""Viewer for per-atom MFA orientation distributions (Plotly / WebGL).

Writes a self-contained HTML file that rotates in any browser (drag to orbit, scroll to
zoom) with a temperature slider over the τ frames and a live arrow-size slider. It renders
with Plotly's WebGL backend (no native VTK window), so it is portable and avoids the macOS
``Segmentation fault: 11`` that native VTK interactors hit on some setups.

Reads a ``scetools/mfa-distributions`` JSON (written by ``write_mfa_distributions`` in
SCETools.jl) and uses the stored basis matrix ``Z`` to recover each atom's density as
``exp(-Z @ c_a)`` (a matrix product; the tesseral-harmonic convention stays owned by Julia).

Usage:
    python viz/mfa_viewer.py PATH [-o out.html] [options]

Options:
    -o, --out FILE     output HTML path (default: alongside PATH, .html)
    --radius R         base sphere radius in Å (default 0.6)
    --scale S          lobe-deformation strength (default 0 = a true sphere coloured by P;
                       >0 bulges the sphere out along the distribution's lobes)
    --arrow-scale A    initial moment-arrow size (also live via the in-figure slider)
    --head-frac F      fraction of the moment arrow taken by the cone head (default 0.5)
    --cmap NAME        Plotly colorscale (default "Inferno")
    --shared-clim      one colour scale across all τ frames (default: per-frame)
    --no-triad         hide the VESTA-style x/y/z axis arrows
    --grid-axes        restore the boxed Plotly tick axes instead of the arrow triad
    --no-cell          do not draw the unit-cell box
    --no-arrows        do not draw the mean-moment arrows
    --no-open          do not open the HTML in a browser after writing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

import numpy as np

try:
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull
except ImportError as exc:  # pragma: no cover - dependency hint
    sys.exit(
        f"missing dependency: {exc}. Install with:\n"
        "    pip install -r viz/requirements.txt"
    )

SCHEMA = "scetools/mfa-distributions"


def load(path):
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("schema") != SCHEMA:
        sys.exit(f"{path}: not a {SCHEMA} file (schema={doc.get('schema')!r})")
    return doc


def probabilities(Z, coeffs_a, weight):
    """Per-direction normalized density for one atom: P = exp(-V)/∫P dΩ, V = Z·c_a."""
    V = Z @ coeffs_a
    p = np.exp(-(V - V.min()))
    p /= p.sum() * weight
    return p


def atom_mesh(center, directions, faces, prob, r0, scale, cmin, cmax, cmap, showscale):
    pmax = prob.max()
    rel = prob / pmax if pmax > 0 else np.zeros_like(prob)
    verts = center[None, :] + (r0 * (1.0 + scale * rel))[:, None] * directions
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=prob, colorscale=cmap, cmin=cmin, cmax=cmax,
        showscale=showscale, colorbar=dict(title="P(e)") if showscale else None,
        # smooth (vertex-normal) shading with no specular highlight, and mostly ambient
        # light: the surface reads as a clean colour heatmap instead of lit polygon facets.
        flatshading=False,
        lighting=dict(ambient=0.9, diffuse=0.2, specular=0.0, roughness=1.0, fresnel=0.0),
        lightposition=dict(x=0, y=0, z=100000),
        name="dist", hoverinfo="skip",
    )


def cell_trace(lattice_vectors):
    a, b, c = (np.asarray(v, float) for v in lattice_vectors)
    corners = {(i, j, k): i * a + j * b + k * c
               for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    xs, ys, zs = [], [], []
    for (i, j, k), p0 in corners.items():
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            n = (i + di, j + dj, k + dk)
            if n in corners:
                p1 = corners[n]
                xs += [p0[0], p1[0], None]
                ys += [p0[1], p1[1], None]
                zs += [p0[2], p1[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color="gray", width=2), hoverinfo="skip",
                        name="cell", showlegend=False)


def axis_triad(origin, length):
    """A VESTA-style x/y/z arrow triad (red/green/blue) anchored at `origin`, no ticks."""
    traces = []
    axes = [(np.array([1.0, 0, 0]), "x", "#d62728"),
            (np.array([0, 1.0, 0]), "y", "#2ca02c"),
            (np.array([0, 0, 1.0]), "z", "#1f77b4")]
    for d, name, color in axes:
        tip = origin + length * d
        traces.append(go.Scatter3d(
            x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
            mode="lines", line=dict(color=color, width=6),
            hoverinfo="skip", showlegend=False))
        traces.append(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[d[0]], v=[d[1]], w=[d[2]],
            anchor="tail", sizemode="absolute", sizeref=0.3 * length,
            colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip"))
        lab = origin + length * 1.2 * d
        traces.append(go.Scatter3d(
            x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text", text=[name],
            textfont=dict(color=color, size=16), hoverinfo="skip", showlegend=False))
    return traces


def moment_arrow_traces(positions, reference, m, full_len, head_frac, color="seagreen"):
    """Mean-moment arrows as a thin shaft + a cone head whose length is `head_frac` of the
    arrow. The arrow length ∝ |m| (a more ordered atom → longer arrow); `full_len` is the
    overall scale. Returns [shaft_line, head_cone]."""
    L = np.maximum(np.abs(m), 1e-3) * full_len            # per-atom full arrow length
    d = reference
    shaft_end = positions + (1.0 - head_frac) * L[:, None] * d
    xs, ys, zs = [], [], []
    for a in range(len(positions)):
        xs += [positions[a, 0], shaft_end[a, 0], None]
        ys += [positions[a, 1], shaft_end[a, 1], None]
        zs += [positions[a, 2], shaft_end[a, 2], None]
    shaft = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                         line=dict(color=color, width=5), hoverinfo="skip",
                         showlegend=False, name="moment")
    head = go.Cone(x=shaft_end[:, 0], y=shaft_end[:, 1], z=shaft_end[:, 2],
                   u=d[:, 0] * L, v=d[:, 1] * L, w=d[:, 2] * L,
                   anchor="tail", sizemode="scaled", sizeref=head_frac,
                   colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip")
    return [shaft, head]


def frame_mesh_traces(doc, Z, faces, directions, k, args, clim):
    coeffs = np.asarray(doc["frames"][k]["coeffs"], float)
    positions = np.asarray(doc["positions_cartesian"], float)
    weight = float(doc["grid"]["weight"])
    natoms = positions.shape[0]
    cmin, cmax = clim
    return [atom_mesh(positions[a], directions, faces,
                      probabilities(Z, coeffs[a], weight),
                      args.radius, args.scale, cmin, cmax, args.cmap, showscale=(a == 0))
            for a in range(natoms)]


def build_figure(doc, args):
    directions = np.asarray(doc["grid"]["directions"], float)
    Z = np.asarray(doc["grid"]["Z"], float)
    positions = np.asarray(doc["positions_cartesian"], float)
    reference = np.asarray(doc["reference"], float)
    weight = float(doc["grid"]["weight"])
    frames = doc["frames"]
    temps = doc["temperatures"]
    natoms = positions.shape[0]

    faces = ConvexHull(directions).simplices

    # per-frame probability arrays (also fixes the colour range)
    all_p = [np.array([probabilities(Z, np.asarray(fr["coeffs"], float)[a], weight)
                       for a in range(natoms)]) for fr in frames]
    gmax = float(max(p.max() for p in all_p))
    clim_of = (lambda k: (0.0, gmax)) if args.shared_clim \
        else (lambda k: (0.0, float(all_p[k].max())))

    # static overlays (drawn once, kept across frames)
    static = []
    if not args.no_cell:
        static.append(cell_trace(doc["lattice_vectors"]))
    static.append(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="markers",
        marker=dict(size=4, color="black"), hoverinfo="text",
        text=[f"atom {a + 1}" for a in range(natoms)], name="atoms", showlegend=False))
    # Mean-moment arrows (shaft + head). The arrow-size slider toggles which precomputed
    # size is visible — robust (only the `visible` attribute, which every trace supports).
    arrow_mults = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    default_mu = 1.0
    base_len = 1.6 * args.radius * args.arrow_scale
    arrow_groups = []                       # (mu, [trace indices]) for the visible ones
    if not args.no_arrows:
        m0 = np.asarray(frames[0]["m"], float)
        for mu in arrow_mults:
            if mu == 0.0:
                continue                    # the 0× step just hides every arrow trace
            idx0 = natoms + len(static)
            traces = moment_arrow_traces(positions, reference, m0,
                                         base_len * mu, args.head_frac)
            for t in traces:
                t.visible = (mu == default_mu)
            static += traces
            arrow_groups.append((mu, list(range(idx0, idx0 + len(traces)))))

    # VESTA-style x/y/z arrow triad just outside the cell (appended after the moment cone
    # so `cone_idx` stays valid). The boxed tick axes are hidden in the layout below.
    if not args.no_triad:
        A = np.asarray(doc["lattice_vectors"], float)               # rows are a, b, c
        corners = np.array([i * A[0] + j * A[1] + k * A[2]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        pts = np.vstack([corners, positions])
        mn, mx = pts.min(0), pts.max(0)
        extent = float((mx - mn).max())
        # anchor below the cell, toward the default camera (+,+,+), so the triad sits
        # clearly outside the box in the front-bottom — VESTA-like and never inside it.
        origin = np.array([mx[0] + 0.12 * extent, mx[1] + 0.12 * extent,
                           mn[2] - 0.12 * extent])
        static += axis_triad(origin, 0.32 * extent)

    base_meshes = frame_mesh_traces(doc, Z, faces, directions, 0, args, clim_of(0))
    fig = go.Figure(data=base_meshes + static)

    # animate only the mesh traces (indices 0..natoms-1); overlays stay put
    go_frames = []
    for k in range(len(frames)):
        go_frames.append(go.Frame(
            name=str(k),
            data=frame_mesh_traces(doc, Z, faces, directions, k, args, clim_of(k)),
            traces=list(range(natoms))))
    fig.frames = go_frames

    sliders = []
    # temperature slider (animates the mesh traces)
    if len(frames) > 1:
        steps = [dict(method="animate", label=f"{temps[k]:.3f}",
                      args=[[str(k)], dict(mode="immediate",
                                           frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))])
                 for k in range(len(frames))]
        sliders.append(dict(active=0, currentvalue=dict(prefix="τ = "),
                            x=0.0, len=1.0, y=0.0, pad=dict(t=30, b=10), steps=steps))
    # arrow-size slider: show exactly one size group (or none at 0×) by toggling visibility
    if arrow_groups:
        all_idx = [i for _, idxs in arrow_groups for i in idxs]
        steps = []
        for mu in arrow_mults:
            on = next((idxs for g, idxs in arrow_groups if g == mu), [])
            steps.append(dict(method="restyle", label=f"{mu:.1f}×",
                              args=[{"visible": [i in on for i in all_idx]}, all_idx]))
        sliders.append(dict(active=arrow_mults.index(default_mu),
                            currentvalue=dict(prefix="arrow size "),
                            x=0.0, len=1.0, y=-0.16, pad=dict(t=30, b=10), steps=steps))

    # VESTA-style: hide the boxed tick axes / grid planes (the xyz arrow triad replaces
    # them). `--grid-axes` restores the labelled Plotly axes instead.
    if args.grid_axes:
        scene = dict(aspectmode="data",
                     xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)")
    else:
        off = dict(visible=False, showgrid=False, showticklabels=False,
                   zeroline=False, showbackground=False, title="")
        scene = dict(aspectmode="data", xaxis=off, yaxis=off, zaxis=off)

    fig.update_layout(title="MFA per-atom orientation distributions",
                      scene=scene, sliders=sliders, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--scale", type=float, default=0.0,
                        help="lobe-deformation strength (default 0 = a true sphere, "
                             "colour-only heatmap; >0 bulges the sphere along the lobes)")
    parser.add_argument("--arrow-scale", type=float, default=1.0,
                        help="initial mean-moment arrow size (also adjustable live with the "
                             "in-figure arrow-size slider)")
    parser.add_argument("--head-frac", type=float, default=0.5,
                        help="fraction of the moment arrow taken by the cone head (default "
                             "0.5; smaller = a longer shaft and a smaller head)")
    parser.add_argument("--cmap", default="Inferno")
    parser.add_argument("--shared-clim", action="store_true")
    parser.add_argument("--no-cell", action="store_true")
    parser.add_argument("--no-arrows", action="store_true")
    parser.add_argument("--no-triad", action="store_true",
                        help="hide the VESTA-style x/y/z axis arrows")
    parser.add_argument("--grid-axes", action="store_true",
                        help="restore the boxed Plotly tick axes instead of the arrow triad")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    doc = load(args.path)
    fig = build_figure(doc, args)
    out = args.out or os.path.splitext(args.path)[0] + ".html"
    fig.write_html(out, include_plotlyjs="cdn", auto_open=False)
    print(f"wrote {out} — open it in a browser and drag to rotate; use the τ slider.")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
