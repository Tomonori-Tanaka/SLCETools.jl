# VASP I/O — the code-specific adapter for the VASP DFT code, kept out of the code-agnostic
# fitting core. The core (`SCEFitting`) owns only the abstract DFT-data seam
# (`AbstractDFTSource` / `SpinDatum` / `read_configs` / `SCEDataset`); this module provides the
# concrete VASP reader and writer:
#
#   - read:  `read_poscar` (POSCAR → Crystal), `Oszicar` (constrained-noncollinear OSZICARs →
#            SpinDatum via the `AbstractDFTSource` / `read_configs` seam) — produces training data.
#   - write: `write_poscar` (Crystal → POSCAR), `write_incar` / `write_inputs` (sampled spin
#            directions → constrained-noncollinear INCAR / input sets) — produces DFT jobs.
#
# Conventions follow VASP and Magesty.jl: the POSCAR scaling factor is a length scale (or, if
# negative, a target cell volume); the OSZICAR moment is the `MW_int` column (or `M_int` with
# `mint = true`) and the constraining field is `lambda*MW_perp`; both are rotated between the
# `SAXIS` quantization frame and Cartesian by `Rz(α)·Ry(β)` (read) / its inverse `Rᵀ` (write), so
# a write → read round-trip is the identity. The torque target is `τ_a = m_a × B_a` (eV).
# MAGMOM / M_CONSTR carry full 3-component vectors per atom (`%.9f`), and their atom order must
# match the POSCAR (atoms grouped by species in `species_labels` order — `write_inputs` handles
# this). A second DFT code is one more sibling adapter; the core does not change.
module VASP

using Printf
using StaticArrays
using LinearAlgebra: norm, det
using SCEFitting: Crystal, Lattice, AbstractDFTSource, SpinDatum, n_atoms
import SCEFitting: read_configs

export read_poscar, write_poscar, Oszicar, write_incar, write_inputs

# --- SAXIS frame -------------------------------------------------------------------------

_is_default_saxis(s) = s[1] == 0 && s[2] == 0 && s[3] == 1

# The SAXIS quantization-axis rotation R = Rz(α)·Ry(β), α/β the azimuth/polar of `saxis`. The
# reader rotates moments from the SAXIS frame to Cartesian by R; the writer rotates Cartesian →
# SAXIS by Rᵀ (= R⁻¹), so a write → read round-trip is the identity.
function _saxis_rotation(s)::SMatrix{3,3,Float64,9}
    sx, sy, sz = float(s[1]), float(s[2]), float(s[3])
    n = sqrt(sx^2 + sy^2 + sz^2)
    (isfinite(n) && n > 0) ||
        throw(ArgumentError("saxis must be a nonzero, finite vector; got $s"))
    α = atan(sy, sx)
    β = atan(hypot(sx, sy), sz)
    Rz = @SMatrix [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    Ry = @SMatrix [cos(β) 0.0 sin(β); 0.0 1.0 0.0; -sin(β) 0.0 cos(β)]
    return Rz * Ry
end

# Do two SAXIS vectors point the same way (the magnitude is irrelevant — only the axis matters)?
_saxis_approx(a, b) = isapprox(collect(float.(a)) ./ norm(float.(collect(a))),
                              collect(float.(b)) ./ norm(float.(collect(b))); atol = 1e-8)

# ── POSCAR ────────────────────────────────────────────────────────────────────────────

"""
    SCETools.VASP.read_poscar(path) -> Crystal

Read a VASP POSCAR/CONTCAR file into a `Crystal`. Handles the scaling factor (a length scale,
or a target volume when negative), `Direct`/`Cartesian` coordinates, the optional
`Selective dynamics` line, and POSCAR files with or without the element-symbol line
(synthesizing `"X1", "X2", …` when it is absent). Each of the three lattice lines is one lattice
vector (a column of `Lattice.vectors`).
"""
function read_poscar(path::AbstractString)::Crystal
    isfile(path) || throw(ArgumentError("POSCAR not found: $path"))
    lines = readlines(path)
    length(lines) >= 8 || throw(ArgumentError("POSCAR is too short to be valid"))

    scale = tryparse(Float64, strip(lines[2]))
    scale === nothing && throw(ArgumentError("invalid POSCAR scaling factor: $(lines[2])"))

    A_raw = MMatrix{3,3,Float64}(undef)
    for i = 1:3
        toks = split(strip(lines[2 + i]))
        length(toks) >= 3 || throw(ArgumentError("bad lattice vector on line $(2 + i)"))
        for j = 1:3
            A_raw[j, i] = parse(Float64, toks[j])   # column i = i-th lattice vector
        end
    end
    # Negative scale = target volume (VASP convention); otherwise a plain length scale.
    s = scale >= 0 ? scale : cbrt(abs(scale) / abs(det(SMatrix(A_raw))))
    A = SMatrix(A_raw) .* s

    toks6 = split(strip(lines[6]))
    counts = map(t -> tryparse(Int, t), toks6)
    local labels::Vector{String}, numbers::Vector{Int}, coordline::Int
    if all(!isnothing, counts) && all(c -> c > 0, counts)
        numbers = Int[c for c in counts]                # VASP4: no symbols line
        labels = String["X$i" for i = 1:length(numbers)]
        coordline = 7
    else
        labels = String.(toks6)                         # VASP5: symbols then counts
        cnt = map(t -> tryparse(Int, t), split(strip(lines[7])))
        all(!isnothing, cnt) && all(c -> c > 0, cnt) ||
            throw(ArgumentError("bad atom counts on POSCAR line 7: $(lines[7])"))
        numbers = Int[c for c in cnt]
        coordline = 8
    end
    length(labels) == length(numbers) ||
        throw(ArgumentError("POSCAR: $(length(labels)) species symbols vs $(length(numbers)) counts"))

    mode = lowercase(strip(lines[coordline]))
    if startswith(mode, "s")                            # optional Selective dynamics line
        coordline += 1
        coordline <= length(lines) ||
            throw(ArgumentError("POSCAR ends after 'Selective dynamics' (no coordinate-mode line)"))
        mode = lowercase(strip(lines[coordline]))
    end
    cartesian = startswith(mode, "c") || startswith(mode, "k")
    startswith(mode, "d") || cartesian ||
        throw(ArgumentError("invalid POSCAR coordinate mode: $(lines[coordline])"))

    nat = sum(numbers)
    pos = Matrix{Float64}(undef, 3, nat)
    first = coordline + 1
    for a = 1:nat
        li = first + a - 1
        li <= length(lines) || throw(ArgumentError("POSCAR ends before all $nat positions"))
        toks = split(strip(lines[li]))
        length(toks) >= 3 || throw(ArgumentError("bad position on POSCAR line $li"))
        for j = 1:3
            pos[j, a] = parse(Float64, toks[j])
        end
    end
    # Cartesian coords carry the same scaling; converting to fractional cancels it.
    frac = cartesian ? (A \ (s .* pos)) : pos

    species = Vector{Int}(undef, nat)
    a = 0
    for (si, c) in enumerate(numbers), _ = 1:c
        species[a += 1] = si
    end
    return Crystal(Lattice(A), frac, species, labels)
end

"""
    SCETools.VASP.write_poscar(path, crystal; comment = "…", cartesian = false)

Write `crystal` to a VASP POSCAR file (scaling factor `1.0`, atoms grouped by species in
`species_labels` order, `Direct` coordinates unless `cartesian = true`). Species with no atoms
are omitted from the symbol/count lines. Note: atoms are **reordered** into species groups, so
any external per-atom array (forces, charges, …) indexed by the original atom order must be
permuted to match (e.g. [`write_inputs`](@ref) does this for MAGMOM).
"""
function write_poscar(path::AbstractString, crystal::Crystal;
                      comment::AbstractString = "generated by SCETools",
                      cartesian::Bool = false)
    A = crystal.lattice.vectors
    nsp = length(crystal.species_labels)
    groups = [findall(==(s), crystal.species) for s = 1:nsp]
    present = [s for s = 1:nsp if !isempty(groups[s])]
    row(v) = join((string(v[k]) for k = 1:3), "  ")
    open(path, "w") do io
        println(io, comment)
        println(io, "1.0")
        for i = 1:3
            println(io, "  ", row(@view A[:, i]))       # line i = i-th lattice vector
        end
        println(io, join((crystal.species_labels[s] for s in present), " "))
        println(io, join((length(groups[s]) for s in present), " "))
        println(io, cartesian ? "Cartesian" : "Direct")
        for s in present, a in groups[s]
            f = SVector{3,Float64}(crystal.frac_positions[1, a], crystal.frac_positions[2, a],
                                   crystal.frac_positions[3, a])
            println(io, "  ", row(cartesian ? A * f : f))
        end
    end
    return path
end

# ── OSZICAR (constrained-noncollinear training data) ──────────────────────────────────

"""
    SCETools.VASP.Oszicar(paths; saxis = [0, 0, 1], energy_kind = :free, mint = false)

An `AbstractDFTSource` over one or more VASP OSZICAR files — each contributes one `SpinDatum`
(in the given order, through `SCEFitting.read_configs`). `paths` may be a single path or a
vector.

# Keyword arguments
- `saxis`: the `SAXIS` quantization axis; moments and fields are rotated from this frame into
  Cartesian coordinates by `Rz(α)·Ry(β)` (default `[0,0,1]` = identity).
- `energy_kind`: `:free` (the `F=` free energy) or `:sigma0` (`E0`, energy σ→0).
- `mint`: read the moment from the `M_int` columns instead of `MW_int`. The constraining field
  `lambda*MW_perp` is referenced to the `MW_int` moment, so the default `mint = false` keeps the
  moment and field (hence the torque) mutually consistent.
"""
struct Oszicar <: AbstractDFTSource
    paths::Vector{String}
    saxis::SVector{3,Float64}
    energy_kind::Symbol
    mint::Bool
end

function Oszicar(paths::AbstractVector{<:AbstractString};
                saxis = SVector{3,Float64}(0, 0, 1),
                energy_kind::Symbol = :free, mint::Bool = false)
    energy_kind in (:free, :sigma0) ||
        throw(ArgumentError("energy_kind must be :free or :sigma0; got $(repr(energy_kind))"))
    return Oszicar(collect(String, paths), SVector{3,Float64}(saxis), energy_kind, mint)
end
Oszicar(path::AbstractString; kwargs...) = Oszicar([path]; kwargs...)

# Final-step energy and per-atom moment vectors (3 × n_atoms) from one OSZICAR. The moment block
# is the `ion … MW_int … M_int` table (7 columns: idx + MW_int xyz + M_int xyz); it ends at a `:`
# line or any non-7-column line (e.g. the `… F= … E0= …` summary). The last committed block /
# last `F=` line (final ionic step) wins.
function _oszicar_energy_moments(path::AbstractString, energy_kind::Symbol, mint::Bool)
    energy = 0.0
    found_e = false
    moments = SVector{3,Float64}[]
    tmp = SVector{3,Float64}[]
    collecting = false
    cols = mint ? (5, 6, 7) : (2, 3, 4)
    for line in eachline(path)
        if occursin("M_int", line)
            collecting = true
            empty!(tmp)
            continue
        end
        if collecting && (occursin(":", line) || length(split(line)) != 7)
            collecting = false
            moments = copy(tmp)
        end
        if collecting
            p = split(line)
            push!(tmp, SVector{3,Float64}(parse(Float64, p[cols[1]]),
                                          parse(Float64, p[cols[2]]),
                                          parse(Float64, p[cols[3]])))
        end
        if occursin("F=", line)
            # take the token right after the keyword (robust to other fields shifting),
            # rather than a fixed column index.
            p = split(line)
            key = energy_kind === :free ? "F=" : "E0="
            k = findfirst(==(key), p)
            if k !== nothing && k < length(p)
                v = tryparse(Float64, p[k + 1])
                v === nothing || (energy = v; found_e = true)
            end
        end
    end
    collecting && (moments = copy(tmp))                 # block running at EOF
    found_e || throw(ArgumentError("no energy (F= line) found in $path"))
    isempty(moments) &&
        throw(ArgumentError("no magnetic-moment (M_int) block found in $path"))
    return energy, reduce(hcat, moments)                # 3 × n_atoms
end

# Per-atom constraining field (3 × n_atoms) from the `lambda*MW_perp` block; rows are
# `idx Bx By Bz`, only for constrained atoms (others stay zero). The last block wins; a missing
# block (unconstrained run) yields a zero field.
function _oszicar_field(path::AbstractString, nat::Int)::Matrix{Float64}
    field = zeros(3, nat)
    tmp = zeros(3, nat)
    in_block = false
    got = false
    for line in eachline(path)
        if occursin("lambda*MW_perp", line)
            in_block = true
            fill!(tmp, 0.0)
            got = false
            continue
        end
        if in_block
            p = split(line)
            idx = length(p) == 4 ? tryparse(Int, p[1]) : nothing
            if idx !== nothing && 1 <= idx <= nat
                tmp[1, idx] = parse(Float64, p[2])
                tmp[2, idx] = parse(Float64, p[3])
                tmp[3, idx] = parse(Float64, p[4])
                got = true
                continue
            else
                in_block = false
                got && (field .= tmp)
            end
        end
    end
    in_block && got && (field .= tmp)                   # block running at EOF
    return field
end

function read_configs(src::Oszicar)::Vector{SpinDatum}
    R = _saxis_rotation(src.saxis)
    data = Vector{SpinDatum}(undef, length(src.paths))
    for (i, path) in enumerate(src.paths)
        isfile(path) || throw(ArgumentError("OSZICAR not found: $path"))
        energy, moments = _oszicar_energy_moments(path, src.energy_kind, src.mint)
        field = _oszicar_field(path, size(moments, 2))
        data[i] = SpinDatum(energy, R * moments, R * field)   # SAXIS → Cartesian frame
    end
    return data
end

# ── INCAR writing (sampled directions → constrained-noncollinear inputs) ───────────────

# The 3 × n moment matrix: column a = magmoms[a] · (unit direction a), in the SAXIS frame.
function _moment_matrix(directions::AbstractMatrix{<:Real}, magmoms::AbstractVector{<:Real}, saxis)
    n = size(directions, 2)
    size(directions, 1) == 3 ||
        throw(ArgumentError("directions must be 3 × n_atoms; got $(size(directions))"))
    length(magmoms) == n ||
        throw(ArgumentError("magmoms must have one magnitude per atom ($n); got $(length(magmoms))"))
    M = Matrix{Float64}(undef, 3, n)
    @inbounds for a = 1:n
        magmoms[a] >= 0 ||
            throw(ArgumentError("magmoms must be non-negative magnitudes (μ_B); got $(magmoms[a]) on atom $a"))
        d = @view directions[:, a]
        nd = norm(d)
        nd > 0 || throw(ArgumentError("direction of atom $a has zero norm"))
        @. M[:, a] = magmoms[a] * d / nd
    end
    _is_default_saxis(saxis) || (M = transpose(_saxis_rotation(saxis)) * M)
    return M
end

# "x y z  x y z  …" with `%.9f`, double space between atoms (matches the VASP/Magesty layout).
_format_moments(M::AbstractMatrix{<:Real}) =
    join((@sprintf("%.9f %.9f %.9f", M[1, a], M[2, a], M[3, a]) for a = 1:size(M, 2)), "  ")

# Expand a VASP numeric value list, honouring the `n*v` repeat syntax, to a flat vector.
function _parse_floats(s::AbstractString)::Vector{Float64}
    out = Float64[]
    for tok in split(s)
        if occursin('*', tok)
            nstr, vstr = split(tok, '*'; limit = 2)
            append!(out, fill(parse(Float64, vstr), parse(Int, nstr)))
        else
            push!(out, parse(Float64, tok))
        end
    end
    return out
end

_strip_comment(line::AbstractString) = strip(replace(line, r"[#!].*$" => ""))

# Join `\`-continued lines (VASP wraps long MAGMOM vectors this way) into single logical lines.
function _join_continuations(text::AbstractString)::Vector{String}
    lines = String[]
    buf = ""
    for raw in split(text, '\n')
        line = rstrip(raw)
        if endswith(line, '\\')
            buf *= chop(line) * " "
        else
            push!(lines, buf * line)
            buf = ""
        end
    end
    isempty(buf) || push!(lines, buf)
    return lines
end

const _TAG_RE = r"^\s*([A-Za-z_0-9]+)\s*="

function _tag_key(bare::AbstractString)
    m = match(_TAG_RE, bare)
    m === nothing && return ""
    cap = m.captures[1]
    cap === nothing && return ""
    return uppercase(cap)
end

# Resolve a `base` argument to INCAR text: an existing file is read; otherwise the string is
# treated as raw INCAR text only if it actually looks like one (a typo'd path errors instead of
# being silently embedded as a stray line).
function _incar_text(base)::String
    isfile(base) && return read(base, String)
    (occursin('\n', base) || occursin('=', base)) && return String(base)
    throw(ArgumentError("`base` looks like a file path but does not exist: $(base)"))
end

# Read a template INCAR (path or raw text). Returns `(kept_lines, magmom, has_icm, saxis)`: every
# line except the MAGMOM / M_CONSTR assignments (preserved verbatim, comments and all), the
# parsed MAGMOM vector (or `nothing`) used as the moment-magnitude source, whether the template
# constrains (`I_CONSTRAINED_M`), and the template's SAXIS (or `nothing`).
function _process_template(base)
    kept = String[]
    magmom = nothing
    has_icm = false
    saxis = nothing
    for line in _join_continuations(_incar_text(base))
        bare = _strip_comment(line)
        key = _tag_key(bare)
        if key == "MAGMOM"
            magmom = _parse_floats(strip(split(bare, '='; limit = 2)[2]))
        elseif key == "M_CONSTR"
            # dropped; re-emitted from the sampled directions
        else
            key == "I_CONSTRAINED_M" && (has_icm = true)
            key == "SAXIS" && (saxis = Tuple(_parse_floats(strip(split(bare, '='; limit = 2)[2]))))
            push!(kept, line)
        end
    end
    return kept, magmom, has_icm, saxis
end

# Per-atom magnitudes from a template's MAGMOM: a `3n` noncollinear vector → per-atom norms; an
# `n` collinear vector → used directly.
function _magmoms_from_template(magmom::Vector{Float64}, n::Int)::Vector{Float64}
    if length(magmom) == 3n
        return [norm(@view magmom[3(a-1)+1:3a]) for a = 1:n]
    elseif length(magmom) == n
        return copy(magmom)
    end
    throw(ArgumentError("template MAGMOM has $(length(magmom)) entries; expected $n (collinear) " *
                        "or $(3n) (noncollinear) for $n atoms"))
end

# Resolve the `magmoms` argument to a per-atom vector (in the crystal's atom order). Accepts a
# scalar (uniform), a per-atom vector, a per-species `label => magnitude` map, or `nothing` (take
# the magnitudes from `template_magmom`).
function _resolve_magmoms(magmoms, n::Int, species, labels,
                          template_magmom::Union{Nothing,Vector{Float64}})::Vector{Float64}
    if magmoms === nothing
        template_magmom === nothing && throw(ArgumentError(
            "no `magmoms` given and no template MAGMOM to take magnitudes from; pass " *
            "`magmoms = <per-atom vector | per-species map | scalar>` or a `base` INCAR with MAGMOM"))
        return _magmoms_from_template(template_magmom, n)
    elseif magmoms isa Real
        return fill(Float64(magmoms), n)
    elseif magmoms isa AbstractDict || (magmoms isa AbstractVector && eltype(magmoms) <: Pair)
        species === nothing && throw(ArgumentError(
            "a per-species `magmoms` map needs the crystal; use `write_inputs` (not `write_incar`)"))
        d = Dict{String,Float64}(string(k) => Float64(v) for (k, v) in magmoms)
        return [haskey(d, labels[species[a]]) ? d[labels[species[a]]] :
                throw(ArgumentError("no magmoms entry for species \"$(labels[species[a]])\""))
                for a = 1:n]
    elseif magmoms isa AbstractVector{<:Real}
        length(magmoms) == n || throw(ArgumentError(
            "magmoms vector must have one entry per atom ($n); got $(length(magmoms))"))
        return collect(Float64, magmoms)
    end
    throw(ArgumentError("magmoms must be a scalar, a per-atom vector, or a per-species map"))
end

_fmt_value(v::Bool) = v ? ".TRUE." : ".FALSE."
_fmt_value(v) = string(v)

"""
    SCETools.VASP.write_incar(path, directions; magmoms = nothing, base = nothing,
                              constrain = true, saxis = nothing,
                              i_constrained_m = 1, lambda = 1.0, extra = Pair[])

Write a single VASP `INCAR` for one spin configuration `directions` (a `3 × n_atoms` matrix of
unit columns). The magnetic tags are built from the directions and the per-atom moment
magnitudes:

- `magmoms` — the moment magnitudes (μ_B): a scalar (uniform), a per-atom vector, or `nothing`
  to take them from the `base` template's MAGMOM (each atom's norm). A per-species map needs the
  crystal, so use [`write_inputs`](@ref) for that.
- `base` — a template INCAR (a path or raw text) whose tags are preserved verbatim; only MAGMOM
  and M_CONSTR are replaced. With `base = nothing` a minimal noncollinear INCAR is written
  (`LNONCOLLINEAR = .TRUE.`, MAGMOM, and — when `constrain` — `I_CONSTRAINED_M` / `LAMBDA` /
  M_CONSTR); supply `RWIGS` and the electronic-structure tags through `base` or `extra` for a
  runnable constrained calculation.
- `constrain` — also write `M_CONSTR` (equal to the moment vectors) for a direction-constrained
  noncollinear run. With a `base` template that already constrains, the template's
  `I_CONSTRAINED_M` / `LAMBDA` are preserved and the `i_constrained_m` / `lambda` kwargs are
  ignored (edit the template to change them); they apply only when writing from scratch.
- `saxis` — the VASP quantization axis. `nothing` (default) uses the `base` template's SAXIS if
  it has one, else the global frame; an explicit axis overrides it (and the moments are written
  in that SAXIS frame, the inverse of the reader's rotation, with a matching `SAXIS` line).
- `extra` — additional `"KEY" => value` tags appended to the file.

The MAGMOM / M_CONSTR atom order must match the POSCAR. Returns `path`.
"""
function write_incar(path::AbstractString, directions::AbstractMatrix{<:Real};
                     magmoms = nothing, base = nothing, constrain::Bool = true,
                     saxis = nothing, i_constrained_m::Integer = 1, lambda::Real = 1.0,
                     extra = Pair[])
    n = size(directions, 2)
    kept, tmpl_mag, has_icm, tmpl_saxis =
        base === nothing ? (String[], nothing, false, nothing) : _process_template(base)
    # no species context here: a per-species magmom Dict needs the crystal and is
    # resolved by `write_inputs`; passing one directly errors with a clear message
    mags = _resolve_magmoms(magmoms, n, nothing, nothing, tmpl_mag)

    # The frame the moments are written in must match the SAXIS the INCAR declares. Prefer an
    # explicit kwarg, else the template's SAXIS, else the global frame; warn if both are given
    # and disagree (writing in the wrong frame silently misaligns every spin in the run).
    override_saxis = saxis !== nothing && tmpl_saxis !== nothing && !_saxis_approx(saxis, tmpl_saxis)
    override_saxis && @warn "write_incar: the `saxis` kwarg disagrees with the template's " *
        "SAXIS; using the kwarg and overwriting the template's SAXIS line." maxlog = 1
    eff_saxis = saxis !== nothing ? saxis : (tmpl_saxis !== nothing ? tmpl_saxis : (0, 0, 1))
    M = _moment_matrix(directions, mags, eff_saxis)
    magstr = _format_moments(M)

    lines = String[]
    if base === nothing
        push!(lines, "# constrained-noncollinear INCAR generated by SCETools.VASP")
        push!(lines, "LNONCOLLINEAR = .TRUE.")
    else
        # Keep the template verbatim, dropping its SAXIS only when we are overriding it (so the
        # declared frame always matches the frame the moments were written in).
        for l in kept
            override_saxis && _tag_key(_strip_comment(l)) == "SAXIS" && continue
            push!(lines, l)
        end
    end
    # Declare SAXIS when the effective frame is non-default and a kept template line does not.
    kept_has_saxis = base !== nothing && tmpl_saxis !== nothing && !override_saxis
    !_is_default_saxis(eff_saxis) && !kept_has_saxis &&
        push!(lines, @sprintf("SAXIS = %.9f %.9f %.9f", eff_saxis[1], eff_saxis[2], eff_saxis[3]))
    push!(lines, "MAGMOM = " * magstr)
    if constrain
        if base === nothing
            push!(lines, "I_CONSTRAINED_M = $(i_constrained_m)")
            push!(lines, "LAMBDA = $(lambda)")
        elseif !has_icm
            @warn "write_incar: `constrain = true` but the template INCAR has no " *
                  "I_CONSTRAINED_M; the constraint will not activate unless you add it " *
                  "(and RWIGS) to the template." maxlog = 1
        end
        push!(lines, "M_CONSTR = " * magstr)
    end
    for (k, v) in extra
        push!(lines, "$(k) = $(_fmt_value(v))")
    end
    base === nothing && constrain &&
        @info "write_incar: a constrained run also needs RWIGS (per species) and the " *
              "electronic-structure tags; add them via `base` or `extra`." maxlog = 1

    open(path, "w") do io
        for l in lines
            println(io, l)
        end
    end
    return path
end

# The atom order `write_poscar` uses: atoms grouped by species in `species_labels` order,
# original order within a species.
function _poscar_order(crystal::Crystal)::Vector{Int}
    perm = Int[]
    for s = 1:length(crystal.species_labels)
        for a = 1:n_atoms(crystal)
            crystal.species[a] == s && push!(perm, a)
        end
    end
    return perm
end

"""
    SCETools.VASP.write_inputs(dir, crystal, config; magmoms = nothing, base = nothing,
                               constrain = true, saxis = nothing,
                               comment = "generated by SCETools", kwargs...)
    SCETools.VASP.write_inputs(rootdir, crystal, configs; prefix = "config", kwargs...)

Write a complete VASP input set — a `POSCAR` (via [`write_poscar`](@ref)) and a matching
`INCAR` — for a spin configuration `config` (`3 × n_atoms` unit columns) on `crystal`. The
INCAR's MAGMOM / M_CONSTR are ordered to match the POSCAR (atoms grouped by species), so the two
files are always consistent.

`magmoms` may additionally be a per-species `label => magnitude` map here (resolved through the
crystal); everything else is forwarded to [`write_incar`](@ref). The second form writes a sweep:
one subdirectory `"<prefix>-NNN"` per configuration in `configs` (e.g. pass `samp.configs` from
an `MFASample`). Returns the directory (or the vector of directories).
"""
function write_inputs(dir::AbstractString, crystal::Crystal, config::AbstractMatrix{<:Real};
                      magmoms = nothing, base = nothing, comment = "generated by SCETools",
                      kwargs...)
    n_atoms(crystal) == size(config, 2) || throw(ArgumentError(
        "config has $(size(config, 2)) atoms but the crystal has $(n_atoms(crystal))"))
    mkpath(dir)
    write_poscar(joinpath(dir, "POSCAR"), crystal; comment = comment)

    n = n_atoms(crystal)
    tmpl_mag = base === nothing ? nothing : _process_template(base)[2]
    mags = _resolve_magmoms(magmoms, n, crystal.species, crystal.species_labels, tmpl_mag)
    perm = _poscar_order(crystal)
    write_incar(joinpath(dir, "INCAR"), config[:, perm]; magmoms = mags[perm], base = base,
                kwargs...)
    return dir
end

function write_inputs(rootdir::AbstractString, crystal::Crystal,
                      configs::AbstractVector{<:AbstractMatrix}; prefix::AbstractString = "config",
                      kwargs...)
    mkpath(rootdir)
    width = max(3, ndigits(length(configs)))
    dirs = String[]
    for (i, cfg) in enumerate(configs)
        sub = joinpath(rootdir, string(prefix, "-", lpad(i, width, '0')))
        write_inputs(sub, crystal, cfg; kwargs...)
        push!(dirs, sub)
    end
    return dirs
end

end # module VASP
