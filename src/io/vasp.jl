# VASP input writing — turn sampled spin configurations into constrained-noncollinear VASP
# inputs (see the active-learning "label" step). Namespaced as `SCETools.VASP`, mirroring the
# reader-side `MagestyRebuild.VASP`. A sampler produces unit spin *directions*; the moment
# *magnitudes* (μ_B) come from elsewhere — a template INCAR's existing MAGMOM (each atom's
# norm), a per-species map, or an explicit per-atom vector.
#
# Conventions (matched to `MagestyRebuild.VASP`'s reader):
#   - MAGMOM / M_CONSTR carry full 3-component vectors per atom, `%.9f`, written in the
#     **global Cartesian** frame for the default `SAXIS = [0,0,1]`; for a non-default SAXIS the
#     vectors are rotated *out* of Cartesian by `Rᵀ` (the inverse of the reader's `Rz(α)Ry(β)`).
#   - The atom order of MAGMOM / M_CONSTR must match the POSCAR. `write_inputs` writes both and
#     orders the moments to match `MagestyRebuild.VASP.write_poscar` (atoms grouped by species
#     in `species_labels` order).
module VASP

using Printf
using LinearAlgebra: norm
import MagestyRebuild
using MagestyRebuild: Crystal, num_atoms

export write_incar, write_inputs

# --- SAXIS frame -------------------------------------------------------------------------

_is_default_saxis(s) = s[1] == 0 && s[2] == 0 && s[3] == 1

# The reader rotates moments from the SAXIS frame to Cartesian by R = Rz(α)·Ry(β); the writer
# rotates Cartesian → SAXIS by Rᵀ, so a write→read round-trip is the identity.
function _saxis_rotation(s)
    sx, sy, sz = float(s[1]), float(s[2]), float(s[3])
    α = atan(sy, sx)
    β = atan(hypot(sx, sy), sz)
    Rz = [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    Ry = [cos(β) 0.0 sin(β); 0.0 1.0 0.0; -sin(β) 0.0 cos(β)]
    return Rz * Ry
end

# --- moment vectors ----------------------------------------------------------------------

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

# Do two SAXIS vectors point the same way (the magnitude is irrelevant — only the axis matters)?
_saxis_approx(a, b) = isapprox(collect(float.(a)) ./ norm(float.(collect(a))),
                              collect(float.(b)) ./ norm(float.(collect(b))); atol = 1e-8)

# "x y z  x y z  …" with `%.9f`, double space between atoms (matches the VASP/Magesty layout).
_format_moments(M::AbstractMatrix{<:Real}) =
    join((@sprintf("%.9f %.9f %.9f", M[1, a], M[2, a], M[3, a]) for a = 1:size(M, 2)), "  ")

# --- INCAR template parsing (line-based passthrough) -------------------------------------

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

# Read a template INCAR (path or raw text). Returns `(kept_lines, magmom, has_icm, saxis)`:
# every line except the MAGMOM / M_CONSTR assignments (preserved verbatim, comments and all),
# the parsed MAGMOM vector (or `nothing`) used as the moment-magnitude source, whether the
# template constrains (`I_CONSTRAINED_M`), and the template's SAXIS (or `nothing`).
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

# --- magnitude resolution ----------------------------------------------------------------

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
# scalar (uniform), a per-atom vector, a per-species `label => magnitude` map, or `nothing`
# (take the magnitudes from `template_magmom`).
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

# --- INCAR writing -----------------------------------------------------------------------

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
                     extra = Pair[], _species = nothing, _labels = nothing)
    n = size(directions, 2)
    kept, tmpl_mag, has_icm, tmpl_saxis =
        base === nothing ? (String[], nothing, false, nothing) : _process_template(base)
    mags = _resolve_magmoms(magmoms, n, _species, _labels, tmpl_mag)

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

# --- full input sets (POSCAR + INCAR) ----------------------------------------------------

# The atom order `MagestyRebuild.VASP.write_poscar` uses: atoms grouped by species in
# `species_labels` order, original order within a species.
function _poscar_order(crystal::Crystal)::Vector{Int}
    perm = Int[]
    for s = 1:length(crystal.species_labels)
        for a = 1:num_atoms(crystal)
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

Write a complete VASP input set — a `POSCAR` (via `MagestyRebuild.VASP.write_poscar`) and a
matching `INCAR` — for a spin configuration `config` (`3 × n_atoms` unit columns) on `crystal`.
The INCAR's MAGMOM / M_CONSTR are ordered to match the POSCAR (atoms grouped by species), so the
two files are always consistent.

`magmoms` may additionally be a per-species `label => magnitude` map here (resolved through the
crystal); everything else is forwarded to [`write_incar`](@ref). The second form writes a sweep:
one subdirectory `"<prefix>-NNN"` per configuration in `configs` (e.g. pass `samp.configs` from
an `MFASample`). Returns the directory (or the vector of directories).
"""
function write_inputs(dir::AbstractString, crystal::Crystal, config::AbstractMatrix{<:Real};
                      magmoms = nothing, base = nothing, comment = "generated by SCETools",
                      kwargs...)
    num_atoms(crystal) == size(config, 2) || throw(ArgumentError(
        "config has $(size(config, 2)) atoms but the crystal has $(num_atoms(crystal))"))
    mkpath(dir)
    MagestyRebuild.VASP.write_poscar(joinpath(dir, "POSCAR"), crystal; comment = comment)

    n = num_atoms(crystal)
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
