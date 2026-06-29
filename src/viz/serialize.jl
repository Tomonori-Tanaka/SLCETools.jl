# Per-atom MFA distribution viz — the self-describing JSON document and a dependency-free
# emitter. `_distribution_doc` builds a serializer-agnostic Dict tree (unit-testable on its
# own); `_emit_json` writes it with no external JSON dependency (the schema is fixed and
# shallow). Grid construction is in `grid.jl`, the per-atom coefficients in `distributions.jl`.

# Normalize -0.0 → 0.0 for reproducible bytes (mirrors SCEFitting's persist.jl `_jnum`).
_jnum(x::Real)::Float64 = (y = Float64(x); y == 0.0 ? 0.0 : y)

# Build the self-describing document (a Dict tree, serializer-agnostic and unit-testable).
function _distribution_doc(sampler::MFASampler, crystal::Crystal;
                           taus::AbstractVector{<:Real}, npoints::Integer)
    n = _natoms(sampler)
    n == n_atoms(crystal) || throw(ArgumentError(
        "sampler atom count $n ≠ crystal atom count $(n_atoms(crystal))"))
    isempty(taus) && throw(ArgumentError("taus must be non-empty"))

    grid = fibonacci_sphere(npoints)
    fields = [mfa_site_coefficients(sampler, τ) for τ in taus]
    lmax = fields[1].lmax
    all(f.lmax == lmax for f in fields) ||
        throw(ArgumentError("lmax varies across τ; expected a single fixed lmax"))
    Z = harmonic_basis(grid, lmax)
    nlm = (lmax + 1)^2

    A = crystal.lattice.vectors                          # columns are lattice vectors
    latt = [[_jnum(A[1, j]), _jnum(A[2, j]), _jnum(A[3, j])] for j = 1:3]
    R = A * crystal.frac_positions                       # 3×n Cartesian positions (Å)
    pos = [[_jnum(R[1, a]), _jnum(R[2, a]), _jnum(R[3, a])] for a = 1:n]
    ref = sampler.reference
    refs = [[_jnum(ref[1, a]), _jnum(ref[2, a]), _jnum(ref[3, a])] for a = 1:n]

    dirs = [[_jnum(d[1]), _jnum(d[2]), _jnum(d[3])] for d in grid.dirs]
    Zrows = [[_jnum(Z[i, k]) for k = 1:nlm] for i = 1:length(grid.dirs)]

    frames = Vector{Dict{String,Any}}(undef, length(taus))
    for (t, τ) in enumerate(taus)
        f = fields[t]
        frames[t] = Dict{String,Any}(
            "tau" => _jnum(τ),
            "m" => [_jnum(x) for x in f.m],
            "coeffs" => [[_jnum(c) for c in f.coeffs[a]] for a = 1:n])
    end

    return Dict{String,Any}(
        "schema" => "scetools/mfa-distributions",
        "version" => 1,
        "lattice_vectors" => latt,
        "positions_cartesian" => pos,
        "species" => collect(crystal.species),
        "species_labels" => collect(crystal.species_labels),
        "reference" => refs,
        "lmax" => lmax,
        "T_MF" => _jnum(mfa_temperature_scale(sampler)),
        "grid" => Dict{String,Any}(
            "kind" => "fibonacci",
            "npoints" => length(grid.dirs),
            "weight" => _jnum(grid.weight),
            "directions" => dirs,
            "Z" => Zrows),
        "temperatures" => [_jnum(τ) for τ in taus],
        "frames" => frames)
end

# A minimal JSON emitter for the fixed, shallow schema (Float64 / Int / String / Bool /
# array / Dict only) — keeps the package's zero-extra-dependency property. Keys are sorted
# for reproducible bytes.
function _emit_json(io::IO, x)
    if x isa AbstractString
        print(io, '"')
        for ch in x
            if ch == '"'
                print(io, "\\\"")
            elseif ch == '\\'
                print(io, "\\\\")
            elseif ch == '\n'
                print(io, "\\n")
            elseif ch == '\r'
                print(io, "\\r")
            elseif ch == '\t'
                print(io, "\\t")
            elseif ch < '\x20'                       # RFC 8259: escape all control chars
                print(io, "\\u", lpad(string(UInt16(ch); base = 16), 4, '0'))
            else
                print(io, ch)
            end
        end
        print(io, '"')
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa Real
        y = Float64(x)
        isfinite(y) || throw(ArgumentError("non-finite value $y cannot be JSON-encoded"))
        print(io, y)
    elseif x isa AbstractDict
        print(io, '{')
        for (i, k) in enumerate(sort(collect(keys(x))))
            i > 1 && print(io, ',')
            _emit_json(io, String(k))
            print(io, ':')
            _emit_json(io, x[k])
        end
        print(io, '}')
    elseif x isa AbstractVector
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ',')
            _emit_json(io, v)
        end
        print(io, ']')
    else
        throw(ArgumentError("unsupported JSON type $(typeof(x))"))
    end
    return io
end

"""
    write_mfa_distributions(path, sampler, crystal; taus, npoints = 1000) -> path

Write the per-atom MFA probability distributions over `taus` to a single self-describing
JSON file (schema `scetools/mfa-distributions`, version 1) for the Python viewer. The file
carries the structure (lattice vectors as rows, Cartesian positions, species, reference
axes), the shared Fibonacci render grid and its tesseral basis matrix `Z`, and one frame
per τ holding the per-atom magnetizations `m_a` and coefficient vectors `c_a`. The viewer
recovers each atom's density as `exp(−Z·c_a)` (normalized). Requires
`size(sampler.reference, 2) == n_atoms(crystal)`.
"""
function write_mfa_distributions(path::AbstractString, sampler::MFASampler, crystal::Crystal;
                                 taus::AbstractVector{<:Real}, npoints::Integer = 1000)
    doc = _distribution_doc(sampler, crystal; taus = taus, npoints = npoints)
    open(path, "w") do io
        _emit_json(io, doc)
        print(io, '\n')
    end
    return path
end
