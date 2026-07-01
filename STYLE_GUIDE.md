# Style guide

> The shared Julia style lives in `~/Packages/CLAUDE.md` ("Julia style"): official
> Julia + DFTK guides, naming, `for i = 1:n` vs `for x in xs`, explicit named
> tuples, ≤ 92 cols, hot-path `SVector`/`MVector`/`@views`/`@inbounds`. Only
> package-specific additions are listed here.

## Physics notation

- Unicode physics names are fine (and preferred) where they match the spec /
  literature: `τ` (reduced temperature), `κ` (vMF concentration), `β` (inverse
  temperature), `ρ` (Perron eigenvalue), `θ`/`φ` (angles), `Ā` (normalized
  molecular-field matrix). Keyword arguments and exported names stay ASCII
  (`tau`, `m`, `nsamples`).
- Conventions themselves (unit spins, `3 × n_atoms` layout, tesseral `Z_lm`,
  `τ = T/T_MF`) are owned by `SCEFitting` and `docs/specs/mfa-sampling.md` —
  never restate or re-derive them in code comments; link them.

## Naming and API tiers

- A leading `_` means **private** — not part of the public surface even when
  reachable (e.g. `MeanFieldEngine._random_unit`); do not document, export, or
  depend on such names across module boundaries beyond the explicit
  `using .MeanFieldEngine: …` re-binds in `src/SCETools.jl`.
- The public surface is two-tiered: a lean `export` list plus a
  **`public`-keyword** tier reached by qualification (`SCETools.MultipoleModel`,
  `SCETools.MeanFieldEngine`, `SCETools.VASP`). New public-but-unexported names
  must be added to the `public` block (machine-checked via `Base.ispublic` / Aqua).

## Inner constructors

Types with invariants (`ExchangeModel`, `MultipoleModel`, `MFASampler`,
`MFASample`) validate them in an **inner** constructor, so an exactly field-typed
call cannot bypass the checks; convenience outer constructors route through it.
