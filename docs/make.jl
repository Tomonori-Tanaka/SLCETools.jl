using SLCETools
using SLCE   # the SLCE fitting core, for the executed `@example` model builds
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(SLCETools, :DocTestSetup, :(using SLCETools); recursive = true)

makedocs(;
    sitename = "SLCETools.jl",
    modules = [SLCETools],
    repo = Remotes.GitHub("Tomonori-Tanaka", "SLCETools.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        canonical = "https://tomonori-tanaka.github.io/SLCETools.jl/dev",
        edit_link = "main",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guide" => [
            "guide/sampling.md",
            "guide/mc_sampling.md",
            "guide/exchange_models.md",
            "guide/vasp.md",
            "guide/distributions.md",
        ],
        "Theory" => [
            "theory/mfa.md",
        ],
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
    doctest = false,
)

# Publishes to https://tomonori-tanaka.github.io/SLCETools.jl/ from the `documentation build`
# CI job (which needs `permissions: contents: write`). Outside CI this is a no-op, so a
# local `julia --project=docs docs/make.jl` still just builds into `docs/build/`.
deploydocs(;
    repo = "github.com/Tomonori-Tanaka/SLCETools.jl",
    devbranch = "main",
    push_preview = false,
)
