using SCETools
using MagestyRebuild   # the SCE fitting core, for the executed `@example` model builds
using Documenter

DocMeta.setdocmeta!(SCETools, :DocTestSetup, :(using SCETools); recursive = true)

makedocs(;
    sitename = "SCETools.jl",
    modules = [SCETools],
    # Local-only build: there is no published remote yet, so do not try to resolve
    # "edit on GitHub" / source links. Add a `repolink`/`deploydocs` when a remote exists.
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        edit_link = nothing,
        repolink = "",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guide" => [
            "guide/sampling.md",
            "guide/exchange_models.md",
            "guide/dft_inputs.md",
        ],
        "Theory" => [
            "theory/mfa.md",
        ],
        "API reference" => "api.md",
    ],
    warnonly = true,
    checkdocs = :exports,
    doctest = false,
)
