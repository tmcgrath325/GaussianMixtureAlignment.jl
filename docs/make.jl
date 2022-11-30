using GaussianMixtureAlignment
using Documenter

DocMeta.setdocmeta!(GaussianMixtureAlignment, :DocTestSetup, :(using GaussianMixtureAlignment); recursive=true)

makedocs(;
    modules=[GaussianMixtureAlignment],
    authors="Tom McGrath <tmcgrath325@gmail.com> and contributors",
    repo="https://github.com/tmcgrath325/GaussianMixtureAlignment.jl/blob/{commit}{path}#{line}",
    sitename="GaussianMixtureAlignment.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tmcgrath325.github.io/GaussianMixtureAlignment.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tmcgrath325/GaussianMixtureAlignment.jl",
    devbranch="master",
)
