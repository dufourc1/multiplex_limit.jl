using multiplex_limit
using Documenter

DocMeta.setdocmeta!(multiplex_limit, :DocTestSetup, :(using multiplex_limit); recursive=true)

makedocs(;
    modules=[multiplex_limit],
    authors="Charles Dufour",
    sitename="multiplex_limit.jl",
    format=Documenter.HTML(;
        canonical="https://dufourc1.github.io/multiplex_limit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dufourc1/multiplex_limit.jl",
    devbranch="main",
)
