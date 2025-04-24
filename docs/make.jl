using Documenter
using Charon

makedocs(
    sitename = "Charon.jl",
    format = Documenter.HTML(; assets = ["assets/favicon.ico"]),
    modules = [Charon],
    pages = [
        "Introduction" => "index.md",
        "Table of contents" => "contents.md",
        "Getting started" => "guide.md",
        "Reference" => "reference.md",
        "Index" => "siteindex.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/Jan-van-Waaij/Charon.jl.git"
)
