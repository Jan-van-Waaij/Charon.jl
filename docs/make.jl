using Pkg
Pkg.instantiate()

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
        "Index" => "allfunctions.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "github.com/Jan-van-Waaij/Charon.jl.git"
    #branch = "gh-pages",
    #devbranch = "main",
    #push_preview = true,
    #make = "make.jl"
)=#

# julia -e 'using LiveServer; serve(dir="docs/build")'    

# updating website:
# git checkout gh-pages
# remove all files, except for docs/ and README.md
# remove all files in docs/build 
# git checkout main 
# make changes. 
# empty docs/build
# build website
# commit
# git checkout gh-pages
# cp -r docs/build/* . 
# commit 
# git push origin gh-pages 