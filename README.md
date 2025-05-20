# Charon.jl: Estimation of population parameters from environmental DNA.

Charon.jl is a Julia package for estimating population parameters from sediment DNA. I have tested this package extensively on all minor 1.X Julia versions (1.0-1.11). However, it is always better to use a recent version of Julia. If you have any problems, please reach out to me. 

## Documentation

[jan-van-waaij.github.io/Charon.jl/](https://jan-van-waaij.github.io/Charon.jl/)

## Install Julia (Juliaup). 
Go to [julialang.org](https://julialang.org/downloads/) and follow the installation instructions for your platform.


## Start Julia

Type `julia` in your terminal to start Julia. 


## Install Charon
### Activate an environment. 
It is a good idea to start a new environment for each project [to reduce the chance of a version conflict](https://pkgdocs.julialang.org/v1/managing-packages/#conflicts). 

Press ] to activate the [package manager mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode). 
Type 

```julia
activate EnvironmentName
```

where EnvironmentName is the name you want to give to your environment. 
Next, install Charon to your environment: 

```julia-repl
add Charon
```

If you want, you can install more packages in your environment. For instance, most likely, you will need CSV and DataFrames:

```julia
add CSV DataFrames.
```
Other valuable packages are GZip and CodecZlib for working with zipped files, StatsBase and Distributions for working with probabilities, LinearAlgebra for matrices, vectors, and related linear algebra functions, and Distributed for distributed computing.

After adding at least one package to the environment, Julia will create a new folder, `EnvironmentName`, in your current working directory. Alternatively, you could set an existing folder as the environment:
```julia
activate path/to/the/folder
``` 
After you add at least one package, Julia will create the files Project.toml and Manifest.toml in that particular folder. Note that this action does not change the working environment. 

[Read more about Julia environments.](https://pkgdocs.julialang.org/v1/environments/)

## Data format

Our program can work with genetic data in the [DICE-2 format](https://github.com/grenaud/dice?tab=readme-ov-file#2-pop-method-input-data-format). 

A CSV file in DICE-2 format has four columns. In this order: number of ancestral reads, number of derived reads, the frequency of the anchor population, and in how many loci this combination of reads and frequency occurs. So, the first two columns contain non-negative integers, the third column is a real number between 0.0 and 1.0, and the last column is a positive integer. 

## Using the sampler. 

### Convenience script.
[Click here for an example script.](runmcmc.jl) called runmcmc.jl. Safe this on your computer. This script sets uniform priors on `τC`, `τA`, `ϵ` and `n`, and four chains with 100'000 samples. The script saves the output as a CSV file. You can use your favourite software to analyse it. It works as follows (assuming Julia is in your path, and you use Julia 1.5 or higher):
```julia
julia --threads=4 path/to/runmcmc.jl path/to/dicefile path/to/outputfile
```
where the output is saved in `outputfile`. `dicefile` might be a CSV file or a zipped CSV file. 

### Steps to take. 
In this paragraph, I describe the steps for manually running the sampler. 
1. Save your data in the DICE-2 format as a (zipped) CSV file, [as described here](#data-format). 
2. Start a Julia session with, for example, four threads if you want to sample four chains parallel.
```
julia --threads=4 
```
3. We must activate the environment we [created before](#activate-an-environment). You can check whether you are in the correct environment by running 
```julia
status 
```
in the package modus. If not, let's activate the environment that we created before by running
```julia
activate path/to/EnvironmentName 
```
in the package modus. Run
```julia
status
```
again to check whether the packages Charon, DataFrames, Distributions and CSV have been installed. If any of them are lacking, please add them to your environment in the package modus via 
```julia
add PackageName
```
Alternatively, you can start your Julia session in the right environment with 
```julia 
julia --threads=4 --project=path/to/environment 
```
Make sure all the required packages are in the environment. 

4. Load the packages. 
In the REPL, run 
```julia
# load the packages. 
using Charon, Distributions, CSV, DataFrames
```
5. Let us specify the path to the file where our data is stored. 
```julia
dicefile = "path/to/dicefile.csv" # UNIX-systems
dicefile = "path\\to\\dicefile.csv" # Windows-sytems. 
dicefile = joinpath("path", "to", "dicefile.csv") # works on all systems 😄. 
```
where "path/to/dicefile.csv" is the relative path to dicefile.csv from your working directory. `dicefile` might also be a zipped csv file. You can find the working directory as follows: Press ";" to come in the [shell mode](https://docs.julialang.org/en/v1/stdlib/REPL/#man-shell-mode).  
```julia-repl
shell> pwd
```
You can use `cd` to change your working directory:
```
shell> cd path/to/dicefilefolder
```
Press backspace to come back to the normal REPL. 

6. Let us specify a prior on n, (τC, τA) and ϵ. For example, you could use a uniform prior for all three:
```julia
using Distributions # Julia package for probability distributions. 
prioronτCτA = Product([Uniform(), Uniform()]) # uniform product prior [0,1]x[0,1] on (τC, τA).
prioronn = DiscreteUniform(1, 10) # discrete uniform prior on {1, 2, ..., 10}.
prioronϵ = Uniform(0, 0.5) # uniform prior on interval[0, 0.5].
```
7. Specify how many MCMC samples you want and how many chains you want to sample. For example, if you want four chains with each 100'000 samples, you can specify 
```julia
nchains = 4 
nsteps = 100_000
```

8. We can now run our sampler, which might take a few hours. 
```julia
using Charon 
chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, dicefile)
```
The output of this function 
`chains` is a vector of tuples. Each tuple represents an MCMC chain. Each tuple consists of six arrays. For instance, if we want to analyse the first chain, we get 
```
nsample, τCsample, τAsample, ϵsample, accepted, logjointprob = chains[1]
```
`nsample` is the MCMC chain for the number of individuals. `τCsample`, `τAsample`, and `ϵsample` are (in this example) 100'000x10 matrices, where the k-th column is the MCMC chain of `τC`, `τA`, and `ϵ`, respectively, with the number of individuals conditioned on k. `accepted` and `logjointprob` are 100'000x11 matrices, where the first column of each matrix indicates whether the mcmc chain for n is accepted and the logjoint probability, respectively. The k+1-th column is the acceptance vector / log joint probability vector of the MCMC conditioned on k individuals.

You can obtain the unconditioned sample from the posterior as follows 
```julia
using Charon 
results = unpackposterior(nsteps, chains)
```
`results` is a DataFrame with a sample from the posterior. It has 7 columns: 
1. nsample sample of the number of individuals
2. τCsample sample of τC values
3. τAsample sample of τA values
4. ϵsample sample of ϵ values 
5. accepted whether the proposal was accepted. If not, the row is equal to the previous row. 
6. logjointprob  the log joint probability of the likelihood and prior, which is the density of posterior up to a constant that only depends on the data, but not on the parameters. 
7. chainid the id of the chain, a number in 1, ..., nchains. 

[]()

9. You can save the data frame as a CSV file as follows:
```julia
using CSV 
resultscsvfile = "path/to/results.csv" # place on your hard disk
# where you want to store your csv file. 
CSV.write(resultscsvfile, results)
```
You can use the DataFrame and the CSV file for further analysis.

## Reference 
Charon has three exported functions, [MCMCsampler](#mcmcsampler), [exactposterior](#exactposterior) and [unpackposterior](#unpackposterior),  that can be used after typing 
```julia
using Charon 
```
### MCMCsampler 
This is the MCMC sampler. It has five methods. 
```julia 
    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, counts::AbstractVector{<:Integer}; messages::Integer=nsteps÷100, scalingmessages::Bool=true)

    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}; messages::Integer=nsteps÷100, scalingmessages::Bool=true)

    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, df::AbstractDataFrame; messages::Integer=nsteps÷100, scalingmessages::Bool=true)

    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, dicefile::IO; messages::Integer=nsteps÷100, scalingmessages::Bool=true, header::Bool=true)

    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, dicefile::AbstractString; messages::Integer=nsteps÷100, scalingmessages::Bool=true, header::Union{Nothing, Bool}=nothing)
```
Data formats:

You can input the data in three formats. 
1. As an (opened) (zipped) DICE file, or in the form of a DataFrame, also in DICE format. So the first column is the number of ancestral reads, the second column is the number of derived reads, the third column is the frequency in the anchor population, and the fourth column is the count of the number of loci where this combination of three numbers occur. 
2. Or, by providing three vectors: `coverages`, `derivedreads`, `frequencies`, of length equal to the number of SNPs, where at loci `i`, there are `derivedreads[i]` derived reads, `coverages[i]` coverage and `frequencies[i]` frequency in the anchor population. 
3. The third format is given with four vectors: `coverages`, `derivedreads`, `frequencies`, `counts`. This means that there are `counts[i]` loci with `coverages[i]` coverage, `derivedreads[i]` derived reads and frequency `frequencies[i]` in the anchor population.

If you provide data in formats 1 or 2 then the program will automatically convert it to format 3. 

Parameters:
* `nchains`, number of chains, which is a positive integer. If you want to run all your chains in parallel, start julia with number of threads equal or higher to `nchains`. 
* `nsteps` number of samples per chain, which is a positive integer. 
* `prioronn` the prior on n, specified as a subtype of `DiscreteUnivariateDistribution` of the Distributions Julia package. Our implementation requires that `prioronn` has support on {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, because otherwise rounding errors will accumulate too much. 
* `prioronτCτA` is the prior on (τC, τA), which allows for correlation between τC and τA. Its type is a subtype of ContinuousMultivariateDistribution. It should have support contained in [0,∞)x[0,∞). 
* `prioronϵ` is the prior on ϵ. It should have support in [0, 0.5). It is a subtype of `ContinuousUnivariateDistribution` in the Distributions package. 
* `coverages` a vector with coverages = ancestral reads + derived reads. Is a subtype of `AbstractVector{<:Integer}`. All coverages should be non-negative integers, and at least one should be positive. 
* `derivedreads` a vector of derived reads. Is a subtype of `AbstractVector{<:Integer}`. All elements of the vector should be non-negative integers.
* `frequencies` a vector of frequencies. Is a subtype of AbstractVector{<:Real}. Each frequency is between 0.0 and 1.0. At least one frequency should be strictly between 0.0 and 1.0 with corresponding positive coverage. 
* `counts`, all elements should be non-negative. For each index, `counts[index]` indicates how many loci there are with `derivedreads[index]` derived reads, coverage `coverages[index]` and frequency `frequencies[index]`. If `counts` is not provided, it will be calculated from `derivedreads`, `coverages` and `frequencies`. 

[Keyword parameters](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments). Keyword parameters should be given as keyword=value to the function, in case you want to set another value then the default. 
* `messages` is an integer. If `messages` is non-positive, no message will be printed. If `messages` is positive, every `messages` steps a message will be printed with the progress of the sampler. The default value is `nsteps÷100`, so every 1% progress a message is printed.  
* `scalingmessage` is either `true` (default) or `false`. If true, a message will be printed when the scaling constant changes. 
* `df` a DataFrame from the DataFrames package in the DICE-2 format. So the first column should be the number of ancestral reads, the second column the number of derived reads, the third column the frequencies in the anchor population, and the fourth column the counts of how many times this particular combination of ancestral reads, derived reads and frequency occurs. 
* `dicefile` is either an opened DICE file, or a path to a DICE file. 
* `header` is `nothing`  `true` (default) or `false`. Has the dicefile a header? If nothing, the software tries to determine whether the dicefile has a header. This works only when you provide a path to a file.


The output is a vector with nchains items. Each item represents a chain. Each item is a tuple consisting of six columns, as described in step 8 of [Steps to take](#steps-to-take).

### exactposterior 

`exactposterior` is a function to calculate the posterior up to a fixed constant, only depending on the data, but not on the parameters. You can use this function for maximum likelihood estimation. `MCMCsampler` uses this function to find a good starting point for the sampler. It has two methods: 
```julia 
    exactposterior(nrange::AbstractVector{<:Integer}, τCrange::AbstractVector{<:Real}, τArange::AbstractVector{<:Real}, ϵrange::AbstractVector{<:Real}, coverages::AbstractVector{<:Integer}, uniquecoverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, counts::AbstractVector{<:Integer}; messages::Integer=0)
    exactposterior(nrange::AbstractVector{<:Integer}, τCrange::AbstractVector{<:Real}, τArange::AbstractVector{<:Real}, ϵrange::AbstractVector{<:Real}, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}; messages::Integer=length(n)*length(τCrange)*length(τArange)*length(ϵrange)÷100)
```
The posterior is calculated for each combination of parameters (n, τC, τA, ϵ) with n in `nrange`, τC in `τCrange`, τA in `τArange` and ϵ in `ϵrange`. So make sure that `length(nrange)*length(τCrange)*length(τArange)*length(ϵrange)` is not too large, as otherwise it will take a very long time and you might run out of memory. 

Parameters:
* `nrange` vector of n values. Subtype of AbstractVector{<:Integer}. Should be a subset of {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. 
* `τCrange` vector of τC values. Subtype of AbstractVector{<:Real}. All values should be non-negative. 
* `τArange` vector of τA values. Subtype of AbstractVector{<:Real}. All values should be non-negative. 
* `ϵrange` vector of ϵ values. Is a subtype of AbstractVector{<:Real}. 
* `coverages` a vector with coverages = ancestral reads + derived reads. Is a subtype of `AbstractVector{<:Integer}`. All coverages should be non-negative integers, and at least one should be positive.
* `uniquecoverages` should be equal to `unique(coverages)`.  
* `derivedreads` a vector of derived reads. Is a subtype of `AbstractVector{<:Integer}`. All elements of the vector should be non-negative integers.
* `frequencies` a vector of frequencies. Is a subtype of AbstractVector{<:Real}. Each frequency is between 0.0 and 1.0. At least one frequency should be strictly between 0.0 and 1.0 with corresponding positive coverage. 
* `counts`, all elements should be non-negative. For each index, `counts[index]` indicates how many loci there are with `derivedreads[index]` derived reads, coverage `coverages[index]` and frequency `frequencies[index]`. If `counts` is not provided, it will be calculated from `derivedreads`, `coverages` and `frequencies`.

Keyword argument. 
* `messages` is an integer. If `messages` is non-positive, no message will be printed. If `messages` is positive, every `messages` steps a message will be printed with the progress of the sampler. The default value is `length(n)*length(τCrange)*length(τArange)*length(ϵrange)÷100`, so every 1% progress a message is printed.

The output are 5 vectors: `ns, τCs, τAs, ϵs, logliks`, of each of length `length(n)*length(τCrange)*length(τArange)*length(ϵrange)`, where `logliks[index]` is the log likelihood, up to an additive constant, with parameters `ns[index]`, `τCs[index]`, `τAs[index]` and `ϵs[index]`. The additive constant only depends on the data, but not on the parameters. 

### unpackposterior 

This function is used to build the unconditional posterior from the MCMC samples conditioned on n, as described in the paper. It has one method with two arguments:

```julia
unpackposterior(nsteps::Integer, chains::AbstractVector{<:Tuple{AbstractVector{<:Integer}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractMatrix{Bool}, AbstractMatrix{<:Real}}})
```
Arguments:
* `nsteps` is the number of MCMC samples, a positive integer. 
* `chains` this the tuple that is the output of `MCMCsampler`. 
