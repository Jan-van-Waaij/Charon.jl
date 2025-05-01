# Reference

Charon has three [exported](https://docs.julialang.org/en/v1/manual/modules/#Export-lists) functions, [MCMCsampler](#mcmcsampler), [exactposterior](#exactposterior) and [unpackposterior](#unpackposterior),  that can be used after typing 
```julia
using Charon 
```
```@docs
Charon.Charon
```


## MCMCsampler 
This is the MCMC sampler. It has seven methods.

Data formats:

You can input the data in four formats. 
1. With two (opened) (gzipped) files: a base count file, which is a CSV file where the first column are the number of derived reads, and the second column is the coverage, and a frequency file, which is a CSV file with one column, containing the frequencies. The data at line i in the base count file corresponds to the same locus as the data on line i of the frequency file.   
2. As an (opened) (gzipped) DICE file, or in the form of a DataFrame, also in DICE format. So the first column is the number of ancestral reads, the second column is the number of derived reads, the third column is the frequency in the anchor population, and the fourth column is the count of the number of loci where this combination of three numbers occur. 
3. Or, by providing three vectors: `coverages`, `derivedreads`, `frequencies`, of length equal to the number of SNPs, where at locus `i`, there are `derivedreads[i]` derived reads, `coverages[i]` coverage and `frequencies[i]` frequency in the anchor population. 
4. The third format is given with four vectors: `coverages`, `derivedreads`, `frequencies`, `counts`. This means that there are `counts[i]` loci with `coverages[i]` coverage, `derivedreads[i]` derived reads and frequency `frequencies[i]` in the anchor population.

If you provide data in formats 1, 2, or 3, then the program will automatically convert it to format 4. 

Parameters:
* `nchains`, number of chains, which is a positive integer. If you want to run all your chains in parallel, start julia with number of threads equal to `nchains`. 
* `nsteps` number of samples per chain, which is a positive integer. 
* `prioronn` the prior on n, specified as a subtype of `DiscreteUnivariateDistribution` of the Distributions Julia package. Our implementation requires that `prioronn` has support on {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, because otherwise rounding errors will accumulate too much. 
* `prioronτCτA` is the prior on (τC, τA), which allows for correlation between τC and τA. Its type is a subtype of ContinuousMultivariateDistribution. It should have support contained in [0,∞)x[0,∞). 
* `prioronϵ` is the prior on ϵ. It should have support in [0, 0.5). It is a subtype of `ContinuousUnivariateDistribution` in the Distributions package. 
* `coverages` a vector with coverages = ancestral reads + derived reads. Is a subtype of `AbstractVector{<:Integer}`. All coverages should be non-negative integers, and at least one should be positive. 
* `derivedreads` a vector of derived reads. Is a subtype of `AbstractVector{<:Integer}`. All elements of the vector should be non-negative integers.
* `frequencies` a vector of frequencies. Is a subtype of AbstractVector{<:Real}. Each frequency is between 0.0 and 1.0. At least one frequency should be strictly between 0.0 and 1.0 with corresponding positive coverage. 
* `counts`, all elements should be non-negative. For each index, `counts[index]` indicates how many loci there are with `derivedreads[index]` derived reads, coverage `coverages[index]` and frequency `frequencies[index]`.  
* `df` a DataFrame from the DataFrames package in the DICE-2 format. So the first column should be the number of ancestral reads, the second column the number of derived reads, the third column the frequencies in the anchor population, and the fourth column the counts of how many times this particular combination of ancestral reads, derived reads and frequency occurs. 
* `dicefile` is either an opened (gzipped) DICE file, or a path to a (gzipped) DICE file. 

[Keyword parameters](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments). Keyword parameters should be given as keyword=value to the function, in case you want to set another value then the default. 
* `messages` is an integer. If `messages` is non-positive, no message will be printed. If `messages` is positive, every `messages` steps a message will be printed with the progress of the sampler. The default value is `nsteps÷100`, so every 1% progress a message is printed.  
* `scalingmessage` is either `true` (default) or `false`. If true, a message will be printed when the scaling constant changes. 
* `header` is `nothing`, `true` (default) or `false`. Has the dicefile a header? If nothing, the software tries to determine whether the dicefile has a header. This works only when you provide a path to a file.

The output is a vector with nchains items. Each item represents a chain. Each item is a tuple consisting of six columns, as described [here](@ref output-mcmc-sampler).

```@docs
MCMCsampler
```

## unpackposterior 

This function is used to build the unconditional posterior from the MCMC samples conditioned on n, as described in the paper. It has two methods. 

Arguments:
* `nsteps` is the number of MCMC samples, a positive integer. 
* `chains` this the tuple that is the output of `MCMCsampler`. 
```@docs
unpackposterior
```

## exactposterior 

`exactposterior` is a function to calculate the posterior up to a fixed constant, only depending on the data, but not on the parameters. You can use this function for maximum posterior estimation. If you use uniform priors, you can use this function for maximum likelihood estimation. `MCMCsampler` uses this function to find a good starting point for the sampler. It has two methods. 

The posterior is calculated for each combination of parameters (n, τC, τA, ϵ) with n in `nrange`, τC in `τCrange`, τA in `τArange` and ϵ in `ϵrange`. So make sure that `length(nrange)*length(τCrange)*length(τArange)*length(ϵrange)` is not too large, as otherwise it will take a very long time and you might run out of memory. 

Parameters:
* `nrange` vector of n values. Subtype of AbstractVector{<:Integer}. Should be a subset of {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. 
* `τCrange` vector of τC values. Subtype of AbstractVector{<:Real}. All values should be non-negative. 
* `τArange` vector of τA values. Subtype of AbstractVector{<:Real}. All values should be non-negative. 
* `ϵrange` vector of ϵ values. Is a subtype of AbstractVector{<:Real}. All values should be non-negative and smaller than 0.5. 
* `coverages` a vector with coverages = ancestral reads + derived reads. Is a subtype of `AbstractVector{<:Integer}`. All coverages should be non-negative integers, and at least one should be positive.
* `uniquecoverages` should be equal to `unique(coverages)`.  
* `derivedreads` a vector of derived reads. Is a subtype of `AbstractVector{<:Integer}`. All elements of the vector should be non-negative integers.
* `frequencies` a vector of frequencies. Is a subtype of AbstractVector{<:Real}. Each frequency is between 0.0 and 1.0. At least one frequency should be strictly between 0.0 and 1.0 with corresponding positive coverage. 
* `counts`, all elements should be non-negative. For each index, `counts[index]` indicates how many loci there are with `derivedreads[index]` derived reads, coverage `coverages[index]` and frequency `frequencies[index]`. 

Keyword argument. 
* `messages` is an integer. If `messages` is non-positive, no message will be printed. If `messages` is positive, every `messages` steps a message will be printed with the progress of the calculations. 

The output are 5 vectors: `ns, τCs, τAs, ϵs, logliks`, of each of length `length(n)*length(τCrange)*length(τArange)*length(ϵrange)`, where `logliks[index]` is the log likelihood, up to an additive constant, with parameters `ns[index]`, `τCs[index]`, `τAs[index]` and `ϵs[index]`. The additive constant only depends on the data, but not on the parameters. 

### Maximum posterior estimation (MAP estimator)

Suppose you have results
```julia
ns, τCs, τAs, ϵs, logliks = exactposterior(args...) 
```
where `args` are your arguments (data and priors, etc.). Then you can calculate then 
```julia
i_max = argmax(logliks)
```
is an index where the posterior is at its maximum. 
So 
```
ns[i_max], τCs[i_max], τAs[i_max], ϵs[i_max]
```
is the MAP estimator. 

```@docs 
exactposterior
```

## Charon EigenExpansion type 

This type is not exported, but can be used after `using Charon: EigenExpansion`, or via `Charon.EigenExpansion`. It represents a matrix decomposition $M=P*D*P^{-1}$, where $P$ is an invertible matrix with inverse $P^{-1}$, and $D$ is a diagonal matrix. It is used to efficiently calculate the exponent of a matrix. 
```@docs
Charon.EigenExpansion
```

## Internal, non-exported functions 

The following functions are not exported, and only available via `Charon.functionname`, or `using Charon: functionname`.

```@docs
Charon.calcloglik
Charon.calcmatrix
Charon.filtervectorsandapplycountmap
Charon.logprobderivedreads!
Charon.makeq
Charon.makeqfixedn
Charon.makeQ
Charon.makeQꜜ
Charon.preparedata
Charon.readcsvfile
Charon.updateq!
```

## Extended Julia base functions. 

I extended several Julia base functions to my custom EigenExpansion type. 
```@docs
Base.exp
Base.show
==
```