@info "I am checking your input."
if !(2 ≤ length(ARGS) ≤ 3)
    println("You should provide either two or three arguments: if you provide two arguments, the first argument should be a path to the dice file and second argument a path to the results. If you provide three arguments, the first argument should be a path to the base count file, the second argument should be a path to the frequency file, and the third argument should be path to the output file.")
    exit(1)
end 

if ((VERSION < v"1.3") && ((ARGS[1][end-2:end] == ".gz") || (ARGS[2][end-2:end] == ".gz")))
    println("You are using an older version of Julia, with a GZipped file. Please unzip the files first, or use Julia 1.3 or higher.")
    exit(2)
end 

resultsfile = ARGS[end]

if length(ARGS) == 2 
    dicefile = ARGS[1]
    
    if !isfile(dicefile)
        println("No file found at $dicefile.")
        exit(3)
    end 
elseif length(ARGS) == 3 
    basecountfile = ARGS[1]
    freqfile = ARGS[2]
    if !isfile(basecountfile)
        println("No base count file found at $basecountfile.")
        exit(4)
    end
    if !isfile(freqfile)
        println("No frequency file found at $freqfile.")
        exit(5)
    end
end 

@info "I am activating a new environment."
using Pkg 
if VERSION ≥ v"1.5" 
    Pkg.activate(; temp=true)
else 
    Pkg.activate("Runmcmc")
end 

@info "I am adding packages to the new environment."
Pkg.add(["Charon", "Distributions", "CSV", "DataFrames"])

@info "I am loading the packages."
using Charon, Distributions, CSV, DataFrames 

@info "I am specifying the number of chains, the number of samples, and the prior."
# specify the number of chains and the number of samples. 
nchains = 4 
nsteps = 100_000

# specify the prior. 
prioronτCτA = product_distribution([Uniform(), Uniform()]) # uniform product prior [0,1]x[0,1] on (τC, τA).
prioronn = DiscreteUniform(1, 10) # discrete uniform prior on the set {1, 2, ..., 10}.
prioronϵ = Uniform(0, 0.5) # uniform prior on the interval [0, 0.5].

@info "I am running the mcmc sampler."
if length(ARGS) == 2 
    chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, dicefile; messages = 0, scalingmessages=false)
elseif length(ARGS) == 3 
    chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, basecountfile, freqfile; messages = 0, scalingmessages=false)
end 
    
@info "I am unpacking the posterior."
results = unpackposterior(chains)

@info "I am writing the results to a CSV file."
CSV.write(resultsfile, results)

@info "I am finished. I saved the results in $resultsfile."

