using Pkg 

Pkg.activate("Runmcmc")
Pkg.add(["Charon", "Distributions", "CSV", "DataFrames"])

using Charon, Distributions, CSV, DataFrames 

if length(ARGS) != 2 
    println("You should provide two arguments: (1) path to the dice file and (2) path to the results.")
    exit(1)
end 

dicefile = ARGS[1]
resultsfile = ARGS[2]

if !isfile(dicefile)
    println("No file found at $dicefile.")
    exit(2)
end 

# specify the number of chains and the number of samples. 
nchains = 4 
nsteps = 100_000

# specify the prior. 
prioronτCτA = Product([Uniform(), Uniform()]) # uniform product prior [0,1]x[0,1] on (τC, τA).
prioronn = DiscreteUniform(1, 10) # discrete uniform prior on {1, 2, ..., 10}.
prioronϵ = Uniform(0, 0.5) # uniform prior on interval[0, 0.5].


chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, dicefile)

results = unpackposterior(nsteps, chains)

CSV.write(resultsfile, results)

