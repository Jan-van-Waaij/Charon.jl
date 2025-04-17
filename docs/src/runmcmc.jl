using Pkg 

Pkg.activate(; temp=true)
Pkg.add(["Charon", "Distributions", "CSV", "DataFrames"])

using Charon, Distributions, CSV, DataFrames 

# specify the number of chains and the number of samples. 
nchains = 4 
nsteps = 100_000

# specify the prior. 
prioronτCτA = Product([Uniform(), Uniform()]) # uniform product prior [0,1]x[0,1] on (τC, τA).
prioronn = DiscreteUniform(1, 10) # discrete uniform prior on {1, 2, ..., 10}.
prioronϵ = Uniform(0, 0.5) # uniform prior on interval[0, 0.5].

if length(ARGS) == 2 
    dicefile = ARGS[1]
    resultsfile = ARGS[2]

    if !isfile(dicefile)
        println("No file found at $dicefile.")
        exit(1)
    end 

    chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, dicefile)
elseif length(ARGS) == 3 
    basecountfile = ARGS[1]
    freqfile = ARGS[2]
    resultsfile = ARGS[3]
    if !isfile(basecountfile)
        println("No base count file found at $basecountfile.")
        exit(2)
    end
    if !isfile(freqfile)
        println("No frequency file found at $freqfile.")
        exit(3)
    end
    chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, basecountfile, freqfile)
else 
    println("You should provide either two or three arguments: if you provide two arguments, the first argument should be a path to the dice file and second argument a path to the results. If you provide three arguments, the first argument should be a path to the base count file, the second argument should be a path to the frequency file, and the third argument should be path to the output file.")
    exit(4)
end 
    
results = unpackposterior(chains)

CSV.write(resultsfile, results)

