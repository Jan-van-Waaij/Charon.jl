using Test # the testing package   
using Charon # the package to be tested
using Distributions # for probability distributions
using LinearAlgebra # for Tridiagonal objects 
using StatsBase # for countmap 
using Random # for setting the seed
using CSV # to test readcsvfile 
using GZip # to test readcsvfile 
using DataFrames # to test unpackposterior

using Charon: EigenExpansion, makeQ, makeQꜜ, calcmatrix, logprobderivedreads!, preparedata, filtervectorsandapplycountmap, makeqfixedn, makeq, updateq!, readcsvfile

@testset "Charon" begin
    @testset "EigenExpansion and exp" begin
        x = rand(2)
        y = rand(3)
        A = Tridiagonal(x, y, x)
        F = EigenExpansion(A)
        @test A ≈ F.P*F.D*F.Pinv
        @test exp(Matrix(A)) ≈ exp(1.0, F)
        @test typeof(exp(1.0, F)) == typeof(F.P)

        A = rand(2,2)
        B = A*A'
        F = EigenExpansion(B)
        @test typeof(exp(2.0, F)) == typeof(F.P)
        @test B ≈ F.P*F.D*F.Pinv
        @test exp(B) ≈ exp(1.0, F)
        @test exp(2.0*B) ≈ exp(2.0, F)
    end

    @testset "==" begin
        A = rand(2,2)
        B = A*A'
        C = copy(B)
        FB = EigenExpansion(B)
        FC = EigenExpansion(C)
        @test FB == FC 
        x = rand(3)
        y = rand(4)
        T = Tridiagonal(x, y, x)
        U = deepcopy(T)
        MT = Matrix(T)
        MU = Matrix(U)
        FT = EigenExpansion(T)
        FU = EigenExpansion(U)
        FMT = EigenExpansion(MT)
        FMU = EigenExpansion(MU)
        @test FT == FU
        @test FMT == FMU
    end

    @testset "makeQꜜ" begin
        @test makeQꜜ(1) == [0 3 0
                            0 -2 1
                            0 1 -2]
        
        for n in 1:10
            Qꜜ = makeQꜜ(n)
            for k in 0:2n 
                for L in 0:2n 
                    if L == k-1 
                        @test Qꜜ[k+1, L+1] == 0.5*k*(k-1)
                    elseif L == k 
                        @test Qꜜ[k+1, L+1] == -k*(2n-k+1)
                    elseif L == k+1 
                        @test Qꜜ[k+1, L+1] == 0.5*(2n-k+1)*(2n-k)
                    else 
                        @test Qꜜ[k+1, L+1] == 0
                    end # if-elseif-else block. 
                end # L 
            end # k 
        end # n 
    end # testset makeQꜜ

    @testset "makeQ" begin
        @test makeQ(1) == [0 1 0
                           0 -1 0
                           0 1 0]

        for n in 1:10
            Q = makeQ(n)
            for k in 0:2n 
                for L in 0:2n 
                    if L == k-1 
                        @test Q[k+1, L+1] == 0.5*k*(k-1)
                    elseif L == k 
                        @test Q[k+1, L+1] == -k*(2n-k)
                    elseif L == k+1 
                        @test Q[k+1, L+1] == 0.5*(2n-k)*(2n-k-1)
                    else 
                        @test Q[k+1, L+1] == 0
                    end # if-elseif-else block 
                end # L 
            end # k 
        end  # n 
    end # testset makeQ 

    @testset "calcmatrix" begin
        for n in 1:10
            Q = makeQ(n)
            Qꜜ = makeQꜜ(n)
            FQ = EigenExpansion(Q)
            FQꜜ = EigenExpansion(Qꜜ)
            @test calcmatrix(0.0, 0.0, FQꜜ, FQ) ≈ I
            τC = rand()
            τA = rand()
            M = calcmatrix(τC, τA, FQꜜ, FQ)
            @test typeof(FQ.P) == typeof(M)
            @test M == exp(τA, FQ)*exp(τC, FQꜜ)
            M = calcmatrix(0.0, τA, FQꜜ, FQ)
            @test M ≈ exp(τA, FQ)
            M = calcmatrix(τC, 0.0, FQꜜ, FQ)
            @test M ≈ exp(τC, FQꜜ)
        end 
    end

    @testset "preparedata" begin
        y = 0.1:0.1:0.9
        for n in 1:10
            t = preparedata(n, y)
            @test length(t) == 4 
            @test t isa NamedTuple 
            Qꜜ, Q, binomialcoefficients, hvals = t
            @test Qꜜ == EigenExpansion(makeQꜜ(n))
            @test Q == EigenExpansion(makeQ(n))
            @test size(hvals) == (2n+1, length(y))
            for k in 0:2n 
                for yindex in eachindex(y) 
                    @test hvals[k+1, yindex] == y[yindex]^k*(1-y[yindex])^(2n-k)
                end 
            end 
            @test binomialcoefficients == binomial.(2n, 0:2n)

            Qꜜ = t.Qꜜ
            Q = t.Q 
            binomialcoefficients = t.binomialcoefficients
            hvals = t.hvals
            @test Q == EigenExpansion(makeQ(n))
            @test Qꜜ == EigenExpansion(makeQꜜ(n))
            @test size(hvals) == (2n+1, length(y))
            for k in 0:2n 
                for yindex in eachindex(y) 
                    @test hvals[k+1, yindex] == y[yindex]^k*(1-y[yindex])^(2n-k)
                end 
            end 
            @test binomialcoefficients == binomial.(2n, 0:2n)
        end 
    end

    @testset "logprobderivedreads!" begin
        # when τC=τA=0, then logprobderivedreads! calculates the log probability 
        # of the binomial distribution with parameters R and (1-ϵ)*k/2n+ϵ*k/(2n), where k itself is binomially distributed with parameters 2n and y. 
       
        # test for equality with τC = τA = 0.0, with non-constant R version. 
        for n in 1:10
            for ϵ in 0.1:0.1:0.4
                expsize = 100_000 
                Random.seed!(1987)
                y = rand(0.1:0.1:0.9, expsize)
                Random.seed!(1987)
                coverages = rand(1:10, expsize)
                uniquecoverages = unique(coverages)
                Random.seed!(1987)
                reads = [rand(0:coverage) for coverage in coverages]
                @assert all(0 .≤ reads .≤ coverages)
                uniquereadsandcoverages = unique(zip(coverages, reads))
                q = Dict((R, d) => Vector{Float64}(undef, 2n+1) for (R, d) in uniquereadsandcoverages)
                cm = countmap(zip(y, reads, coverages))
                y = [key[1] for key in keys(cm)]
                reads = [key[2] for key in keys(cm)]
                coverages = [key[3] for key in keys(cm)]
                counts = collect(values(cm))
                Qꜜ, Q, binomialcoefficients, hvals = preparedata(n, y)
                τC = 0.3 
                τA = 0.2 
                @test logprobderivedreads!(q, n, -0.1, τA, ϵ, coverages, uniquecoverages, reads, counts, Qꜜ, Q, binomialcoefficients, hvals) == -Inf 
                @test logprobderivedreads!(q, n, τC, -0.1, ϵ, coverages, uniquecoverages, reads, counts,  Qꜜ, Q, binomialcoefficients, hvals) == -Inf 
                @test logprobderivedreads!(q, n, τC, τA, -0.1, coverages, uniquecoverages, reads, counts, Qꜜ, Q, binomialcoefficients, hvals) == -Inf 
                @test logprobderivedreads!(q, n, τC, τA, 1.1, coverages, uniquecoverages, reads, counts,  Qꜜ, Q, binomialcoefficients, hvals) == -Inf 
                logpdffunction = logprobderivedreads!(q, n, 0.0, 0.0, ϵ, coverages, uniquecoverages, reads, counts,  Qꜜ, Q, binomialcoefficients, hvals)
                #@assert logpdffunction > -Inf 
                logpdftheoretical = 0.0
                for index in eachindex(reads, coverages, counts, y)
                    d = reads[index]
                    R = coverages[index]
                    distk = Binomial(2n, y[index])
                    prob = 0.0 
                    for k in 0:2n 
                        distgivenk = Binomial(R, (1-ϵ)*k/(2n)+ϵ*(2n-k)/(2n))
                        prob += pdf(distgivenk, d)*pdf(distk, k)
                    end 
                    logpdftheoretical += counts[index] * log(prob)
                end 
                @test logpdffunction ≈ logpdftheoretical 
            end # ϵ  
        end # n 
    end # testset logprobderivedreads!

    @testset "filtervectorsandapplycountmap" begin
        y = vcat([1, 1, 2, 3, 4], rand(10))
        d = vcat([1, 1, 2, 3, 4], rand([1,2,3], 10))
        R = vcat([3, 3, 2, 3, 4], rand(4:6, 10 ))
        allowedindices = vcat(trues(5), falses(10))
        R, d, y, counts = filtervectorsandapplycountmap(R, d, y, allowedindices)
        @test Set(y) == Set([1, 2, 3, 4])
        @test Set(d) == Set([1, 2, 3, 4])
        @test Set(R) == Set([2, 3, 4])
        @test Set(counts) == Set([2, 1, 1, 1])
    end

    @testset "makeqfixedn" begin 
        for n in 1:10 
            d = rand(1:10, 1000)
            R = [rand(d[i]:10) for i in 1:1000]
            uniqueRandd = unique(zip(R, d))
            q = makeqfixedn(n, R, d)
            @test length(q) == length(uniqueRandd)
            for value in values(q) 
                @test length(value) == 2n+1
            end
            q = makeqfixedn(n, uniqueRandd)
            @test length(q) == length(uniqueRandd)
            for value in values(q) 
                @test length(value) == 2n+1
            end 
        end 
    end 

    @testset "makeq" begin
        for nmax in 1:10 
            d = rand(1:10, 1000)
            R = [rand(d[i]:10) for i in 1:1000]
            q = makeq(nmax, R, d)
            @test length(q) == nmax
            lendic = length(unique(zip(R, d)))
            for item in q 
                @test length(item) == lendic 
            end  
            for n in 1:nmax 
                for value in values(q[n]) 
                    @test length(value) == 2n+1
                end 
            end 
        end
    end

    @testset "updateq!" begin 
        for n in 1:10 
            dic = Dict((R, d) => Vector{Float64}(undef,2n+1) for R in 1:10 for d in 0:R)
            binomialcoefficients = binomial.(2n, 0:2n)
            updateq!(dic, n, 0.1, 1:10, binomialcoefficients)
            for R in 1:10
                for d in 0:R 
                    for k in 0:2n 
                        dist = Binomial(R, k*0.9/(2n)+(2n-k)*0.1/(2n))
                        @test dic[(R, d)][k+1] == binomialcoefficients[k+1] * pdf(dist, d)
                    end 
                end 
            end 
        end  
    end 

    @testset "calcloglik" begin 
    end 

    @testset "readcsvfile" begin 
        filewithheader = "testwithheader.csv"
        filewithoutheader = "testwithoutheader.csv"
        filewithheaderzipped = "testwithheader.csv.gz"
        filewithoutheaderzipped = "testwithoutheader.csv.gz"

        dfh = readcsvfile(filewithheader, nothing)
        dfw = readcsvfile(filewithoutheader, nothing)
        dfhz = readcsvfile(filewithheaderzipped, nothing)
        dfwz = readcsvfile(filewithoutheaderzipped, nothing)
        dfhk = readcsvfile(filewithheader, true)
        dfwk = readcsvfile(filewithoutheader, false)
        dfhzk = readcsvfile(filewithheaderzipped, true)
        dfwzk = readcsvfile(filewithoutheaderzipped, false)

        #@test Matrix(dfhz) == Matrix(dfwz)
        @test Matrix(dfh) == Matrix(dfw) == Matrix(dfhz) == Matrix(dfwz) == Matrix(dfhk) == Matrix(dfwk) == Matrix(dfhzk) == Matrix(dfwzk)
    end 

    @testset "unpackposterior" begin
        # unpackposterior(chains::Vector{Tuple}, nsteps::Integer)
        # chains is a vector of tuples.
        # each tuple consists of six arrays: a vector, three matrices, a bitmatrix and a matrix.  
        #=
        nsample = Vector{Int}(undef, nsteps)
        τCsample = Matrix{Float64}(undef, nsteps, nmax)
        τAsample = Matrix{Float64}(undef, nsteps, nmax)
        ϵsample = Matrix{Float64}(undef, nsteps, nmax)
        acceptedvec = BitMatrix(undef, nsteps, nmax+1) 
        =#
        nsteps = 100
        nmax = 10 
        nsample = repeat(1:10, 10)
        τCsample = Matrix{Float64}(undef, nsteps, nmax)
        logjointprob = Matrix{Float64}(undef, nsteps, nmax+1)
        for step in 1:nsteps 
            for n in 1:nmax 
                τCsample[step, n] = (step-1)*nmax+n 
            end 
        end  
        τAsample = -τCsample 
        ϵsample = τCsample .^ 2 
        acceptedvec = BitMatrix(undef, nsteps, nmax+1) 
        chains = [(nsample, τCsample, τAsample, ϵsample, acceptedvec, logjointprob)]
        df = unpackposterior(nsteps, chains)
        @test df.nsample == nsample 
        for index in 1:nsteps 
            @test df.τCsample[index] % nmax == df.nsample[index] % nmax 
            @test df.τAsample[index] % nmax == -df.nsample[index] % nmax 
        end 
        @test df.ϵsample == df.τCsample .^ 2
        @test df.accepted  == trues(nsteps)
    end

end # testset Charon