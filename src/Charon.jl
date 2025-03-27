"""
    module Charon

This package provides functions to sample from the posterior. 
"""
module Charon

import Base: ==, exp, show # redefine exp, == and show for EigenExpansion types. 
# This requires an import statement. Redefine show for customized printing of EigenExpansion types. 

# external packages that I use. 
using Distributions # for probability distributions
using StatsBase # for the countmap function
using LinearAlgebra # Needed for the identity matrix I and the different matrix types and functions. 
using StaticArrays # for SizedVector, they are generally faster 
# than the build-in Matrix types, because they store their size in their type.
using Base.Threads # for multithreading.
using DataFrames # for processing Dice files. 
using CSV # for processing CSV files.
      
# functions that you can use when you type 'using Charon'. 
export MCMCsampler # MCMC sampler.
export exactposterior # Calculate the posterior up to a fixed constant depending on the data, 
# but not depending on the parameters. Can be used for maximum likelihood estimation.
export unpackposterior # Constructs the unconditional posterior from the conditional chains. 

# Likelihood functions
# Tests written
"""
    EigenExpansion{Q<:AbstractMatrix{<:Real}, R<:Real, S<:AbstractVector{R}, T<:AbstractMatrix{<:Real}}

Representation of a matrix decomposition M=P*D*P^{-1}, where D is a diagonal matrix and P an invertible matrix with inverse Pinv. The fields are P, D and Pinv of types Q, Diagonal{R, S}, and T, respectively. 

This decomposition can be used for fast matrix exponentiation, using that exp(M) = P*exp(D)*Pinv, and the fact that the exponent of a diagonal matrix is formed by exponentiating its diagonal elements.
"""
struct EigenExpansion{Q<:AbstractMatrix{<:Real}, R<:Real, S<:AbstractVector{R}, T<:AbstractMatrix{<:Real}}
    P::Q
    D::Diagonal{R, S}
    Pinv::T
    # inner constructor.
    function EigenExpansion(M::AbstractMatrix{<:Real})
        F = eigen(M)
        P = F.vectors
        D = Diagonal(F.values)
        Pinv = inv(P)
        return new{typeof(P), eltype(F.values), typeof(F.values), typeof(Pinv)}(P, D, Pinv)
    end
end 

# outer constructor
# Tests written
"""
    EigenExpansion(M::Tridiagonal{S, T}}) where {S<:Real, T<:AbstractVector{S}}
    EigenExpansion(M::Diagonal{S, T}}) where {S<:Real, T<:AbstractVector{S}}
    EigenExpansion(M::SymTridiagonal{S, T}}) where {S<:Real, T<:AbstractVector{S}}
    EigenExpansion(M::Symmetric{S, T}}) where {S<:Real, T<:AbstractVector{S}}
    EigenExpansion(M::AbstractMatrix{<:Real})
    EigenExpansion(M::Bidiagonal{S, T}}) where {S<:Real, T<:AbstractVector{S}}

Try to find a decomposition M = P*D*P^(-1).
"""
EigenExpansion(M::Tridiagonal{S, T}) where {S<:Real, T<:AbstractVector{S}} = EigenExpansion(SizedMatrix{size(M)...,S}(M))
EigenExpansion(M::Diagonal{S, T}) where {S<:Real, T<:AbstractVector{S}} = EigenExpansion(SizedMatrix{size(M)...,S}(M))
EigenExpansion(M::Bidiagonal{S, T}) where {S<:Real, T<:AbstractVector{S}} = EigenExpansion(SizedMatrix{size(M)...,S}(M))
EigenExpansion(M::SymTridiagonal{S, T}) where {S<:Real, T<:AbstractVector{S}} = EigenExpansion(SizedMatrix{size(M)...,S}(M))
EigenExpansion(M::Symmetric{S, T}) where {S<:Real, T<:AbstractVector{S}} = EigenExpansion(SizedMatrix{size(M)...,S}(M))


"""
    Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::EigenExpansion{Q, R, S, T}) where {Q<:AbstractMatrix{<:Real}, R<:Real, S<:AbstractVector{R}, T<:AbstractMatrix{<:Real}}

Pretty display of EigenExpansion types. 
"""
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::EigenExpansion{Q, R, S, T}) where {Q<:AbstractMatrix{<:Real}, R<:Real, S<:AbstractVector{R}, T<:AbstractMatrix{<:Real}}
    print(io, "EigenExpansion{$Q, $R, $S, $T}\n\nP:\n\n")
    show(io, mime, F.P) 
    print(io, "\n\nD:\n\n")
    show(io, mime, F.D)
    print(io, "\n\nPinv:\n\n")
    show(io, mime, F.Pinv)
end



# Tests written. 
"""
    ==(E1::EigenExpansion, E2::EigenExpansion)

Test whether E1 and E2 have the same decomposition P*D*P^(-1), so whether E1.P == E2.P, E1.D == E2.D and E1.Pinv == E2.Pinv.
"""
==(E1::EigenExpansion, E2::EigenExpansion) = E1.P == E2.P && E1.D == E2.D && E1.Pinv == E2.Pinv

# tests written
"""
    exp(s::Real, F::EigenExpansion)

Calculate e^(s*M) in a numerically efficient way. 

If F=EigenExpansion(M), then exp(s*M)=exp(s, F)=F.P*exp(s*F.D)*F.Pinv, which makes 
use of the fast implementation of exp for diagonal matrices (it is just the exponentiation of the diagonal elements).     
"""
exp(s::Real, F::EigenExpansion) =  F.P*exp(s*F.D)*F.Pinv

# Tests written.
"""
    makeQ(n::Integer)

Calculate the (2n+1)x(2n+1) tridiagonal matrix Q as in Schraiber 2018. 
"""
function makeQ(n::Integer)
    lowerdiagonal = 0.5*(1.0:2n).*(0.0:2n-1.0)
    diagonal = -(0.0:2n).*(2n:-1.0:0.0)
    upperdiagonal = 0.5*(2n:-1.0:1.0).*(2n-1:-1.0:0.0)
    return Tridiagonal(lowerdiagonal, diagonal, upperdiagonal)
end 

# Tests written.
"""
    makeQꜜ(n::Integer)

Calculate the (2n+1)x(2n+1) tridiagonal matrix Qꜜ as in Schraiber 2018.  
"""
function makeQꜜ(n::Integer)
    lowerdiagonal = 0.5*(1.0:2n).*(0.0:2n-1.0)
    diagonal = -(0.0:2n).*(2n+1.0:-1.0:1.0)
    upperdiagonal = 0.5*(2n+1:-1.0:2.0).*(2n:-1.0:1.0)
    return Tridiagonal(lowerdiagonal, diagonal, upperdiagonal)
end

# Tests written. 
"""
    calcmatrix(τC::Real, τA::Real, Qꜜ::EigenExpansion, Q::EigenExpansion)

Calculate the matrix `e^{Q*τ_A}e^{Qꜜτ_C}` as in Schraiber. Note that in general, for matrices, exp(A+B) ≠ exp(A)exp(B) (equality holds if and only if A and B commute, which is not the case for Q and Qꜜ). 
"""
calcmatrix(τC::Real, τA::Real, Qꜜ::EigenExpansion, Q::EigenExpansion) =  exp(τA, Q)*exp(τC, Qꜜ)

# Tests written 
"""
    preparedata(n::Integer, frequencies::AbstractVector{<:Real})

Returns a named tuple (Qꜜ=Qꜜ, Q=Q, binomialcoefficients=binomialcoefficients, hvals=hvals), where 
Qꜜ and Q are EigenExpansions of Qꜜ and Q (as in Schraiber 2018), respectively. 
hvals is a (2n+1) x length(y) matrix of type SizedMatrix. SizedMatrix is used rather than SMatrix because this matrix is so large and 
SMatrix is slow for large matrices. The matrix hvals is defined as hvals[k+1,i] = frequencies[i]^k*(1-frequencies[i])^(2n-k) (note that Julia has 1-based indexing).
binomialcoefficients is a vector (of type SVector) containing the binomial coefficients 2n over k, k=0,...,2n.  
"""
function preparedata(n::Integer, frequencies::AbstractVector{<:Real})
    Qꜜ = EigenExpansion(makeQꜜ(n))
    Q = EigenExpansion(makeQ(n))
    binomialcoefficients = SVector{2n+1, Int}(binomial.(2n, 0:2n))
    hvals = SizedMatrix{2n+1, length(frequencies), Float64}(undef)
    for j in eachindex(frequencies)
        for k in 0:2n 
            hvals[k+1, j] = frequencies[j]^k*(1.0-frequencies[j])^(2n-k)
        end 
    end
    return (Qꜜ=Qꜜ, Q=Q, binomialcoefficients=binomialcoefficients, hvals=hvals)
end

# tests written 
"""
    logprobderivedreads!(q::AbstractDict{<:Tuple{Integer, Integer}, <:AbstractVector{<:Real}}, n::Integer, τC::Real, τA::Real, ϵ::Real, coverages::AbstractVector{<:Integer}, uniquecoverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, Qꜜ::EigenExpansion, Q::EigenExpansion, binomialcoefficients::AbstractVector{<:Integer}, hvals::AbstractMatrix{<:Real})

Calculate the log probability of the data, a.k.a. the likelihood. This function mutates q, which is a dictionary, and is used for intermediate calculations. 
"""
function logprobderivedreads!(q::AbstractDict{<:Tuple{Integer, Integer}, <:AbstractVector{<:Real}}, n::Integer, τC::Real, τA::Real, ϵ::Real, coverages::AbstractVector{<:Integer}, uniquecoverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, Qꜜ::EigenExpansion, Q::EigenExpansion, binomialcoefficients::AbstractVector{<:Integer}, hvals::AbstractMatrix{<:Real})
    if τC < 0 || τA < 0 || ϵ < 0 || ϵ > 1
        return -Inf
    else
        M = calcmatrix(τC, τA, Qꜜ, Q) 
        updateq!(q, n, ϵ, uniquecoverages, binomialcoefficients)
        loglik = 0.0 # shouldn't that be zero(Type)?
        for index in eachindex(coverages, derivedreads, counts)
            loglik += calcloglik(coverages, derivedreads, counts, M, hvals, q, index)
        end
        return loglik
    end
end
# End likelihood functions. 

# tests written 
"""
    filtervectorsandapplycountmap(coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, allowedindices::AbstractVector{Bool})

This function does the following:
coverages_filtered = coverages[allowedindices]
derivedreads_filtered = derivedreads[allowedindices]
frequencies_filtered = frequencies[allowedindices]

Next, it counts how many times (coverages_filtered[i], derivedreads_filtered[i], frequencies_filtered[i]) occurs, for all triples, and next returns four vectors (coverages_filtered_unique, derivedreads_filtered_unique, frequencies_filtered_unique, counts), where each triple (coverages_filtered_unique[i], derivedreads_filtered_unique[i], frequencies_filtered_unique[i]) is unique and occurs counts[i] times in zip(coverages_filtered, derivedreads_filtered, frequencies_filtered).
"""
function filtervectorsandapplycountmap(coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, allowedindices::AbstractVector{Bool})
    coverages = coverages[allowedindices]
    derivedreads = derivedreads[allowedindices]
    frequencies = frequencies[allowedindices]
    tuples = zip(coverages, derivedreads, frequencies)
    cm = countmap(tuples)
    coverages = [key[1] for key in keys(cm)]
    derivedreads = [key[2] for key in keys(cm)]
    frequencies = [key[3] for key in keys(cm)]
    counts = collect(values(cm))
    return coverages, derivedreads, frequencies, counts
end 

# tests written 
"""
    makeqfixedn(n::Integer, uniquecoveragesandderivedreads::AbstractVector{<:Tuple{Integer, Integer}})

Construct a dictionary with keys (coverages[i], derivedreads[i]) and values real vectors of length 2n+1.
"""
makeqfixedn(n::Integer, uniquecoveragesandderivedreads::AbstractVector{<:Tuple{Integer, Integer}}) = Dict((coverage, numderivedreads) => Vector{Float64}(undef, 2n+1) for (coverage, numderivedreads) in uniquecoveragesandderivedreads)

# test written.
"""
    makeqfixedn(n::Integer, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer})
"""
function makeqfixedn(n::Integer, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer})
    uniquecoveragesandderivedreads = unique(zip(coverages, derivedreads))
    return makeqfixedn(n, uniquecoveragesandderivedreads)
end 

# tests written 
"""
    makeq(nmax::Integer, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer})

Construct a vector of length nmax, where the n-th item is a dictionary with keys (coverages[i], derivedreads[i]) and values real vectors of length 2n+1. 
"""
function makeq(nmax::Integer, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer})
    uniquecoveragesandderivedreads = unique(zip(coverages, derivedreads))
    return [makeqfixedn(n, uniquecoveragesandderivedreads) for n in 1:nmax]
end 

# tests written 
"""
    updateq!(q::AbstractDict{<:Tuple{Integer, Integer}, <:AbstractVector{<:Real}}, n::Integer, ϵ::Real, uniquecoverages::AbstractVector{<:Integer}, binomialcoefficients::AbstractVector{<:Integer})

Update the dictionary q, so that q[(R, d)][k+1] is the probability of d derived reads out of R when you have k derived alleles with n inidividuals and error rate ϵ. 
"""
function updateq!(q::AbstractDict{<:Tuple{Integer, Integer}, <:AbstractVector{<:Real}}, n::Integer, ϵ::Real, uniquecoverages::AbstractVector{<:Integer}, binomialcoefficients::AbstractVector{<:Integer})
    for R in uniquecoverages
        for k in 0:2n
            dist = Binomial(R, k*(1.0-ϵ)/(2n) + (2n-k)*ϵ/(2n))
            for d in 0:R
                if haskey(q, (R, d))
                    q[R, d][k+1] = binomialcoefficients[k+1] * pdf(dist, d)
                end 
            end
        end     
    end          
end 

"""
    calcloglik(coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, M::AbstractMatrix{<:Real}, hvals::AbstractMatrix{<:Real}, q::AbstractDict{<:Tuple{Integer, Integer}, <:AbstractVector{<:Real}}, index::Integer)

Calculate the log-likelihood at locus index. 
"""
calcloglik(coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, M::AbstractMatrix{<:Real}, hvals::AbstractMatrix{<:Real}, q::AbstractDict{<:Tuple{Integer, Integer}, <:AbstractVector{<:Real}}, index::Integer) = counts[index]*log(max(0.0, dot(q[coverages[index], derivedreads[index]], M, view(hvals, :, index))))

"""
    readcsvfile(filename::AbstractString)

Read CSV file and turn it into a DataFrame (from DataFrames package). It automatically detects whether the CSV file has a header. 
"""
function readcsvfile(filename::AbstractString, header::Union{Nothing, Bool})
    # if header=nothing, determine whether the file has a header. 
    if header === nothing 
        dftest = CSV.read(filename, DataFrame, header = false, limit=1)  
        if typeof(dftest[1,1]) <: AbstractString 
            # the first item in the first row was a string, so the csv file has a header.
            header = true 
        else
            # the first item in the first row was not a string, so the csv file has no header. 
            header = false 
        end 
    end 
    return CSV.read(filename, DataFrame, header = header)
end 

"""
    exactposterior(nrange::AbstractVector{<:Integer}, τCrange::AbstractVector{<:Real}, τArange::AbstractVector{<:Real}, ϵrange::AbstractVector{<:Real}, coverages::AbstractVector{<:Integer}, uniquecoverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, counts::AbstractVector{<:Integer}, messages::Integer)

Calculate the exact posterior up to a fixed unknown constant that depends on the data but does not depend on the parameters. 
"""
function exactposterior(nrange::AbstractVector{<:Integer}, τCrange::AbstractVector{<:Real}, τArange::AbstractVector{<:Real}, ϵrange::AbstractVector{<:Real}, coverages::AbstractVector{<:Integer}, uniquecoverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, counts::AbstractVector{<:Integer}; messages::Integer=0)
    # Check conditions.
    length(nrange) > 0 || throw(ArgumentError("nrange should have at least one element."))
    minimum(nrange) > 0 || throw(ArgumentError("all n values in nrange should be positive."))
    length(τCrange) > 0 || throw(ArgumentError("τCrange should have at least one element."))
    minimum(τCrange) ≥ 0 || throw(ArgumentError("All values in τCrange should be non-negative."))
    length(τArange) > 0 || throw(ArgumentError("τArange should have at least one element."))
    minimum(τArange) ≥ 0 || throw(ArgumentError("All values in τArange should be non-negative."))
    length(ϵrange) > 0 || throw(ArgumentError("ϵrange should have at least one element."))
    all(0 .≤ ϵrange .<0.5) || throw(DomainError("All values in ϵrange should be in [0.0,0.5)."))
    maximum(coverages) > 0 || throw(ArgumentError("Inference is impossible when all coverages are zero."))
    minimum(derivedreads) ≥ 0 || throw(DomainError("All reads should be non-negative."))
    all(derivedreads .≤ coverages) || throw(DomainError("All reads should be at most the coverage."))
    all(0.0 .≤ frequencies .≤ 1.0) || throw(DomainError("The frequencies should be between 0.0 and 1.0."))
    length(frequencies) == length(derivedreads) == length(coverages) || throw(DimensionMismatch("frequencies, derivedreads and coverages should be of the same length."))
    minimum(counts) ≥ 0 || throw(DomainError("All elements of counts should be non-negative."))
    length(nrange) > 0 || throw(ArgumentError("nrange should have at least one element."))
    length(τCrange) > 0 || throw(ArgumentError("τCrange should have at least one element."))
    length(τArange) > 0 || throw(ArgumentError("τArange should have at least one element."))
    length(ϵrange) > 0 || throw(ArgumentError("ϵrange should have at least one element."))
    all(0 .≤ ϵrange .< 0.5) || throw(DomainError("ϵ should be in [0.0,0.5)."))
    
    nrows = length(nrange)*length(τCrange)*length(τArange)*length(ϵrange)
    
    ns = Vector{Int}(undef, nrows)
    τCs = Vector{Float64}(undef, nrows)
    τAs = Vector{Float64}(undef, nrows)
    ϵs = Vector{Float64}(undef, nrows)
    logliks = Vector{Float64}(undef, nrows)

    index = 1
    for n in nrange
        Qꜜ, Q, binomialcoefficients, hvals = preparedata(n, frequencies)
        q = makeqfixedn(n, coverages, derivedreads)
        M = Matrix{Float64}(undef, 2n+1, 2n+1)
        for τC in τCrange 
            expτCQꜜ = exp(τC, Qꜜ)
            for τA in τArange
                expτAQ = exp(τA, Q)    
                mul!(M, expτAQ, expτCQꜜ)
                for ϵ in ϵrange
                    # update q
                    updateq!(q, n, ϵ, uniquecoverages, binomialcoefficients)
                    loglik = 0.0
                    for index in eachindex(coverages, derivedreads, counts)
                        loglik += calcloglik(coverages, derivedreads, counts, M, hvals, q, index)
                    end
                    ns[index] = n 
                    τCs[index] = τC 
                    τAs[index] = τA
                    ϵs[index] = ϵ
                    logliks[index] = loglik 
                    if messages > 0 # No messages at all when messages ≤ 0.
                        if index % messages == 0
                            println("I did ", index, " posterior calculations, ", round(Int, 100*index / nrows, RoundDown), "% finished!")
                        end
                    end
                    index += 1 
                end # ϵ
            end # τA
        end # τC 
    end # n
    @assert index == nrows + 1 
    return ns, τCs, τAs, ϵs, logliks
end # function.

"""
    exactposterior(nrange::AbstractVector{<:Integer}, τCrange::AbstractVector{<:Real}, τArange::AbstractVector{<:Real}, ϵrange::AbstractVector{<:Real}, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, messages::Integer=length(n)*length(τCrange)*length(τArange)*length(ϵrange)÷100)
"""
function exactposterior(nrange::AbstractVector{<:Integer}, τCrange::AbstractVector{<:Real}, τArange::AbstractVector{<:Real}, ϵrange::AbstractVector{<:Real}, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}; messages::Integer=length(n)*length(τCrange)*length(τArange)*length(ϵrange)÷100)
    # Check conditions.
    length(nrange) > 0 || throw(ArgumentError("nrange should have at least one element."))
    minimum(nrange) > 0 || throw(ArgumentError("all n values in nrange should be positive."))
    length(τCrange) > 0 || throw(ArgumentError("τCrange should have at least one element."))
    minimum(τCrange) ≥ 0 || throw(ArgumentError("All values in τCrange should be non-negative."))
    length(τArange) > 0 || throw(ArgumentError("τArange should have at least one element."))
    minimum(τArange) ≥ 0 || throw(ArgumentError("All values in τArange should be non-negative."))
    length(ϵrange) > 0 || throw(ArgumentError("ϵrange should have at least one element."))
    all(0 .≤ ϵrange .<0.5) || throw(DomainError("All values in ϵrange should be in [0.0,0.5)."))
    maximum(coverages) > 0 || throw(ArgumentError("Inference is impossible when all coverages are zero."))
    minimum(derivedreads) ≥ 0 || throw(DomainError("All reads should be non-negative."))
    all(derivedreads .≤ coverages) || throw(DomainError("All reads should be at most the coverage."))
    all(0.0 .≤ frequencies .≤ 1.0) || throw(DomainError("The frequencies should be between 0.0 and 1.0."))
    length(frequencies) == length(derivedreads) == length(coverages) || throw(DimensionMismatch("frequencies, derivedreads and coverages should be of the same length."))
    
    # Remove frequencies 0.0 and 1.0 and zero coverages. 
    allowedindices = (coverages .> 0) .& (0.0 .< frequencies .< 1.0)
    any(allowedindices) || throw(ArgumentError("For at least one index i, coverages[i]>0 and 0.0<frequencies[i]<1.0."))
    
    coverages, derivedreads, frequencies, counts = filtervectorsandapplycountmap(coverages, derivedreads, frequencies, allowedindices)

    uniquecoverages = unique(coverages)

    return exactposterior(nrange, τCrange, τArange, ϵrange, coverages, uniquecoverages, derivedreads, frequencies, counts; messages = messages)
end # function.

"""
    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, counts::AbstractVector{<:Integer}, messages::Integer=nsteps÷100, scalingmessages::Bool=true)

MCMC with simulated proposals. There are counts[i] loci with coverage coverages[i], derivedreads[i] derived alleles and frequency frequencies[i] in the anchor population. 
"""
function MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, counts::AbstractVector{<:Integer}; messages::Integer=nsteps÷100, scalingmessages::Bool=true)
    # Check conditions.
    length(prioronτCτA) == 2 || throw(DimensionMismatch("prior on [τC, τA] must be of length 2."))
    minimum(prioronn) ≥ 1 || throw(ArgumentError("prioronn should have support on positive integers. There is always at least one individual."))
    nmax = maximum(prioronn)
    nmax ≤ 10 || throw(ArgumentError("For the current implementation we require that the prior on n is bounded by 10. I.e. P(prioronn ≤ 10)=1"))
    minimum(prioronϵ) ≥ 0 || throw(ArgumentError("prioronϵ should have support on [0,0.5], but $prioronϵ can take negative values."))
    maximum(prioronϵ) ≤ 0.5 || throw(ArgumentError("prioronϵ should have support on [0,0.5], but $prioronϵ can take values larger than 0.5."))
    nchains > 0 || throw(ArgumentError("There should be at least one chain."))
    nsteps > 0 || throw(ArgumentError("nsteps should be positive."))
    maximum(coverages) > 0 || throw(DomainError("At least some coverages should be positive.")) 
    minimum(derivedreads) ≥ 0 || throw(DomainError("All reads should be non-negative."))
    all(derivedreads .≤ coverages) || throw(DomainError("All reads should be at most the coverage."))
    all(0.0 .≤ frequencies .≤ 1.0) || throw(DomainError("The frequencies should be between 0.0 and 1.0."))
    length(coverages) == length(derivedreads) == length(frequencies) == length(counts) || throw(DimensionMismatch("coverages, derivedreads, frequencies and counts should be of the same length."))
    
    
    # Remove frequencies 0.0 and 1.0
    allowedindices = (coverages .> 0) .& (0.0 .< frequencies .< 1.0) 
    any(allowedindices) || throw(ArgumentError("At least one element of y with corresponding positive coverage should be strictly between 0 and 1."))

    frequencies = frequencies[allowedindices]
    derivedreads = derivedreads[allowedindices]
    coverages = coverages[allowedindices]
    counts = counts[allowedindices]

    uniquecoverages = unique(coverages)
    smallestn, largestn = extrema(prioronn)
    smallestτC, smallestτA = minimum(prioronτCτA)
    largestτC, largestτA = maximum(prioronτCτA)
    smallestϵ, largestϵ = extrema(prioronϵ)
    gridτC = range(smallestτC, largestτC, length=23)[2:end] # avoid a 0.0 value
    gridτA = range(smallestτA, largestτA, length=23)[2:end] # avoid a 0.0 value
    gridϵ = range(smallestϵ, largestϵ, length=24)[2:end-1] # avoid a 0.0 and 0.5 value
    ns, τCs, τAs, ϵs, logliks = exactposterior(smallestn:largestn, gridτC, gridτA, gridϵ, coverages, uniquecoverages, derivedreads, frequencies, counts)
    maxlikindex = argmax(logliks)
    nmaxlik = ns[maxlikindex]
    τAmaxliksconditionedonn = Vector{Float64}(undef, nmax)
    τCmaxliksconditionedonn = Vector{Float64}(undef, nmax)
    ϵmaxliksconditionedonn = Vector{Float64}(undef, nmax)
    for n in 1:nmax 
        filteronn = ns .== n 
        τCsconditionedonn = τCs[filteronn]
        τAsconditionedonn = τAs[filteronn]
        ϵsconditionedonn = ϵs[filteronn]
        logliksconditionedonn = logliks[filteronn]
        maxlikindex = argmax(logliksconditionedonn)
        τCmaxliksconditionedonn[n] = τCsconditionedonn[maxlikindex]
        τAmaxliksconditionedonn[n] = τAsconditionedonn[maxlikindex]
        ϵmaxliksconditionedonn[n] = ϵsconditionedonn[maxlikindex]
        @assert unique(ns[filteronn]) == [n]
        @assert n ∉ ns[.!filteronn]
    end 
    
    coverages = SizedVector{length(coverages), Int}(coverages)
    derivedreads = SizedVector{length(derivedreads), Int}(derivedreads)
    Qꜜs = Vector{EigenExpansion}(undef, nmax)
    Qs = Vector{EigenExpansion}(undef, nmax)
    hvals = Vector{Any}(undef, nmax) # contains the matrices with values y^k(1-y)^{2n-k}.
    binomialcoefficients = Vector{Any}(undef, nmax)
    for n in 1:nmax
        Qꜜs[n], Qs[n], binomialcoefficients[n], hvals[n] = preparedata(n, frequencies)
    end
    hvals = [item for item in hvals] # change type of hvals to more specific one then Any.
    binomialcoefficients = [item for item in binomialcoefficients] # same.
    # Each item in results contains a tuple 
    # (nsample, τCsample, τAsample, ϵsample, acceptedvec, logjointprob), 
    # which contain the states of the Markov chain. 
    results = Vector{Tuple{Vector{Int}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, BitMatrix, Matrix{Float64}}}(undef,nchains)
    @threads for chainindex in eachindex(results) 
        q = makeq(nmax, coverages, derivedreads)
        nsample = Vector{Int}(undef, nsteps)
        τCsample = Matrix{Float64}(undef, nsteps, nmax)
        τAsample = Matrix{Float64}(undef, nsteps, nmax)
        ϵsample = Matrix{Float64}(undef, nsteps, nmax)
        acceptedvec = BitMatrix(undef, nsteps, nmax+1) # BitMatrix is more efficient than Matrix{Bool} 
        # (roughly, one bit per item, instead of one byte.)
        logjointprob = Matrix{Float64}(undef, nsteps, nmax+1)
        # Initialise.
        nprevious = nmaxlik
        τCprevious = copy(τCmaxliksconditionedonn)
        τAprevious = copy(τAmaxliksconditionedonn)
        ϵprevious = copy(ϵmaxliksconditionedonn)
        
        nsample[1] = nprevious
        τCsample[1,:] = τCprevious
        τAsample[1,:] = τAprevious 
        ϵsample[1,:] = ϵprevious
        acceptedvec[1,:] .= true 
        # Calculate log likelihood and log prior density with initial values.
        loglikprevious = Vector{Float64}(undef, nmax)
        loglikproposal = Vector{Float64}(undef, nmax)
        for n in 1:nmax
            loglikprevious[n] = logprobderivedreads!(q[n], n, τCprevious[n], τAprevious[n], ϵprevious[n],coverages, uniquecoverages, derivedreads, counts, Qꜜs[n], Qs[n], binomialcoefficients[n], hvals[n])
        end
        # check if the initial value is valid. 
        any(loglikprevious .== -Inf) && throw(DomainError("The data does not support the initial value."))
        logpdfpriorprevious = Vector{Float64}(undef, nmax)
        for n in 1:nmax
            logpdfpriorprevious[n] = logpdf(prioronτCτA, [τCprevious[n], τAprevious[n]]) + logpdf(prioronϵ, ϵprevious[n])
        end
        logpdfnprev = logpdf(prioronn, nprevious)
                
        any(logpdfpriorprevious .== -Inf) && throw(DomainError("The initial value is outside the support of the prior."))
        logpdfnprev == -Inf && throw(DomainError("The initial value is outside the support of the prior on n."))
        logjointprob[1,2:end] .= loglikprevious .+ logpdfpriorprevious
        logjointprob[1,1] = loglikprevious[nprevious] + logpdfpriorprevious[nprevious] + logpdfnprev
        # Define proposal densities. 
        Σ = Diagonal(fill(0.000001, 2))
        distproposalτCτA = fill(MvNormal(Σ), nmax)# Multivariate normal distribution with mean zero and 
        # covariance matrix Σ.
        σ = 0.0001
        distproposalϵ = fill(Normal(0.0, σ),nmax)
        nproposalprobvec = fill(0.2/(nmax-1), nmax)
        nproposalprobvec[nprevious] = 0.8
        distproposaln = Categorical(nproposalprobvec) # so with 80% chance nproposal is
        # equal to the previous value, or equal to one of the other values 
        # 1,..., nprevious-1, nprevious+1,...,nmax with probability 0.2/(nmax-1).  
        scalefactor = fill(2.38^2/4, nmax) # = 2.38^2/4, as we have dimension of prior equal to 4.
        τCproposal = Vector{Float64}(undef, nmax)
        τAproposal = Vector{Float64}(undef, nmax)
        ϵproposal = Vector{Float64}(undef, nmax)
        logpdfpriorproposal = Vector{Float64}(undef, nmax)
        logacceptanceprobability = Vector{Float64}(undef, nmax)
        acceptanceprobability = Vector{Float64}(undef, nmax)
        accepted = BitVector(undef, nmax)
        for step in 2:nsteps  
            # draw proposals.
            for index in eachindex(τCproposal, τAproposal, ϵproposal) 
                τCproposal[index], τAproposal[index] = [τCprevious[index], τAprevious[index]] + rand(distproposalτCτA[index])
                ϵproposal[index] = ϵprevious[index] + rand(distproposalϵ[index])
            end
            for n in 1:nmax
                loglikproposal[n] = logprobderivedreads!(q[n], n, τCproposal[n], τAproposal[n], ϵproposal[n], coverages, uniquecoverages, derivedreads, counts, Qꜜs[n], Qs[n], binomialcoefficients[n], hvals[n])
            end 
            for n in 1:nmax
                logpdfpriorproposal[n] = logpdf(prioronτCτA, [τCproposal[n], τAproposal[n]]) + logpdf(prioronϵ, ϵproposal[n])
            end 
            logacceptanceprobability .= loglikproposal .- loglikprevious .+ logpdfpriorproposal .- logpdfpriorprevious
            acceptanceprobability .= exp.(logacceptanceprobability)
            # accept or reject the proposal with probability max(1, acceptanceprobability), 
            # note that rand() samples from the uniform distribution.
            for index in eachindex(accepted, acceptanceprobability)
                accepted[index] = rand() ≤ acceptanceprobability[index]
            end 
            for n in 1:nmax
                if accepted[n]
                    τCsample[step, n] = τCprevious[n] = τCproposal[n]
                    τAsample[step, n] = τAprevious[n] = τAproposal[n]
                    ϵsample[step, n] = ϵprevious[n] = ϵproposal[n]
                    acceptedvec[step, n+1] = true
                    logjointprob[step, n+1] = loglikproposal[n] + logpdfpriorproposal[n]
                    # Update everything for the next cycle. 
                    loglikprevious[n] = loglikproposal[n]
                    logpdfpriorprevious[n] = logpdfpriorproposal[n]
                else
                    τCsample[step, n] = τCprevious[n]
                    τAsample[step, n] = τAprevious[n]
                    ϵsample[step, n] = ϵprevious[n]
                    acceptedvec[step, n+1] = false
                    logjointprob[step, n+1] = loglikprevious[n] + logpdfpriorprevious[n]
                    # no updating: loglikprevious and logpdfpriorprevious remain the same.
                end 
            end
            nproposal = rand(distproposaln) # so with 80% chance nproposal is equal to nprevious
            if nproposal == nprevious
                nsample[step] = nproposal
                acceptedvec[step, 1] = true
                logjointprob[step, 1] = logjointprob[step, nproposal+1] + logpdfnprev
            else 
                logpdfnprop = logpdf(prioronn, nproposal)
                logacceptprobn = logjointprob[step, nproposal+1] - logjointprob[step, nprevious+1] + logpdfnprop - logpdfnprev
                acceptprobn = exp(logacceptprobn)
                if rand() ≤ acceptprobn # proposal accepted 
                    nsample[step] = nprevious = nproposal
                    acceptedvec[step, 1] = true
                    logjointprob[step, 1] = logjointprob[step, nproposal+1] + logpdfnprop
                    nproposalprobvec = fill(0.2/(nmax-1), nmax)
                    nproposalprobvec[nprevious] = 0.8
                    distproposaln = Categorical(nproposalprobvec) # update the proposal distribution for n.
                    logpdfnprev = logpdfnprop
                else # proposal rejected 
                    nsample[step] = nprevious
                    acceptedvec[step, 1] = false 
                    logjointprob[step, 1] = logjointprob[step, nprevious+1] + logpdfnprev
                end 
            end                         
            if step % 1_000 == 0 && step ≥ 5_000
                # the covariance is calculated from the previous steps and used in the next iteration.
                for n in 1:nmax
                    Σ = Diagonal([var(τCsample[3_000:20:step, n], corrected=true), var(τAsample[3_000:20:step, n], corrected=true)])
                    σ = std(ϵsample[3_000:20:step, n], corrected=true)
                    if 10_000 < step < nsteps÷4  
                        acceptancerate = mean(acceptedvec[step-999:step, n+1]) # acceptance rate in the last 1000 steps.
                        if acceptancerate < 0.1 # acceptancerate too low, take smaller steps.
                            scalefactor[n] = 3*scalefactor[n]/4
                            scalingmessages && println("Chain ", chainindex, " talking, I scaled the scalefactor with factor 3/4, new scalefactor = ", scalefactor, '.')
                        elseif acceptancerate > 0.15 # acceptancerate too large, take bigger steps.
                            scalefactor[n] = 4*scalefactor[n]/3
                            scalingmessages && println("Chain ", chainindex, " talking, I scaled the scalefactor with factor 4/3, new scalefactor = ", scalefactor, '.')
                        end # So do nothing when the acceptancerate is between 0.1 and 0.15.     
                    end 
                    distproposalτCτA[n] = MvNormal(scalefactor[n]*Σ+1.0e-14*I)
                    distproposalϵ[n] = Normal(0.0, sqrt(scalefactor[n]) * σ + 1.0e-14)
                end 
            end
            if messages > 0 # No messages at all when messages ≤ 0.
                if step % messages == 0
                    println("Chain ", chainindex, " talking, I finished step ", step, " of ", nsteps, ", ", round(Int, 100*step / nsteps, RoundDown), " percent finished.")
                end
            end
        end # MCMC step
        results[chainindex] = (nsample, τCsample, τAsample, ϵsample, acceptedvec, logjointprob)
    end # Chains. 
    return results
end # function.

"""
    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}, messages::Integer=nsteps÷100, scalingmessages::Bool=true)

coverages[i] is the coverage, derivedreads[i] the number of derived reads and frequencies[i] the frequency of the derived allele in the anchor population at locus i. 
"""
function MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, coverages::AbstractVector{<:Integer}, derivedreads::AbstractVector{<:Integer}, frequencies::AbstractVector{<:Real}; messages::Integer=nsteps÷100, scalingmessages::Bool=true)
    length(frequencies) == length(derivedreads) == length(coverages) || throw(DimensionMismatch("frequencies, derivedreads and coverages should be of the same length."))
    
    # Remove frequencies 0.0 and 1.0
    allowedindices = (coverages .> 0) .& (0.0 .< frequencies .< 1.0)
    any(allowedindices) || throw(ArgumentError("At least one element of frequencies with corresponding positive coverage should be strictly between 0 and 1."))

    coverages, derivedreads, frequencies, counts = filtervectorsandapplycountmap(coverages, derivedreads, frequencies, allowedindices)
    
    return MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, coverages, derivedreads, frequencies, counts, messages, scalingmessages)
end 

"""
    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, df::AbstractDataFrame, messages::Integer=nsteps÷100, scalingmessages::Bool=true)

df is a DataFrame (from the DataFrames package) in DICE 2-pop format: It should contain exactly four columns, the first one contains the number of ancestral reads, the second the number of derived reads, the third the frequency of the derived allele in the anchor population and the fourth the number of loci with exactly this combination of data. 
"""
function MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, df::AbstractDataFrame; messages::Integer=nsteps÷100, scalingmessages::Bool=true)
    coverages = df[!, 1] + df[!, 2]
    derivedreads = df[!, 2]
    frequencies = df[!, 3]
    counts = df[!, 4]

    return MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, coverages, derivedreads, frequencies, counts; messages = messages, scalingmessages = scalingmessages)
end 

"""
    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, dicefile::IO, messages::Integer=nsteps÷100, scalingmessages::Bool=true, header::Bool=true)

dicefile is an opened CSV-file in DICE 2-pop format: It should contain exactly 4 columns, the first one contains the number of ancestral reads, the second the number of derived reads, the third the frequency of the derived allele in the anchor population and the fourth the number of loci with exactly this combination of data.

If you have Julia 1.3 or higher, and the file is a zipped file, open it using a zip decompressor:
```julia
using CodecZlib
dicefile = GzipDecompressorStream(open("path/to/your/file.csv.gz"))
```
With the named argument "header" one should indicate whether the CSV file has a header. The default is true.
"""
function MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, dicefile::IO; messages::Integer=nsteps÷100, scalingmessages::Bool=true, header::Bool=true)
    df = CSV.read(dicefile, DataFrame, header = header) 
    return MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, df; messages = messages, scalingmessages = scalingmessages)
end  


"""
    MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, dicefile::AbstractString, messages::Integer=nsteps÷100, scalingmessages::Bool=true)

 dicefile is a (zipped) CSV-file in DICE 2-pop format: It should contain exactly four columns, the first one contains the number of ancestral reads, the second the number of derived reads, the third the frequency of the derived allele in the anchor population and the fourth the number of loci with exactly this combination of data.
"""
function MCMCsampler(nchains::Integer, nsteps::Integer, prioronn::DiscreteUnivariateDistribution, prioronτCτA::ContinuousMultivariateDistribution, prioronϵ::ContinuousUnivariateDistribution, dicefile::AbstractString; messages::Integer=nsteps÷100, scalingmessages::Bool=true, header::Union{Nothing, Bool}=nothing)
    # Can handle both CSV files with and without headers. 
    df = readcsvfile(dicefile, header)
    return MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, df; messages = messages, scalingmessages = scalingmessages)
end  


"""
    unpackposterior(nsteps::Integer, chains::AbstractVector{<:Tuple{AbstractVector{<:Integer}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractMatrix{Bool}, AbstractMatrix{<:Real}}})

Calculate the unconditional posterior from the output of MCMCsampler. `chains` is the output of the sampler, `nsteps` is the number of MCMC samples. 
"""
function unpackposterior(nsteps::Integer, chains::AbstractVector{<:Tuple{AbstractVector{<:Integer}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractMatrix{Bool}, AbstractMatrix{<:Real}}})
    results = DataFrame(nsample = Int[],
            τCsample = Float64[],
            τAsample = Float64[],
            ϵsample = Float64[],
            accepted = Bool[],
            logjointprob = Float64[],
            chainid = Int[]
            )
    for index in eachindex(chains)
        nsample, τCsample, τAsample, ϵsample, _, logjointprob = chains[index]
        τCsampleunpacked = Vector{Float64}(undef, nsteps)
        τAsampleunpacked = Vector{Float64}(undef, nsteps)
        ϵsampleunpacked = Vector{Float64}(undef, nsteps)
        for index in 1:nsteps
            τCsampleunpacked[index] = τCsample[index, nsample[index]]
            τAsampleunpacked[index] = τAsample[index, nsample[index]]
            ϵsampleunpacked[index] = ϵsample[index, nsample[index]]
        end 
        acceptedvecunpacked = BitVector(undef, nsteps)
        acceptedvecunpacked[1] = true 
        for index in 2:nsteps
            everythingthesame = (τCsampleunpacked[index] == τCsampleunpacked[index-1]) && (τAsampleunpacked[index] == τAsampleunpacked[index-1]) && (ϵsampleunpacked[index] == ϵsampleunpacked[index-1]) && (nsample[index] == nsample[index-1])
            acceptedvecunpacked[index] = !everythingthesame
        end 
        df = DataFrame(nsample = nsample,
                            τCsample = τCsampleunpacked,
                            τAsample = τAsampleunpacked,
                            ϵsample = ϵsampleunpacked,
                            accepted = acceptedvecunpacked,
                            logjointprob = logjointprob[:,1],
                            chainid = fill(Int(index), nsteps)
                            )
        results = vcat(results, df)
    end 
    return results
end 

end # module 

