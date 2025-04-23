# Introduction

Environmental DNA or forensic DNA can contain the DNA of multiple human individuals. Sequencing this DNA and aligning it to the human reference will result in alignments in BAM format. 
CHARON: **C**alculating **HA**plotype distribution to **R**esolve **O**riginal **N**umber of individuals and demography is software written in Julia to infer 

1. **Number of Diploid Individuals**: An estimate of the current population size based on genetic data.
2. **Demography**: Drift time from a particular population and can determine which population has the lowest drift time thus the closest proxy.

This software provides an MCMC sampler, so in addition to estimators, it provides (Bayesian) confidence intervals for the estimators.  

The name comes from the largest moon of Pluto, which has 5 moons: Charon, Styx, Nix, Kerberos, and Hydra.


## Assumptions

We generally assume:

- **equal contribution**: Each individual contributed more or less the same amount of DNA to the mix. 
- **same population of origin**: All the individuals come from the same population.

## Julia 

**Julia** is a high-level, high-performance programming language designed for technical computing. It combines the ease of use of languages like Python and R with the speed of C++.

- **Syntax and Ease of Use**: Julia's syntax is simple and intuitive, similar to Python and R, making it easy to learn and write. It supports multiple dispatch, allowing functions to behave differently based on their argument types, which is particularly useful for scientific computing.

- **Dynamic and 1-Based Indexing**: Julia is a dynamic language, meaning types are checked at runtime, which offers flexibility and ease of use. Additionally, Julia uses 1-based indexing, similar to R, which can be more intuitive for those coming from a mathematical background.

- **Performance**: Julia is designed for speed. It compiles to efficient machine code using LLVM, offering performance comparable to C++ without sacrificing ease of use. This makes it ideal for tasks that require high computational power.

- **Libraries and Ecosystem**: Julia has a rich ecosystem with numerous libraries for data analysis, machine learning, and scientific computing. It also integrates well with existing Python and R code, allowing users to leverage their existing tools and libraries.

- **Parallel and Distributed Computing**: Julia has built-in support for parallel and distributed computing, making it easier to write code that scales across multiple cores and machines.

Julia is a powerful tool for anyone looking to combine the simplicity of high-level languages with the performance of low-level languages. If you're familiar with R, Python, or C++, you'll find Julia to be a versatile and efficient addition to your programming toolkit.
