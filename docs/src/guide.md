
# Getting started 

  
## Install Julia 

Our software is written in the Julia language, and should be run by the Julia interpretor. Julia compiles the code in the background and thereby achieves speeds comparable with C. The current version of the Charon package is 0.3.2 and it should run with every Julia 1.X version. 

Go to [julialang.org](https://julialang.org/downloads/) and follow the installation instructions for your platform.

### Add Julia to the path. 
Check if Julia is added to your path by typing, for instance, `julia --version` in the terminal. [If not, follow the instructions.](https://julialang.org/downloads/platform/)


## Quick start and test data

After installing, you can use the following base count:

```bash
wget ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/IBS_ind4_basecount_IBS_ind4.gz
```

and allele frequency:

```bash
wget ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/IBS_ind4_freq_IBS.gz
```
and download the following [script](runmcmc.jl):
```bash
wget https://jan-van-waaij.github.io/Charon.jl/runmcmc.jl 
```

Put all three files in one folder, and navigate to that folder in the terminal. Then copy-paste 
```julia
julia --threads=4 runmcmc.jl IBS_ind4_basecount_IBS_ind4.gz IBS_ind4_freq_IBS.gz output.csv
```
in your terminal and press enter. This executes an MCMC sampler, whose output is saved to `output.csv`. This might take a few hours. See the ["Output format" section](#Output-format) below to interpret the output. The script takes care of installing and loading the appropriate Julia packages, including Charon. It generates 100'000 samples, and sets uniform priors on n, τC, τA and ϵ. 

### Windows
If you use Windows Powershell, you can use 
```powershell
Invoke-WebRequest -Uri "https://jan-van-waaij.github.io/Charon.jl/runmcmc.jl" -OutFile "runmcmc.jl"
Invoke-WebRequest -Uri "ftp://ftp.healthtech.dtu.dk/public/edna_allelefreq/IBS_ind4_basecount_IBS_ind4.gz" -OutFile "IBS_ind4_basecount_IBS_ind4.gz"
Invoke-WebRequest -Uri "ftp://ftp.healthtech.dtu.dk/public/edna_allelefreq/IBS_ind4_freq_IBS.gz" -OutFile "IBS_ind4_freq_IBS.gz"
```
instead of `wget`.

## Output format

The output is a [CSV-file](https://en.wikipedia.org/wiki/Comma-separated_values) consists of 7 columnns, 
1. nsample: n values
2. τCsample: τC values
3. τAsample: τA values
4. ϵsample: ϵ values 
5. accepted: was the proposal accepted?
6. logjointprob: the log probability, up to a additive constant, only depending on the data, but not on the parameters
7. chainid: whether this is the first, second, ... chain.

Each row is a draw from the posterior. 


## Prepare data

Prepare your eDNA sample data in the native format. CHARON required 2 files:
A) A base count derived from the BAM files for each segregating position in the population as such:

```
[derived count]  [total count]
```

For instance, say at position 239923 on chr 1, there are 4 reads supporting a derived variant and a total of 20 reads. The line for this position would be:

```
4 20
```

B) A file describing the frequency of the derived base (as a floating point number) such as in a population:

```
[freq]
```

For instance, say at our position 239923 on chr 1, 20% of a particular population has the derived variant, then the line for this position would be:

```
0.2
```

Please note that both files should have the same number of lines. Both files can be gzipped.

### Alternative data format.
We also allow data in the [DICE-2 format](https://github.com/grenaud/dice?tab=readme-ov-file#2-pop-method-input-data-format). 

A CSV file in DICE-2 format has four columns. In this order: number of ancestral reads, number of derived reads, the frequency of the anchor population, and in how many loci this combination of reads and frequency occurs. So, the first two columns contain non-negative integers, the third column is a real number between 0.0 and 1.0, and the last column is a positive integer. So an example is 
```
3    5    0.48    10
4    1    0.01    2
5    0    0.03    3
4    2    0.23    5
```

### Example files



An example DICE file is available via 
```bash
wget ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/IBS_ind4.dice.gz
```
Or in Windows Power Shell:
```powershell
Invoke-WebRequest -Uri "ftp://ftp.healthtech.dtu.dk/public/edna_allelefreq/IBS_ind4.dice.gz" -OutFile "IBS_ind4.dice.gz"
```

An example base count file is available  via 
```bash
wget ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/IBS_ind4_basecount_IBS_ind4.gz
```
Or in Windows Power Shell: 
```powershell
Invoke-WebRequest -Uri "ftp://ftp.healthtech.dtu.dk/public/edna_allelefreq/IBS_ind4_basecount_IBS_ind4.gz" -OutFile "IBS_ind4_basecount_IBS_ind4.gz"
```

An example frequency file is available  via 
```bash
wget ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/IBS_ind4_freq_IBS.gz
```
Or in Windows Power Shell: 
```powershell
Invoke-WebRequest -Uri "ftp://ftp.healthtech.dtu.dk/public/edna_allelefreq/IBS_ind4_freq_IBS.gz" -OutFile "IBS_ind4_freq_IBS.gz"
```

## Convenience script
[Click here for an example script.](runmcmc.jl) This script sets uniform priors on `τC`, `τA`, `ϵ` and `n`, and generates four chains with 100'000 samples. It works both with the DICE-2 format and the other format with seperate base count and frequency files. The script saves the output as a CSV file. You can use your favourite software to analyse the output. Save runmcmc.jl on your computer. It works as follows (assuming Julia is in your path, and you use Julia 1.5 or higher):
```julia
julia --threads=4 path/to/runmcmc.jl path/to/basecountfile path/to/frequencyfile path/to/outputfile
```
or 
```julia
julia --threads=4 path/to/runmcmc.jl path/to/dicefile path/to/outputfile
```
where the output is saved in `outputfile`. `dicefile`, `basecountfile`, and `frequencyfile` might be a CSV files or a gzipped CSV files. Example files can be found [here](#Example-files). The script installs Charon, and other necessary Julia packages, loads them and executes the MCMC sampler. 


### Example
So if you have downloaded `IBS_ind4_basecount_IBS_ind4.gz` and `IBS_ind4_freq_IBS.gz` (or `IBS_ind4.dice.gz`) and `runmcmc.jl` from the former steps and have saved them in the same folder, and you navigate to the folder in the terminal, then the following command runs the MCMC and saves the results to `output.csv`.
```julia
julia --threads=4 runmcmc.jl IBS_ind4_basecount_IBS_ind4.gz IBS_ind4_freq_IBS.gz output.csv
```
or
```julia
julia --threads=4 runmcmc.jl IBS_ind4.dice.gz output.csv
```

### Older versions of Julia
Older version of Julia (≤1.4), do not have the `--threads` flag, instead you should set the environment variable `JULIA_NUM_THREADS`. In Unix systems 
```base
export JULIA_NUM_THREADS=4
```
or in Windows Powershell:
```powershell
$env:JULIA_NUM_THREADS = 4
```
Then, run 
```julia
julia runmcmc.jl IBS_ind4_basecount_IBS_ind4.gz IBS_ind4_freq_IBS.gz output.csv
```
or
```julia
julia runmcmc.jl IBS_ind4.dice.gz output.csv
```
### Very old versions of Julia
In very old versions of Julia (≤1.2), the script does not work with gzipped files. So first unzip your files, and then run them as above. 

## Detailed description of the software 

Here follows a detailed description of the use of the MCMC sampler. Here we work interactively in the Julia REPL. You can also put your code in a script, and run it similar to the [example script](#Convenience-script). The easiest way is to adjust the [`runmcmc.jl` file](runmcmc.jl).

### Start Julia

Type 
```
julia --threads=4 
```
this starts `julia` with 4 threads, and enables you to run 4 mcmc chains in parallel. You can use another number, if you want to run fewer or more chains in parallel. This starts the Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/). For Julia 1.4 or older, you need to set the `JULIA_NUM_THREADS` environment variable, as described [here](#older-versions-of-julia).


### Install Charon.

In Julia run 
```julia
using Pkg
Pkg.add("Charon")
```
`Pkg` is the [Julia package manager](https://pkgdocs.julialang.org/v1/). It installs Julia packages and keeps track of package versions. This installs the `Charon` package to the current environment. No need to use `git clone`. `Charon` is retrieved from the [General registry](https://github.com/JuliaRegistries/General). 

You also need to install the packages [Distributions](https://juliastats.org/Distributions.jl/stable/), [CSV](https://csv.juliadata.org/stable/) and [DataFrames](https://dataframes.juliadata.org/stable/). 
```julia
using Pkg 
Pkg.add(["Distributions", "CSV", "DataFrames"])
```
`Distributions` is a software to work with probability distributions. We use it to specify priors. `CSV` is software to work with CSV files. It can load and save CSV files. `DataFrames` is software to work with data frames, comparable with `data.frame` in R, or `pandas`' `DataFrame` in Python.  

#### Hint
To minimise the risk of conflicting package versions, use a new [environment](https://pkgdocs.julialang.org/v1/environments/). You need to do that before loading the packages. You can install the packages in the new environment. If you use Julia 1.5 or newer, you can use a temporary environment with `using Pkg; Pkg.activate(; temp=true)` that last for as long as the session runs.


#### Load packages
Load the packages Charon, Distributions, CSV and DataFrames. Once you have loaded the packages, you can use it's functions. In this step the packages are precompiled. This might take a few seconds. 

In Julia run 
```julia
# load the packages. 
using Charon, Distributions, CSV, DataFrames
```
If one of them is not installed, you can install them with 
```julia
using Pkg
Pkg.add("PackageName")
```
where you replace PackageName with the name of the uninstalled package. 

#### Specify files

We assume here that the data is prepared in one of the two [prescribed data formats](#Prepare-data).

Specify the relative, or absolute paths to the basecount and frequency files. The working directory can be found with `pwd()` and can be changed with `cd("path/to/folder")`.
```julia
basecountfile = "path/to/basecountfile.csv" # unix
basecountfile = "path\\to\\basecountfile.csv" # windows
basecountfile = "C:\\path\\to\\basecountfile.csv" # absolute path windows
basecountfile = joinpath("path", "to", "basecountfile.csv") # works on all platforms
basecountfile = joinpath("C:\\", "path", "to", "basecountfile.csv") # Windows, absolute path.
```
Similarly, set the path to the frequency file. 
```julia
frequencyfile = "path/to/frequencyfile.csv" # unix
```
Alternatively, you can specify the path to the DICE file. 
```julia
dicefile = "path/to/dicefile.csv" # UNIX-systems
```

#### Specify prior 

We need to specify a prior on n, (τC, τA) and ϵ. For example, you could use a uniform prior for all three:
```julia
using Distributions # Julia package for probability distributions. 
prioronτCτA = product_distribution([Uniform(), Uniform()]) # uniform product prior [0,1]x[0,1] on (τC, τA).
prioronn = DiscreteUniform(1, 10) # discrete uniform prior on {1, 2, ..., 10}.
prioronϵ = Uniform(0, 0.5) # uniform prior on interval[0, 0.5].
```
`Uniform()` is a Julia object that represents a [uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) on [0,1]. `product_distribution([Uniform(), Uniform()])` represents a [product measure](https://en.wikipedia.org/wiki/Product_measure), where each component is a uniform distribution. `DiscreteUniform(1, 10)` is a [discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution) on the values 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. So P(k)=1/10, for k=1,...,10. `Uniform(0, 0.5)` is the uniform distribution on the interval [0,0.5]. 
Check the [Distributions documentation](https://juliastats.org/Distributions.jl/stable/) for other distributions. 


#### Specify chains and number of samples. 
Specify how many MCMC samples you want and how many chains you want to sample. For example, if you want four chains with each 100'000 samples, you can specify 
```julia
nchains = 4 
nsteps = 100_000
```


#### [Execute sampler](@id output-mcmc-sampler)
We can now execute our sampler, which might take a few hours. 
```julia
using Charon 
chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, basecountfile, frequencyfile)
```
or
```julia
using Charon 
chains = MCMCsampler(nchains, nsteps, prioronn, prioronτCτA, prioronϵ, dicefile)
```

The output of this function 
`chains` is a vector of tuples. Each tuple represents an MCMC chain. Each tuple consists of six arrays. For instance, if we want to analyse the first chain, we get 
```
nsample, τCsample, τAsample, ϵsample, accepted, logjointprob = chains[1]
```
* `nsample` is the MCMC chain for the number of individuals. 
* `τCsample` is (in this example) a 100'000x10 matrix. The k-th  column is the τC sample of the posterior conditioned on k individuals. Keep in mind that Julia has 1-based indexing.
* `τAsample` similar to `τCsample`, but now for τA.
* `ϵsample` similar to `τCsample`, but now for ϵ.
* `accepted` is a 100'000x11 matrices of true/false values. The first column indicates whether a proposal for n is accepted. The k+1-th column indicates whether a proposal for (τC, τA,ϵ) of the posterior conditioned on k individuals was accepted.
* `logjointprob` is a 100'000x11 matrices of log joint probability values. The first column indicates the log joint probability of the unconditioned posterior with `nsample[i]` individuals. The k+1-th column is the log joint probability conditioned on k individuals.  

You can obtain the unconditioned sample from the posterior as follows 
```julia
using Charon 
results = unpackposterior(chains)
```
`unpackposterior` does the following. It produces a [DataFrame](https://dataframes.juliadata.org/stable/man/working_with_dataframes/) with the unconditional posterior. Each row is an draw from the posterior. We have at at row i, for the first chain, `τC[i]=τCsample[i, nsample[i]]`, so it uses the `τC` value belonging to conditioning on `k=nsample[i]`. Similar for `τA`, and `ϵ`. It concatenates all chains, and gives each an ID, going from 1,...,4 (in this example). 


`results` is a `DataFrame` with a sample from the unconditioned posterior. It has 7 columns, which are described [here](#output-format).  



You can save the data frame as a CSV file as follows:
```julia
using CSV 
resultscsvfile = "path/to/results.csv" # place on your hard disk
# where you want to store your csv file. 
CSV.write(resultscsvfile, results)
```
You can use the DataFrame or the CSV file for further analysis.

#### Exit Julia

Type `CTRL+D` or `exit()`.
 
 ### How to transform CRAM/BAM files into CHARON's native format?

Either you use custom scripts or use [glactools](https://github.com/grenaud/glactools) and please cite glactools' paper [here](https://academic.oup.com/bioinformatics/article/34/8/1398/4665419). 
 
glactools is a tool to work with ACF (allele count format). For the allele frequency, you can simply use the ACF file of the allele counts for different populations from the 1000 Genomes project using wget:

```bash
wget ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/1000gmeld2.acf.gz 
```

or read below to learn how to make your own. Then use glactools' bam2acf on your CRAM/BAM file:

```bash
glactools bam2acf --epo /path/all_hg38withchr.epo.gz  --bed  /path/mappablechr1_22_99.bed.gz /path/GRCh38_full_analysis_set_plus_decoy_hla.fa <(samtools view -b  -T /path/GRCh38_full_analysis_set_plus_decoy_hla.fa  HG01777.alt_bwamem_GRCh38DH.20150718.IBS.low_coverage.cram )  IBS_ind1 > IBS_ind1.acf.gz
```

The file /path/all_hg38withchr.epo.gz is tab-delimited with one line per position and contains the human and ancestral allele using other great apes. It can be found here: 

```
ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/all_hg38withchr.epo.gz. 
```

The file `mappablechr1_22_99.bed.gz` is a bed file of highly mappable regions on chromosomes 1-22. It can be found: 

```
ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/mappablechr1_22_99.bed.gz
```

The file  `/path/GRCh38_full_analysis_set_plus_decoy_hla.fa` is the FASTA file of the 1000 Genomes reference. The second nested command, the "samtools view" transforms CRAM to BAM.

This will give you an ACF of your BAM file, you need to use glactools' intersect to intersect it with the 1000 Genomes allele frequency:

```
glactools intersect IBS_ind1.acf.gz 1000gmeld2.acf.gz | glactools view -h - | python3 parseCount.py  IBS_ind1
```

The script parseCount.py is found in the scripts/folder of CHARON. The script IBS_ind1 is the output prefix. It will create a IBS_ind1_basecount_IBS_ind1.gz file with the based counts and several files: IBS_ind1_freq_ACB.gz  IBS_ind1_freq_CEU.gz  IBS_ind1_freq_ESN.gz  IBS_ind1_freq_GWD.gz  IBS_ind1_freq_KHV.gz  IBS_ind1_freq_PEL.gz  IBS_ind1_freq_TSI.gz IBS_ind1_freq_ASW.gz  IBS_ind1_freq_CHB.gz  IBS_ind1_freq_FIN.gz  IBS_ind1_freq_IBS.gz  IBS_ind1_freq_LWK.gz  IBS_ind1_freq_PJL.gz  IBS_ind1_freq_YRI.gz IBS_ind1_freq_BEB.gz  IBS_ind1_freq_CHS.gz  IBS_ind1_freq_GBR.gz  IBS_ind1_freq_ITU.gz  IBS_ind1_freq_MSL.gz  IBS_ind1_freq_PUR.gz IBS_ind1_freq_CDX.gz  IBS_ind1_freq_CLM.gz  IBS_ind1_freq_GIH.gz  IBS_ind1_freq_JPT.gz  IBS_ind1_freq_MXL.gz  IBS_ind1_freq_STU.gz
which represent the allele frequencies for each population to test.




For detailed instructions and examples, refer to the `USER_GUIDE.md` file.





## How to make my own ACF file of populations?

First, use glactools' vcfm2acf to transform vcf into acf and meld individuals from the same populations together:

```bash
for i in `seq 1 22`; do  echo "glactools vcfm2acf --epo /path/all_hg38withchr.epo.gz  --fai  /path/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai   <(wget -q -O /dev/stdout  ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.chr"$i".shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz |zcat - |sed  '/^#/!s/^/chr/g')  |~/Software/glactools/glactools removepop -u  /dev/stdin HG01783 | glactools meld /dev/stdin HG02013,HG02051,HG01882,HG01894,HG01914,HG02012,HG01988,HG01990,HG01956,HG01879,HG01886,HG02107,HG02309,HG02323,HG02330,HG02334,HG02339,HG02284,HG02144,HG02549,HG02308,HG02315,HG02322,HG02283,HG02455,HG02479,HG02481,HG02485,HG02501,HG02555,HG02537,HG02429,HG02436,HG02450,HG02497,HG02505,HG02536,HG02442,HG01890,HG01915,HG01958,HG01986,HG01989,HG01912,HG02014,HG01880,HG01885,HG01883,HG02052,HG02095,HG02108,HG02009,HG02054,HG02343,HG02282,HG02307,HG02314,HG02317,HG02143,HG02256,HG02557,HG02554,HG02580,HG02420,HG02449,HG02477,HG02484,HG02489,HG02502,HG02545,HG02470,HG02427,HG02439,HG02496,HG02511,HG02010,HG01985,HG01889,HG01896,HG02111,HG02281,HG02255,HG02318,HG02325,HG02332,HG02337,HG02053,HG02471,HG02476,HG02558,HG02577,HG02508,HG02541,HG02546,HG02419,HG02433,HG02445	 ACB NA19901,NA19913,NA19920,NA19982,NA20282,NA19704,NA19711,NA19835,NA19900,NA20287,NA19917,NA20294,NA20299,NA20274,NA20314,NA20281,NA20321,NA20340,NA20357,NA19703,NA20298,NA20318,NA20320,NA20332,NA20351,NA20356,NA19712,NA19700,NA19914,NA19921,NA20276,NA19818,NA19625,NA19707,NA20334,NA20339,NA20346,NA19834,NA19904,NA19909,NA19916,NA19923,NA20127,NA20278,NA20348,NA20355,NA20362,NA20317,NA19701,NA19713,NA20359,NA20412,NA19819,NA19908,NA19922,NA19984,NA20126,NA20289,NA20291,NA20296,NA20342	 ASW HG03009,HG03585,HG03600,HG03802,HG03589,HG03604,HG03611,HG03616,HG03814,HG03793,HG03821,HG03826,HG03833,HG03832,HG04134,HG03908,HG03910,HG03922,HG04141,HG04146,HG04153,HG04158,HG04177,HG04189,HG03934,HG03941,HG03902,HG03907,HG03914,HG04195,HG04164,HG04171,HG04176,HG04183,HG04188,HG04140,HG04152,HG03919,HG03926,HG03940,HG03012,HG03007,HG03593,HG03598,HG03595,HG03603,HG03615,HG03803,HG03808,HG03815,HG03812,HG03817,HG03824,HG03829,HG03911,HG03916,HG04161,HG04173,HG04180,HG04185,HG04159,HG03928,HG03800,HG03805,HG04182,HG04194,HG03925,HG03937,HG04144,HG04156,HG03913,HG03920,HG03006,HG03607,HG03594,HG03809,HG03823,HG03830,HG03796,HG03931,HG03905,HG03917,HG04131,HG04155,HG04162,HG04186	 BEB HG00978,HG00864,HG01046,HG01798,HG01801,HG01806,HG01813,HG02188,HG02190,HG01817,HG02152,HG02164,HG02169,HG02176,HG01797,HG01800,HG01805,HG01812,HG02353,HG02358,HG02360,HG02373,HG02380,HG02385,HG02392,HG02151,HG02397,HG02156,HG02168,HG02170,HG02182,HG02187,HG02384,HG02389,HG02391,HG02396,HG02405,HG02409,HG00844,HG00851,HG01028,HG01794,HG01799,HG01802,HG01807,HG01796,HG01804,HG01809,HG01811,HG01816,HG02355,HG02367,HG02374,HG02379,HG02184,HG02181,HG02386,HG02186,HG02398,HG02401,HG02406,HG02153,HG02155,HG02165,HG02179,HG02364,HG02371,HG01029,HG01031,HG00759,HG00766,HG00867,HG00879,HG00881,HG00956,HG00982,HG02408,HG02410,HG02383,HG02390,HG02395,HG01795,HG01808,HG01810,HG01815,HG02250,HG02154,HG02166,HG02173,HG02178,HG02180,HG02185,HG02351,HG02356,HG02375,HG02382,HG02394,HG02399,HG02402,HG02407	 CDX NA12828,NA12830,NA12842,NA12873,NA12878,NA11832,NA11894,NA11919,NA11933,NA11995,NA12006,NA12044,NA12234,NA12272,NA11829,NA12342,NA11831,NA12347,NA11843,NA12400,NA11881,NA11893,NA11918,NA11920,NA12760,NA11932,NA12777,NA11994,NA12005,NA12043,NA06984,NA06989,NA12776,NA12815,NA12827,NA12872,NA12889,NA12144,NA12156,NA12283,NA12341,NA07051,NA07056,NA07347,NA07037,NA12778,NA12812,NA12829,NA12843,NA12874,NA12273,NA12348,NA12413,NA12716,NA12761,NA11830,NA11892,NA11931,NA12004,NA12155,NA12249,NA12275,NA12282,NA12287,NA06985,NA12340,NA12383,NA12489,NA12718,NA12749,NA11840,NA12751,NA12045,NA12763,NA12775,NA12814,NA06994,NA07000,NA10847,NA07048,NA12890,NA10851,NA12286,NA12399,NA12414,NA12546,NA12717,NA12748,NA12750,NA12762,NA12813,NA11930,NA11992,NA12003,NA12046,NA12058,NA12154,NA07357,NA06986	 CEU NA18528,NA18530,NA18535,NA18542,NA18547,NA18559,NA18561,NA18566,NA18573,NA18592,NA18527,NA18597,NA18534,NA18605,NA18539,NA18612,NA18541,NA18617,NA18546,NA18624,NA18553,NA18558,NA18560,NA18565,NA18572,NA18629,NA18631,NA18636,NA18643,NA18648,NA18749,NA18577,NA18591,NA18596,NA18609,NA18611,NA18616,NA18623,NA18628,NA18630,NA18635,NA18642,NA18647,NA18748,NA18606,NA18613,NA18618,NA18620,NA18625,NA18632,NA18637,NA18644,NA18740,NA18745,NA18757,NA18531,NA18536,NA18543,NA18548,NA18550,NA18555,NA18562,NA18567,NA18574,NA18579,NA18593,NA18526,NA18533,NA18538,NA18545,NA18552,NA18557,NA18564,NA18571,NA18576,NA18595,NA18603,NA18608,NA18610,NA18615,NA18622,NA18627,NA18634,NA18639,NA18641,NA18646,NA18747,NA18599,NA18602,NA18614,NA18619,NA18621,NA18626,NA18633,NA18638,NA18640,NA18645,NA18791,NA18525,NA18532,NA18537,NA18544,NA18549,NA18563,NA18570,NA18582	 CHB HG00404,HG00409,HG00428,HG00442,HG00403,HG00473,HG00410,HG00422,HG00446,HG00458,HG00472,HG00478,HG00500,HG00524,HG00531,HG00536,HG00543,HG00581,HG00530,HG00542,HG00651,HG00559,HG00656,HG00663,HG00566,HG00675,HG00580,HG00699,HG00707,HG00593,HG00598,HG00613,HG00650,HG00620,HG00625,HG00662,HG00632,HG00674,HG00693,HG00698,HG00701,HG00592,HG00629,HG00631,HG00513,HG00525,HG00537,HG00556,HG00436,HG00443,HG00448,HG00479,HG00599,HG00607,HG00614,HG00619,HG00407,HG00419,HG00626,HG00421,HG00445,HG00452,HG00457,HG00464,HG00476,HG00729,HG00657,HG00671,HG00683,HG00690,HG00708,HG00534,HG00560,HG00565,HG00584,HG00589,HG00654,HG00692,HG00705,HG00717,HG00596,HG00611,HG00623,HG00628,HG00406,HG00437,HG00449,HG00451,HG00463,HG00475,HG00533,HG00557,HG00583,HG00590,HG00653,HG00672,HG00684,HG00689,HG00704,HG00728,HG00595,HG00608,HG00610,HG00622,HG00634	 CHS HG01142,HG01250,HG01130,HG01344,HG01351,HG01122,HG01356,HG01134,HG01363,HG01139,HG01281,HG01438,HG01440,HG01348,HG01350,HG01464,HG01362,HG01254,HG01259,HG01280,HG01471,HG01488,HG01495,HG01432,HG01437,HG01444,HG01456,HG01375,HG01374,HG01468,HG01494,HG01119,HG01121,HG01133,HG01148,HG01140,HG01251,HG01345,HG01357,HG01369,HG01112,HG01124,HG01131,HG01136,HG01383,HG01390,HG01441,HG01465,HG01253,HG01256,HG01260,HG01275,HG01272,HG01277,HG01284,HG01489,HG01491,HG01342,HG01354,HG01359,HG01366,HG01474,HG01479,HG01486,HG01498,HG01378,HG01551,HG01556,HG01431,HG01443,HG01455,HG01462,HG01341,HG01353,HG01360,HG01365,HG01149,HG01113,HG01125,HG01137,HG01257,HG01269,HG01271,HG01550,HG01372,HG01377,HG01384,HG01389,HG01435,HG01447,HG01459,HG01461,HG01485,HG01492,HG01497	 CLM HG02943,HG02979,HG02981,HG03100,HG03105,HG03268,HG03270,HG02923,HG03112,HG03117,HG03124,HG03129,HG03193,HG03198,HG03136,HG03162,HG02974,HG03369,HG03511,HG03130,HG03135,HG03159,HG03109,HG03111,HG03123,HG02947,HG02973,HG03166,HG03294,HG03515,HG03352,HG03267,HG03279,HG03298,HG03301,HG03363,HG03370,HG03313,HG03351,HG02944,HG03168,HG03175,HG03199,HG02922,HG03202,HG02941,HG03118,HG03120,HG03132,HG03163,HG02968,HG02970,HG03172,HG03189,HG03196,HG02946,HG02953,HG02977,HG03517,HG03103,HG03372,HG03108,HG03115,HG03271,HG03127,HG03295,HG03303,HG03139,HG03160,HG03499,HG03514,HG03521,HG03343,HG03367,HG03280,HG03297,HG03300,HG02938,HG03114,HG03121,HG03126,HG02952,HG02971,HG02976,HG03133,HG03157,HG03099,HG03265,HG03518,HG03520,HG03366,HG03169,HG03171,HG03190,HG03195,HG03311,HG03342,HG03354,HG03291,HG03304	 ESN HG00271,HG00276,HG00288,HG00290,HG00303,HG00308,HG00310,HG00315,HG00327,HG00334,HG00339,HG00341,HG00346,HG00353,HG00358,HG00360,HG00365,HG00372,HG00182,HG00187,HG00174,HG00179,HG00181,HG00269,HG00186,HG00377,HG00384,HG00268,HG00383,HG00270,HG00275,HG00282,HG00302,HG00319,HG00321,HG00326,HG00338,HG00345,HG00357,HG00364,HG00369,HG00371,HG00376,HG00173,HG00178,HG00180,HG00171,HG00185,HG00176,HG00183,HG00188,HG00190,HG00267,HG00274,HG00281,HG00306,HG00272,HG00313,HG00277,HG00318,HG00320,HG00325,HG00332,HG00337,HG00344,HG00349,HG00351,HG00356,HG00368,HG00378,HG00375,HG00380,HG00284,HG00304,HG00309,HG00311,HG00323,HG00328,HG00330,HG00335,HG00342,HG00359,HG00361,HG00366,HG00373,HG00382,HG00177,HG00189,HG00266,HG00273,HG00278,HG00379,HG00381,HG00280,HG00285,HG00312,HG00324,HG00329,HG00331,HG00336,HG00343,HG00350,HG00355,HG00362,HG00367	 FIN HG00132,HG00137,HG00149,HG00151,HG00156,HG00136,HG00233,HG00143,HG00238,HG00148,HG00240,HG00150,HG00245,HG00155,HG00252,HG00257,HG00264,HG00232,HG00237,HG00244,HG00249,HG00251,HG00256,HG00263,HG00101,HG00106,HG00113,HG00118,HG00120,HG00125,HG00097,HG00100,HG00105,HG00112,HG00117,HG00129,HG00131,HG02215,HG00135,HG00142,HG00138,HG00154,HG00140,HG00159,HG00145,HG00152,HG00157,HG00231,HG00236,HG00243,HG00234,HG00250,HG00239,HG00255,HG00246,HG00262,HG00253,HG00258,HG00260,HG00265,HG00096,HG00104,HG00109,HG00111,HG00116,HG00123,HG00128,HG00130,HG00099,HG00102,HG00107,HG00114,HG00119,HG00121,HG00126,HG00133,HG00134,HG00139,HG00141,HG00146,HG00158,HG00160,HG00235,HG01789,HG00242,HG00254,HG00259,HG00261,HG01791,HG00103,HG00108,HG00110,HG00115,HG00122,HG00127,HG01334,HG01790,HG04301,HG04303,HG04302 GBR NA20845,NA20852,NA20864,NA20869,NA20876,NA20883,NA20908,NA20910,NA21088,NA21090,NA21095,NA20888,NA20890,NA20895,NA20903,NA21127,NA20849,NA20851,NA21141,NA20856,NA21103,NA21126,NA21108,NA21133,NA21110,NA21115,NA21122,NA21102,NA21107,NA21114,NA21119,NA21121,NA20863,NA20868,NA20870,NA20875,NA20882,NA21087,NA21094,NA21099,NA20887,NA20894,NA20899,NA20902,NA21135,NA21142,NA20891,NA20896,NA20904,NA20911,NA21109,NA21111,NA21116,NA21123,NA21128,NA21130,NA20846,NA20853,NA20858,NA20872,NA20877,NA20884,NA20889,NA21089,NA21091,NA21104,NA20850,NA21125,NA21137,NA21144,NA20862,NA20867,NA20874,NA20881,NA20886,NA20901,NA20906,NA21086,NA21093,NA21098,NA21101,NA21106,NA21113,NA21118,NA21120,NA20847,NA20854,NA20859,NA20861,NA20866,NA20892,NA20897,NA20900,NA20905,NA21092,NA21097,NA21100,NA21105,NA20873,NA20878,NA20885,NA21143,NA21112,NA21117,NA21124,NA21129	 GIH HG02568,HG02570,HG02582,HG02594,HG02804,HG02811,HG02816,HG02462,HG02703,HG02757,HG02715,HG02722,HG02769,HG02771,HG02614,HG02621,HG02461,HG02645,HG02760,HG02772,HG02676,HG02808,HG02810,HG02562,HG02574,HG02586,HG02613,HG02854,HG02620,HG02861,HG02878,HG02885,HG02675,HG02702,HG02721,HG03028,HG02839,HG02860,HG02884,HG02891,HG02896,HG03027,HG03039,HG03046,HG03539,HG02610,HG02634,HG02646,HG02759,HG02624,HG02629,HG02716,HG02643,HG02571,HG02583,HG02561,HG02588,HG02573,HG02595,HG02585,HG02805,HG02667,HG02817,HG02679,HG02836,HG02465,HG02756,HG02763,HG02768,HG02855,HG02879,HG02881,HG02982,HG03024,HG03048,HG03240,HG03040,HG03045,HG02840,HG02852,HG02888,HG02890,HG02895,HG02799,HG02807,HG02814,HG02819,HG03538,HG03247,HG03259,HG02611,HG02623,HG02628,HG02635,HG02642,HG02666,HG02678,HG02464,HG02589,HG02798,HG02813,HG02820,HG02837,HG02851,HG02983,HG03025,HG02870,HG02882,HG02887,HG03049,HG03241,HG03246,HG03258	 GWD HG01503,HG01510,HG01515,HG01522,HG01527,HG01623,HG01628,HG01630,HG01673,HG01678,HG01680,HG01685,HG01507,HG01697,HG01700,HG01519,HG01521,HG01704,HG01705,HG01709,HG01747,HG01762,HG01761,HG01766,HG01603,HG01608,HG01610,HG01615,HG01672,HG01684,HG01767,HG01779,HG01781,HG01786,HG02239,HG01773,HG01785,HG02233,HG02238,HG02219,HG02221,HG01504,HG01605,HG01612,HG01509,HG01516,HG01528,HG01530,HG01501,HG01617,HG01624,HG01619,HG01631,HG01626,HG01669,HG01679,HG01686,HG01768,HG01770,HG01602,HG01775,HG01607,HG01506,HG01513,HG01518,HG01756,HG01525,HG01537,HG01676,HG01695,HG01777,HG01784,HG01708,HG01710,HG01746,HG01765,HG02235,HG02223,HG02230,HG02220,HG02232,HG01531,HG01536,HG01512,HG01524,HG01500,HG01618,HG01620,HG01625,HG01632,HG01668,HG01670,HG01675,HG01606,HG01613,HG01682,HG01694,HG01699,HG01702,HG01707,HG01757,HG01771,HG01776,HG02236,HG02224,HG02231	 IBS HG03787,HG03869,HG03871,HG03713,HG03718,HG03720,HG03864,HG03770,HG03775,HG03782,HG03779,HG03781,HG03786,HG04076,HG04090,HG04209,HG04211,HG03717,HG04216,HG04235,HG03774,HG03863,HG03868,HG03870,HG03875,HG03882,HG03960,HG03965,HG03977,HG03729,HG03731,HG04002,HG04014,HG04019,HG04026,HG04063,HG04070,HG04094,HG04001,HG04018,HG04020,HG03969,HG03971,HG03976,HG04025,HG04056,HG04222,HG04239,HG03714,HG03788,HG03790,HG03771,HG03872,HG04200,HG04212,HG03716,HG04015,HG03730,HG04022,HG03742,HG03773,HG03778,HG03780,HG03785,HG03792,HG04060,HG03973,HG03978,HG04096,HG03862,HG03867,HG04062,HG04093,HG04098,HG04118,HG04017,HG03874,HG03963,HG03968,HG04202,HG04214,HG04219,HG04238,HG03722,HG03727,HG03772,HG03777,HG03784,HG03789,HG03967,HG03974,HG03861,HG04054,HG04059,HG04061,HG04080,HG03866,HG03873,HG04023,HG04198,HG04206,HG04225	 ITU NA18939,NA18940,NA18941,NA18945,NA18946,NA18952,NA18953,NA18957,NA18960,NA18964,NA18969,NA18971,NA18976,NA18983,NA18988,NA18990,NA18995,NA19001,NA19006,NA18965,NA18972,NA18977,NA18984,NA18989,NA18991,NA19002,NA19007,NA19056,NA19063,NA19068,NA19070,NA19057,NA19075,NA19064,NA19082,NA19076,NA19087,NA19083,NA19088,NA19090,NA18942,NA18947,NA18954,NA18959,NA18961,NA18966,NA18973,NA18978,NA18980,NA18985,NA18992,NA18997,NA19003,NA19010,NA19058,NA19060,NA19065,NA19072,NA19077,NA19084,NA19089,NA19091,NA18944,NA18949,NA18951,NA18956,NA18963,NA19055,NA19062,NA19067,NA19074,NA19079,NA19081,NA19086,NA18968,NA18970,NA18975,NA18982,NA18987,NA18994,NA18999,NA19000,NA19005,NA19012,NA19054,NA19059,NA19066,NA19078,NA19080,NA19085,NA18948,NA18950,NA18955,NA18962,NA18967,NA18974,NA18979,NA18981,NA18986,NA18993,NA18998,NA19004,NA19009,NA19011,NA18943	 JPT HG01596,HG02020,HG02025,HG02032,HG02049,HG02070,HG01595,HG02075,HG01863,HG01868,HG01870,HG02017,HG02029,HG02031,HG02048,HG02050,HG02067,HG02079,HG01862,HG01867,HG01874,HG01844,HG01849,HG01851,HG01843,HG01848,HG01850,HG01855,HG02082,HG02087,HG02081,HG02086,HG02121,HG02113,HG02133,HG02138,HG02140,HG02137,HG02513,HG02512,HG01597,HG01600,HG01599,HG02019,HG01866,HG02026,HG01873,HG01878,HG01864,HG01869,HG01871,HG02064,HG02069,HG02076,HG02040,HG02057,HG01840,HG01845,HG01852,HG01857,HG02088,HG01842,HG01847,HG01859,HG01861,HG02122,HG02127,HG02016,HG02134,HG02023,HG02139,HG02028,HG02141,HG02035,HG02047,HG02061,HG02073,HG02078,HG02131,HG02136,HG02085,HG02521,HG01598,HG01841,HG01846,HG01853,HG01858,HG01860,HG01865,HG01872,HG02084,HG02116,HG02058,HG02060,HG02128,HG02130,HG02142,HG02072,HG02522	 KHV NA19019,NA19026,NA19312,NA19317,NA19324,NA19331,NA19350,NA19355,NA19374,NA19379,NA19393,NA19398,NA19401,NA19437,NA19449,NA19451,NA19456,NA19463,NA19468,NA19475,NA19309,NA19316,NA19323,NA19328,NA19347,NA19359,NA19378,NA19380,NA19385,NA19397,NA19429,NA19431,NA19436,NA19443,NA19448,NA19455,NA19462,NA19467,NA19474,NA19020,NA19025,NA19037,NA19044,NA19038,NA19027,NA19041,NA19471,NA19375,NA19394,NA19399,NA19438,NA19440,NA19445,NA19452,NA19457,NA19308,NA19310,NA19031,NA19036,NA19043,NA19318,NA19320,NA19332,NA19351,NA19315,NA19327,NA19334,NA19346,NA19360,NA19372,NA19377,NA19384,NA19391,NA19404,NA19428,NA19430,NA19435,NA19454,NA19461,NA19466,NA19473,NA19017,NA19024,NA19023,NA19028,NA19030,NA19035,NA19042,NA19472,NA19371,NA19376,NA19383,NA19390,NA19395,NA19403,NA19434,NA19439,NA19446,NA19307,NA19314,NA19319,NA19321,NA19338	 LWK HG03078,HG03085,HG03097,HG03054,HG03061,HG03066,HG03073,HG03212,HG03224,HG03225,HG03058,HG03060,HG03376,HG03072,HG03077,HG03388,HG03084,HG03091,HG03484,HG03096,HG03410,HG03439,HG03446,HG03458,HG03460,HG03472,HG03547,HG03559,HG03578,HG03558,HG03565,HG03452,HG03457,HG03464,HG03469,HG03476,HG03394,HG03419,HG03433,HG03445,HG03572,HG03577,HG03382,HG03055,HG03074,HG03079,HG03081,HG03086,HG03478,HG03485,HG03209,HG03461,HG03052,HG03057,HG03473,HG03064,HG03069,HG03548,HG03567,HG03088,HG03095,HG03391,HG03428,HG03557,HG03442,HG03571,HG03583,HG03393,HG03398,HG03401,HG03432,HG03437,HG03449,HG03451,HG03470,HG03063,HG03082,HG03549,HG03556,HG03563,HG03479,HG03378,HG03380,HG03385,HG03397,HG03431,HG03436,HG03455,HG03462,HG03575	 MSL NA19658,NA19747,NA19759,NA19761,NA19773,NA19780,NA19785,NA19792,NA19684,NA19716,NA19652,NA19723,NA19657,NA19728,NA19664,NA19669,NA19735,NA19676,NA19746,NA19758,NA19777,NA19789,NA19722,NA19734,NA19741,NA19717,NA19729,NA19731,NA19750,NA19755,NA19762,NA19654,NA19661,NA19678,NA19774,NA19779,NA19786,NA19649,NA19651,NA19663,NA19670,NA19682,NA19719,NA19726,NA19740,NA19752,NA19764,NA19771,NA19776,NA19783,NA19788,NA19795,NA19648,NA19655,NA19679,NA19681,NA19720,NA19725,NA19732,NA19749,NA19756,NA19770,NA19782,NA19794	 MXL HG01565,HG01572,HG01577,HG01571,HG02006,HG01921,HG01926,HG01933,HG01938,HG01945,HG01951,HG01968,HG01970,HG01971,HG01976,HG01932,HG01944,HG02253,HG01893,HG02260,HG01918,HG02265,HG01920,HG01982,HG02102,HG02304,HG02272,HG02277,HG02291,HG02252,HG02271,HG01566,HG01578,HG01953,HG01979,HG01965,HG01977,HG01892,HG01991,HG02002,HG01917,HG01924,HG01927,HG01939,HG01941,HG01950,HG01967,HG01974,HG01936,HG02090,HG01948,HG02278,HG02285,HG02292,HG02146,HG02259,HG02266,HG02348,HG02275,HG02299,HG02312,HG02150,HG02345,HG02105,HG02425,HG02003,HG02008,HG01980,HG01992,HG01997,HG01923,HG01954,HG01961,HG01973,HG01935,HG01942,HG01947,HG02089,HG02104,HG02274,HG02286,HG02298,HG02262,HG02147,HG02301 PEL HG01589,HG01583,HG02784,HG02789,HG02493,HG02688,HG02733,HG02690,HG02727,HG02783,HG02652,HG02734,HG02657,HG02790,HG02601,HG02649,HG02651,HG02682,HG02687,HG02694,HG02699,HG03016,HG03237,HG03229,HG03491,HG03015,HG03022,HG03624,HG03629,HG03631,HG03636,HG03488,HG03490,HG03667,HG03706,HG03705,HG03762,HG03767,HG02603,HG02778,HG02780,HG02600,HG02691,HG02696,HG02728,HG02648,HG02792,HG02597,HG02655,HG02681,HG02658,HG02491,HG02660,HG02737,HG02684,HG02775,HG02494,HG02725,HG03019,HG03021,HG03238,HG01586,HG01593,HG02787,HG03228,HG03625,HG03235,HG03702,HG03649,HG03663,HG03668,HG03709,HG03653,HG03660,HG03634,HG02604,HG02654,HG02661,HG02685,HG02697,HG02700,HG02724,HG02731,HG02490,HG02786,HG02793,HG03018,HG02736,HG02774,HG03234,HG03619,HG03640,HG03652,HG03703,HG03708,HG03765	 PJL HG00740,HG01161,HG01173,HG01197,HG01200,HG01205,HG01248,HG00554,HG00732,HG00737,HG00637,HG01089,HG01104,HG01111,HG01177,HG01191,HG01204,HG01058,HG01242,HG01060,HG01247,HG01072,HG01088,HG01077,HG01095,HG01325,HG01108,HG01110,HG01286,HG01414,HG01312,HG01305,HG01052,HG01064,HG01069,HG01083,HG01413,HG01402,HG01393,HG01398,HG00551,HG00638,HG00731,HG00736,HG00743,HG00640,HG01094,HG01102,HG01107,HG01162,HG00553,HG01167,HG01174,HG01198,HG01326,HG01092,HG01097,HG01105,HG01047,HG01054,HG01049,HG01061,HG01051,HG01066,HG01073,HG01063,HG01080,HG01085,HG01070,HG01075,HG01082,HG01395,HG01164,HG01403,HG01171,HG01176,HG01183,HG01188,HG01190,HG00734,HG00739,HG01241,HG01302,HG01311,HG01323,HG01392,HG01405,HG01412,HG00641,HG01048,HG01055,HG01067,HG01079,HG01086,HG00742,HG01168,HG01170,HG01182,HG01187,HG01098,HG01101,HG01303,HG01308,HG01396	 PUR HG03744,HG03756,HG03888,HG03890,HG03895,HG03643,HG03679,HG03681,HG03838,HG03857,HG03686,HG03693,HG03698,HG03837,HG03844,HG03849,HG03851,HG03856,HG03697,HG03755,HG03642,HG03887,HG03953,HG03673,HG03680,HG03685,HG03692,HG03989,HG03991,HG03736,HG03743,HG03750,HG04033,HG04075,HG04099,HG04107,HG04038,HG03990,HG03995,HG04006,HG03894,HG03899,HG04210,HG03945,HG04227,HG03644,HG03757,HG03846,HG03858,HG03687,HG03694,HG03733,HG03738,HG03740,HG03745,HG03752,HG03884,HG03896,HG03836,HG03711,HG04003,HG03754,HG03646,HG04039,HG03672,HG03684,HG03947,HG03689,HG03691,HG03696,HG03985,HG03848,HG03850,HG04106,HG03944,HG03949,HG03999,HG04029,HG03886,HG03898,HG03951,HG04229,HG03690,HG03695,HG03645,HG03741,HG03746,HG03753,HG03760,HG03950,HG03955,HG03854,HG03943,HG03897,HG03900,HG04035,HG04042,HG04047,HG04100,HG03885,HG03986,HG03998	 STU NA20787,NA20799,NA20802,NA20807,NA20814,NA20819,NA20821,NA20826,NA20504,NA20509,NA20511,NA20516,NA20528,NA20530,NA20535,NA20542,NA20585,NA20756,NA20763,NA20768,NA20770,NA20775,NA20503,NA20508,NA20510,NA20515,NA20522,NA20786,NA20798,NA20801,NA20806,NA20813,NA20818,NA20832,NA20527,NA20534,NA20539,NA20541,NA20589,NA20755,NA20762,NA20767,NA20774,NA20505,NA20512,NA20517,NA20524,NA20529,NA20531,NA20536,NA20543,NA20581,NA20586,NA20752,NA20757,NA20764,NA20769,NA20771,NA20783,NA20790,NA20795,NA20803,NA20808,NA20810,NA20815,NA20822,NA20827,NA20533,NA20538,NA20540,NA20588,NA20754,NA20759,NA20761,NA20766,NA20773,NA20778,NA20785,NA20502,NA20507,NA20514,NA20519,NA20521,NA20792,NA20797,NA20800,NA20805,NA20812,NA20829,NA20831,NA20506,NA20513,NA20518,NA20520,NA20525,NA20532,NA20537,NA20544,NA20582,NA20587,NA20753,NA20758,NA20760,NA20765,NA20772,NA20796,NA20804,NA20809,NA20811,NA20816,NA20828	 TSI NA18489,NA18504,NA18511,NA18516,NA18523,NA18508,NA18510,NA18522,NA18876,NA18908,NA18910,NA18915,NA18934,NA18933,NA18864,NA18871,NA18856,NA19108,NA18488,NA19141,NA19146,NA19153,NA19160,NA19172,NA19184,NA19189,NA18868,NA18870,NA18907,NA19152,NA19171,NA19190,NA19210,NA19222,NA19239,NA19204,NA19209,NA19223,NA19235,NA19247,NA19095,NA19099,NA19102,NA19107,NA19114,NA19119,NA19121,NA19138,NA18853,NA18858,NA18505,NA18517,NA18865,NA18877,NA18909,NA18916,NA18923,NA18499,NA18502,NA18507,NA18519,NA18867,NA18874,NA18879,NA19147,NA18881,NA19159,NA19185,NA19197,NA19200,NA19207,NA19214,NA19238,NA19257,NA19236,NA19248,NA19093,NA19098,NA19096,NA19116,NA19130,NA19113,NA19118,NA19137,NA19144,NA19149,NA19175,NA18861,NA18520,NA18486,NA18501,NA19092,NA19213,NA19225,NA19256,NA19143,NA19198,NA19201,NA19206,NA18873,NA18878,NA18912,NA18917,NA18924,NA19117,NA19129,NA19131	 YRI > "1000gmeld2_$i".acf.gz"; done 
```

The file `/path/all_hg38withchr.epo.gz` is tab-delimited with one line per position and contains the human and ancestral allele using other great apes. It can be found here: 

```bash
ftp://ftp.healthtech.dtu.dk:/public/edna_allelefreq/all_hg38withchr.epo.gz
```
    
The file  `/path/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai` is the samtools faidx file of the 1000 Genomes reference. 

The command above will generate commands, one for each chromosome. You can combine them using glactools' cat:

```bash
glactools  cat   1000gmeld2_1.acf.gz 1000gmeld2_2.acf.gz 1000gmeld2_3.acf.gz 1000gmeld2_4.acf.gz 1000gmeld2_5.acf.gz 1000gmeld2_6.acf.gz 1000gmeld2_7.acf.gz 1000gmeld2_8.acf.gz 1000gmeld2_9.acf.gz 1000gmeld2_10.acf.gz 1000gmeld2_11.acf.gz 1000gmeld2_12.acf.gz 1000gmeld2_13.acf.gz 1000gmeld2_14.acf.gz 1000gmeld2_15.acf.gz 1000gmeld2_16.acf.gz 1000gmeld2_17.acf.gz 1000gmeld2_18.acf.gz 1000gmeld2_19.acf.gz 1000gmeld2_20.acf.gz 1000gmeld2_21.acf.gz 1000gmeld2_22.acf.gz > 1000gmeld2.acf.gz 
```

The last file should be identical to the one we have on our ftp.
