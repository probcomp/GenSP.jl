using Gen, GenSP, BenchmarkTools
include("model.jl")

# Generate example datasets by rejection sampling on certain properties:
partition1 = Set()
while length(partition1) != 3
    partition1, = GenSP.random_weighted(agglom, dataset, 1.0)
end
partition2 = copy(partition1)
while length(partition2) > 1
    partition2, = GenSP.random_weighted(agglom, dataset, 1.0)
    println(length(partition2))
end
# Benchmark:
@benchmark GenSP.estimate_logpdf(agglom, partition1, dataset, 1.0)
@benchmark GenSP.estimate_logpdf(agglom, partition2, dataset, 1.0)
@benchmark GenSP.estimate_logpdf(agglom_unsafe, partition1, dataset, 1.0)
@benchmark GenSP.estimate_logpdf(agglom_unsafe, partition2, dataset, 1.0)

partition3 ,  = GenSP.random_weighted(agglom, dataset[1:10], 1.0)
@benchmark GenSP.estimate_logpdf(agglom, partition3, dataset[1:10], 1.0)
@benchmark GenSP.estimate_logpdf(agglom_unsafe, partition3, dataset[1:10], 1.0)
@benchmark Gen.logpdf(agglom_exact, partition3, dataset[1:10], 1.0)