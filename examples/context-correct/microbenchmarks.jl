using Gen, GenSP, BenchmarkTools

include("distributions/language_model.jl")
include("distributions/distributions.jl")
include("model.jl")

test_cases = ["confferenc" => "conference", "phones" => "pohnes", "friend" => "phones", "oodl" => "old"]
edit_dists = [2, 2, 5, 3]
max_edit_dists = [2, 2, 2, 3]

NUM_SAMPLES = 10000

# Corrupt: safe vs. unsafe
println("corrupt: safe vs. unsafe")
for ((s1, s2), dist) in zip(test_cases, edit_dists)
    println("$(length(s1)) letters, edit distance $dist")
    println("\tsafe")
    display(@benchmark GenSP.estimate_logpdf(corrupt, $s1, $s2))
    println("\tunsafe")
    display(@benchmark GenSP.estimate_logpdf(corrupt_unsafe, $s1, $s2))
end

# Truncated corrupt: exact, safe, unsafe.
println("truncated: exact vs. safe vs. unsafe")
for ((s1, s2), dist, max_dist) in zip(test_cases, edit_dists, max_edit_dists)
    println("$(length(s1)) letters, edit distance $dist, max dist $max_dist")
    println("\texact")
    display(@benchmark Gen.logpdf(corrupt_truncated_exact, $s1, $s2, $max_dist))
    println("\tsafe")
    display(@benchmark GenSP.estimate_logpdf(corrupt_truncated, $s1, $s2, $max_dist))
    println("\tunsafe")
    display(@benchmark GenSP.estimate_logpdf(corrupt_truncated_unsafe, $s1, $s2, $max_dist))
end