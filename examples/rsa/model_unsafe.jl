using Gen, GenSP
using BenchmarkTools

include("nested_infer_unsafe.jl")
include("shared.jl")
@gen function speaker(depth, object)
    # Prior on utterances
    valid_utterances = [u for u in U if meaning(u, object)]
    utterance ~ uniform_from_list(valid_utterances)
    
    if depth == 1
        understood_object ~ object_prior()
        return utterance
    end

    # Model how our utterance is understood by the listener
    understood_object ~ importance_unsafe(listener, (depth-1,), :object, Gen.select(:imagined_utterance), choicemap(:imagined_utterance => utterance), NUM_PARTICLES)
end

@gen function listener(depth)
    object ~ object_prior()
    imagined_utterance ~ importance_unsafe(speaker, (depth, object), :utterance, Gen.select(:understood_object), choicemap(:understood_object => object), NUM_PARTICLES)
end

println("Unsafe")

# Depth 1
println("\t Depth 1")
@benchmark GenSP.estimate_logpdf(importance_unsafe, "green square", 
    listener, (1,), :object, Gen.select(:imagined_utterance), choicemap(:imagined_utterance => "square"), NUM_PARTICLES)

# Depth 2
println("\t Depth 2")
@benchmark GenSP.estimate_logpdf(importance_unsafe, "blue square",
    listener, (2,), :object, Gen.select(:imagined_utterance), choicemap(:imagined_utterance => "square"), NUM_PARTICLES)

