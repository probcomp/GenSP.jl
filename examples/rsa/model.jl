using Gen, GenSP
using BenchmarkTools

include("nested_infer_exact.jl")
include("shared.jl")

# If we want to model someone as using inference instead of sampling
# from the exact posterior, we can instead use `importance`:
#  importance(model, args, selection, constraints, k)

NUM_PARTICLES = 3

UNIFORM_OBJECT_PROPOSAL = target -> Dict(o => -log(length(O)) for o in O)

UNIFORM_VALID_UTTERANCE_PROPOSAL = target -> begin
  valid_utterances = [u for u in U if meaning(u, target.args[2])]
  Dict(u => -log(length(valid_utterances)) for u in valid_utterances)
end

APPROXIMATE_ALG_SAFE  = importance(NUM_PARTICLES)

# The listener models the speaker as using importance sampling to infer an utterance.
LISTENER_ALG_EXACT = ImportanceExact(NUM_PARTICLES, :utterance, UNIFORM_VALID_UTTERANCE_PROPOSAL)

# The speaker models the listener as using importance sampling to infer an object.
SPEAKER_ALG_EXACT = ImportanceExact(NUM_PARTICLES, :object, UNIFORM_OBJECT_PROPOSAL)

@gen function speaker(depth, object)
    # Prior on utterances
    valid_utterances = [u for u in U if meaning(u, object)]
    utterance ~ uniform_from_list(valid_utterances)
    
    if depth == 1
        # understood_object ~ object_prior()
        listener_thought ~ ChoiceMapDistribution(listener)(depth-1)
        return utterance
    end

    # Model how our utterance is understood by the listener
    listener_thought ~ SPEAKER_ALG(Target(listener, (depth-1,), choicemap(:speaker_thought => choicemap(:utterance => utterance))))
    #understood_object ~ ALG(Target(listener, (depth-1,), choicemap(:imagined_utterance => choicemap(:utterance => utterance))))
    #understood_object ~ default_importance_v(listener, (depth-1,), :object, select(:imagined_utterance), choicemap(:imagined_utterance => utterance), NUM_PARTICLES)
end

@gen function listener(depth)
    object ~ object_prior()
    if depth > 0
        speaker_thought ~ LISTENER_ALG(Target(speaker, (depth, object), choicemap(:listener_thought => choicemap(:object => object))))
    end
    # imagined_utterance ~ default_importance_v(speaker, (depth, object), :utterance, select(:understood_object), choicemap(:understood_object => object), NUM_PARTICLES)
end

# Benchmarking

# Depth 1:
desired_listener_thought = choicemap(:object => "green square")
listener_inference_problem = Target(listener, (1,), choicemap(:speaker_thought => choicemap(:utterance => "square")))

println("Safe:")
SPEAKER_ALG = APPROXIMATE_ALG_SAFE
LISTENER_ALG = APPROXIMATE_ALG_SAFE
display(@benchmark GenSP.estimate_logpdf(SPEAKER_ALG, desired_listener_thought, listener_inference_problem))

println("Exact:")
SPEAKER_ALG = SPEAKER_ALG_EXACT
LISTENER_ALG = LISTENER_ALG_EXACT
display(@benchmark GenSP.logpdf(SPEAKER_ALG, desired_listener_thought, listener_inference_problem))


# Depth 2:
desired_listener_thought = choicemap(:object => "blue square")
listener_inference_problem = Target(listener, (2,), choicemap(:speaker_thought => choicemap(:utterance => "square")))

println("Safe:")
SPEAKER_ALG = APPROXIMATE_ALG_SAFE
LISTENER_ALG = APPROXIMATE_ALG_SAFE
display(@benchmark GenSP.estimate_logpdf(SPEAKER_ALG, desired_listener_thought, listener_inference_problem))

println("Exact:")
SPEAKER_ALG = SPEAKER_ALG_EXACT
LISTENER_ALG = LISTENER_ALG_EXACT
display(@benchmark GenSP.logpdf(SPEAKER_ALG, desired_listener_thought, listener_inference_problem))