# Setup
@dist uniform_from_list(l) = l[uniform_discrete(1, length(l))]

blue_square = (color="blue", shape="square", string="blue square")
blue_circle = (color="blue", shape="circle", string="blue circle")
green_square = (color="green", shape="square", string="green square")

# The set of all states, S
S = [blue_square, blue_circle, green_square]

# The set of all objects as strings
O = [s.string for s in S]

# The set of all utterances, U
U = ["blue", "green", "square", "circle"]

# uninformed prior over utterances
@dist utterance_prior() = uniform_from_list(U)

# uninformed prior over world states
@dist object_prior() = uniform_from_list(O)

function meaning(utterance::String, object::String)::Bool
    occursin(utterance, object)
end

# Create a distribution representing the posterior over a single discrete latent variable
# of a Gen model, given observations of the other variables.
normalize(scores) = exp.(scores .- Gen.logsumexp(scores))
posterior_probs(model, obs, var, options) = normalize([first(Gen.assess(model, (), merge(obs, choicemap(var => o)))) for o in options])
@dist posterior(model, observation, variable_to_infer, options) = options[categorical(posterior_probs(model, observation, variable_to_infer, options))]

# If we want to model someone as using inference instead of sampling
# from the exact posterior, we can instead use `importance`:
#  importance(model, args, selection, constraints, k)

NUM_PARTICLES = 3