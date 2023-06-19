# Enumerate all possible sequences of proposals,
# and then evaluate the probability that the given particle 
# was chosen from each sequence.
# Note that if the weights themselves are random, this is not 
# enough: we need to enumerate every possible weight.

using IterTools

struct ImportanceExact <: Distribution{ChoiceMap}
    num_particles :: Int
    address :: Symbol
    proposal :: Function # Target -> Dict, values and logprobs
end

function Gen.logpdf(d::ImportanceExact, cm, target)
    proposal = d.proposal(target)
    # Enumerate all possible sequences of proposals.
    possible_values = collect(keys(proposal))
    chosen_value = cm[d.address]
    if !(chosen_value in possible_values)
        return -Inf
    end
    # Compute importance weights for each possible proposal
    #weights = Dict(v => (get_score(GenSP.generate(target, choicemap(d.address => v))) - proposal[v]) for v in possible_values)
    #chosen_weight = weights[chosen_value]
    prob = -Inf
    for q in product(fill(possible_values, d.num_particles)...)
       # println(q)
        if !(chosen_value in q)
            continue
        end
        # Compute weights for each particle
        particle_weights = [get_score(GenSP.generate(target, choicemap(d.address => v))) - proposal[v] for v in q]
        chosen_value_indices = findall(x -> x == chosen_value, q)
        choice_prob = logsumexp(particle_weights[chosen_value_indices]) - logsumexp(particle_weights)
        # Probability contributed by this trace.
        prob = logsumexp(prob, sum(proposal[k] for k in q) + choice_prob) # log(count(x -> x == chosen_value, q)) + chosen_weight - logsumexp([weights[k] for k in q]))
       # println(prob)
    end
    return prob
end