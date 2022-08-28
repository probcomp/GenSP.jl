# An algorithm is a distribution that takes a Target as input and 
# outputs a ChoiceMap.

struct CustomImportance <: ProxDistribution{ChoiceMap}
    proposal :: Distribution{ChoiceMap}
    num_particles :: Int
end
custom_importance(q, num_particles) = CustomImportance(q, num_particles)

function random_weighted(c::CustomImportance, target)
    # Generate c.num_particles particles from the proposal
    particles = [simulate(c.proposal, ()) for _ in 1:c.num_particles] # TODO: what arguments should c.proposal take?
    # Compute weights
    target_scores = [Gen.assess(target.p, target.args, merge(target.constraints, get_retval(p)))[1] for p in particles]
    weights = [target_score - get_score(p) for (target_score, p) in zip(target_scores, particles)]

    if all(isinf.(weights))
        return get_retval(particles[1]), NaN # TODO: make this weight a real estimate of the probability of this particle being returned
    end

    # Select one particle at random, based on the weights
    total_weight = logsumexp(weights)
    normalized_weights = weights .- total_weight
    average_weight = total_weight - log(c.num_particles)
    selected_particle_index = categorical(exp.(normalized_weights))
    selected_particle = particles[selected_particle_index]
    # Return the selected particle, and an estimate of Q
    return get_retval(selected_particle), target_scores[selected_particle_index] - average_weight
end

function estimate_logpdf(c::CustomImportance, choices, target)
    # Generate k-1 particles
    unchosen = [Gen.simulate(c.proposal, ()) for _ in 1:c.num_particles-1]
    # Retained proposal trace
    push!(unchosen, generate(c.proposal, (), ValueChoiceMap{ChoiceMap}(choices))[1])
    # Compute weights
    target_scores = [Gen.assess(target.p, target.args, merge(target.constraints, get_retval(p)))[1] for p in unchosen]
    weights = [target_score - get_score(p) for (target_score, p) in zip(target_scores, unchosen)]
    # Return the estimate of Q
    return last(target_scores) - logsumexp(weights) + log(c.num_particles)
end

export custom_importance