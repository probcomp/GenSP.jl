struct DefaultImportance <: ProxDistribution{ChoiceMap}
    num_particles :: Int
end
importance(num_particles) = DefaultImportance(num_particles)

function random_weighted(alg::DefaultImportance, target)
    # Generate alg.num_particles particles
    particles = [generate(target.p, target.args, target.constraints) for _ in 1:alg.num_particles]
    # Compute weights
    weights = map(last, particles)

    if all(isinf.(weights))
        return get_choices(particles[1]), NaN # TODO: make this weight a real estimate of the probability of this particle being returned
    end

    # Select one particle at random, based on the weights
    total_weight = logsumexp(weights)
    normalized_weights = weights .- total_weight
    average_weight = total_weight - log(alg.num_particles)
    selected_particle_index = categorical(exp.(normalized_weights))
    selected_particle = particles[selected_particle_index][1]
    # Return the selected particle, and an estimate of Q
    return get_choices(selected_particle), get_score(selected_particle) - average_weight
end

function estimate_logpdf(alg::DefaultImportance, choices, target)
    # Generate k-1 particles
    unchosen = [Gen.generate(target.p, target.args, target.constraints) for _ in 1:alg.num_particles-1]
    # Retained proposal trace
    retained_trace, _ = Gen.generate(target.p, target.args, merge(choices, target.constraints))
    constrained_choices = selection_from_choicemap(target.constraints)
    retained_weight   = Gen.project(retained_trace, constrained_choices)
    push!(unchosen, (retained_trace, retained_weight))
    # Compute weights
    weights = map(last, unchosen)
    # Return the estimate of Q
    return Gen.get_score(retained_trace) - logsumexp(weights) + log(alg.num_particles)
end

export importance