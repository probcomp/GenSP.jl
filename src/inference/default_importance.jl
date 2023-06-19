struct DefaultImportance <: SPDistribution{ChoiceMap}
    num_particles :: Int
end
importance(num_particles) = DefaultImportance(num_particles)

struct NoResult <: ChoiceMap end

# Returns log(1/N).
function estimate_zero_weight_prob(target, twoloops=false)
    i = 1
    while !isinf(Gen.generate(target.p, target.args, target.constraints)[2])
        i += 1
    end
    if twoloops
        while !isinf(Gen.generate(target.p, target.args, target.constraints)[2])
            i += 1
        end
    end
    return -log(i)
end

function random_weighted(alg::DefaultImportance, target)
    # Generate alg.num_particles particles
    particles = Vector{Trace}(undef, alg.num_particles)
    weights   = Vector{Float64}(undef, alg.num_particles)
    
    Threads.@threads for i in 1:alg.num_particles
        particles[i], weights[i] = Gen.generate(target.p, target.args, target.constraints)
    end
    
    if all(isinf.(weights))
        # How to compute a weight for this case?
        # Meta-inference needs to do rejection sampling
        # to propose a sequence of particles that are all
        # zero weight. Then we should get "\prod full_q(xi) / rej_q(xi)" 
        # as the weight, and since rej_q = full_q / prob(zero_weight)
        # this is just prob(zero_weight)^K.
        # For rejq, simulate generates some rejections and 
        # an accepted value, then does it again, and returns the 
        # (total number of rejections + 1) * q(x). 
        # Assess runs rejection sampling,
        # and returns q(x) * N_samples. 
        # What we want: assess rejq of each particle.
        # -sum(log(N_i) for N_i in rej_loops).

        return NoResult(), sum(estimate_zero_weight_prob(target) for _ in 1:alg.num_particles)
        # return get_latents(target, particles[1]), NaN # TODO: make this weight a real estimate of the probability of this particle being returned
    end

    # Select one particle at random, based on the weights
    total_weight = logsumexp(weights)
    normalized_weights = weights .- total_weight
    average_weight = total_weight - log(alg.num_particles)
    selected_particle_index = categorical(exp.(normalized_weights))
    selected_particle = particles[selected_particle_index]
    # Return the selected particle, and an estimate of Q
    return get_latents(target, selected_particle), get_score(selected_particle) - average_weight
end

# TODO: make this support the case where choices is NoResult(),
# because all particles were zero-weight. In this case, we need to
# simulate(rejq) K times, and compute the product of 1/thatN's
function estimate_logpdf(alg::DefaultImportance, choices, target)
    if choices isa NoResult
        return sum(estimate_zero_weight_prob(target, true) for _ in 1:alg.num_particles)
    end

    # Generate k-1 particles
    weights = Vector{Float64}(undef, alg.num_particles)
    Threads.@threads for i in 1:(alg.num_particles-1)
        _, weights[i] = Gen.generate(target.p, target.args, target.constraints)
    end
    # Retained proposal trace
    retained_trace, _ = Gen.generate(target.p, target.args, merge(choices, target.constraints))
    constrained_choices = selection_from_choicemap(target.constraints)
    retained_weight   = Gen.project(retained_trace, constrained_choices)
    weights[alg.num_particles] = retained_weight
    # Compute weights
    if isinf(retained_weight) # If the retained particle itself has zero weight, then the probability of returning it is zero.
        return -Inf
    end
    # Return the estimate of Q
    return Gen.get_score(retained_trace) - logsumexp(weights) + log(alg.num_particles)
end

export importance
