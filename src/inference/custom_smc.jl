# Produces an 'unfold'-like choicemap that has top-level keys 1, 2, ..., n.
# At each step, the choices are those of the step_proposal.
struct CustomSMC <: ProxDistribution{ChoiceMap}
    initial_state             # Final Target -> State
    step_model    :: Function # State, Final Target -> Next Target
    step_proposal :: Distribution{ChoiceMap} # State, Next Target, Final Target -> ChoiceMap
    num_steps     :: Function # Final Target -> Int
    num_particles :: Int
end
custom_smc(init,step,q,n,k) = CustomSMC(init,step,q,n,k)

# here, target is over the entire trajectory
function random_weighted(c::CustomSMC, target)
    # Run SMC.
    init      = c.initial_state(target)
    states    = [init for _ in 1:c.num_particles]
    particles = [choicemap() for _ in 1:c.num_particles]
    target_weights   = [0.0 for _ in 1:c.num_particles]
    weights   = [0.0 for _ in 1:c.num_particles]
    N = c.num_steps(target)
    for step in 1:N
        for i in 1:c.num_particles
            # Next target
            new_target   = c.step_model(states[i])
            # Generate a new particle.
            particle = simulate(c.step_proposal, (states[i], new_target, target))
            # Score it under the new target
            new_target_trace = generate(new_target, get_retval(particle))
            # Compute the weight of the new particle.
            target_weights[i] += get_score(new_target_trace)
            weights[i]        += get_score(new_target_trace) - get_score(particle)
            # Update the particle.
            Gen.set_submap!(particles[i], step, get_retval(particle))
            # Update the state.
            states[i] = get_retval(new_target_trace)
        end

        # Resample the particles.
        total_weight = logsumexp(weights)
        normalized_weights = exp.(weights .- total_weight)
        selected_particle_indices = [categorical(normalized_weights) for _ in 1:c.num_particles]
        particles = [choicemap(particles[i]) for i in selected_particle_indices]
        target_weights = [target_weights[i] for i in selected_particle_indices]
        average_weight = total_weight - log(c.num_particles)
        weights   = [average_weight for i in selected_particle_indices]
        states    = [states[i] for i in selected_particle_indices]
    end

    # Reweight for the final target
    final_target_scores = [Gen.get_score(generate(target, p)) for p in particles]
    final_weights = [w - tw + ftw for (w, tw, ftw) in zip(weights, target_weights, final_target_scores)]
    
    if all(isinf.(final_weights))
        return (particles[1]), NaN # TODO: make this weight a real estimate of the probability of this particle being returned
    end

    # Select one particle at random, based on the weights
    total_weight = logsumexp(final_weights)
    normalized_weights = final_weights .- total_weight
    average_weight = total_weight - log(c.num_particles)
    selected_particle_index = categorical(exp.(normalized_weights))
    selected_particle = particles[selected_particle_index]

    # Return the selected particle, and an estimate of Q
    return selected_particle, final_target_scores[selected_particle_index] - average_weight
end

function estimate_logpdf(c::CustomSMC, retained_particle, target)
    # Run cSMC
    init = c.initial_state(target)
    states    = [init for _ in 1:c.num_particles]
    particles = [choicemap() for _ in 1:c.num_particles]
    target_weights = [0.0 for _ in 1:c.num_particles]
    weights   = [0.0 for _ in 1:c.num_particles]
    N = c.num_steps(target)
    for step in 1:N
        for i in 1:c.num_particles
            # New target
            new_target   = c.step_model(states[i])
            # Generate a new particle.
            if i == 1 # retained particle
                particle, = Gen.generate(c.step_proposal, (states[i], new_target, target), ValueChoiceMap{ChoiceMap}(get_submap(retained_particle, step)))
            else
                particle = Gen.simulate(c.step_proposal, (states[i], new_target, target))
            end
            # Score it under the new target
            new_target_trace = generate(new_target, get_retval(particle))
            # Compute the weight of the new particle.
            target_weights[i] += get_score(new_target_trace)
            weights[i] += get_score(new_target_trace) - get_score(particle)
            # Update the particle.
            Gen.set_submap!(particles[i], step, get_choices(particle))
            # Update the state.
            states[i] = get_retval(new_target_trace)
        end

        # Resample the particles.
        normalized_weights = exp.(weights .- logsumexp(weights))
        selected_particle_indices = [1, [categorical(normalized_weights) for _ in 1:c.num_particles-1]...]
        particles = [choicemap(particles[i]) for i in selected_particle_indices]
        target_weights = [target_weights[i] for i in selected_particle_indices]
        weights   = [weights[i] for i in selected_particle_indices]
        states    = [states[i] for i in selected_particle_indices]
    end

    # Reweight for the final target
    final_target_scores = [Gen.get_score(generate(target, p)) for p in particles]
    final_weights = [w - tw + ftw for (w, tw, ftw) in zip(weights, target_weights, final_target_scores)]
    
    if all(isinf.(final_weights))
        return NaN # TODO: make this weight a real estimate of the probability of this particle being returned
    end

    # Select one particle at random, based on the weights
    average_weight = logsumexp(final_weights) - log(c.num_particles)

    # Return the estimate of Q
    return final_target_scores[1] - average_weight
end

export custom_smc