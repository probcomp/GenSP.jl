# Produces an 'unfold'-like choicemap that has top-level keys 1, 2, ..., n.
# At each step, the choices are those of the step_proposal.
struct CustomSMC <: ProxDistribution{ChoiceMap}
    initial_state             # :: State
    step_model    :: Function # State -> Target
    step_proposal :: GenerativeFunction 
    num_steps     :: Int
    num_particles :: Int
end
custom_smc(init,step,q,n,k) = CustomSMC(init,step,q,n,k)

# here, target is over the entire trajectory
function random_weighted(c::CustomSMC, target)
    # Run SMC.
    states    = [c.initial_state for _ in 1:c.num_particles]
    particles = [choicemap() for _ in 1:c.num_particles]
    target_weights   = [0.0 for _ in 1:c.num_particles]
    weights   = [0.0 for _ in 1:c.num_particles]
    for step in 1:c.num_steps
        for i in 1:c.num_particles
            # Generate a new particle.
            particle = simulate(c.step_proposal, (states[i],))
            # Score it under the new target
            new_target   = c.step_model(states[i])
            target_score, new_state = Gen.assess(new_target.p, new_target.args, merge(get_choices(particle), new_target.constraints))
            # Compute the weight of the new particle.
            target_weights[i] += target_score
            weights[i] += target_score - get_score(particle)
            # Update the particle.
            Gen.set_submap!(particles[i], step, get_choices(particle))
            # Update the state.
            states[i] = new_state
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
    final_target_scores = [Gen.assess(target.p, target.args, merge(p, target.constraints))[1] for p in particles]
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
    states    = [c.initial_state for _ in 1:c.num_particles]
    particles = [choicemap() for _ in 1:c.num_particles]
    target_weights   = [0.0 for _ in 1:c.num_particles]
    weights   = [0.0 for _ in 1:c.num_particles]
    for step in 1:c.num_steps
        for i in 1:c.num_particles
            # Generate a new particle.
            if i == 1 # retained particle
                particle, = generate(c.step_proposal, (states[i],), get_submap(retained_particle, step))
            else
                particle = simulate(c.step_proposal, (states[i],))
            end
            # Score it under the new target
            new_target   = c.step_model(states[i])
            target_score, new_state = Gen.assess(new_target.p, new_target.args, merge(get_choices(particle), new_target.constraints))
            # Compute the weight of the new particle.
            target_weights[i] += target_score
            weights[i] += target_score - get_score(particle)
            # Update the particle.
            Gen.set_submap!(particles[i], step, get_choices(particle))
            # Update the state.
            states[i] = new_state
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
    final_target_scores = [Gen.assess(target.p, target.args, merge(p, target.constraints))[1] for p in particles]
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