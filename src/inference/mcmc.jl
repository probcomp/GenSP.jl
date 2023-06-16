struct MCMC <: ProxDistribution{ChoiceMap}
    initial_distribution :: Distribution{ChoiceMap}
    kernel :: Function # Gen kernel DSL
    kernel_args :: Tuple
    target :: Union{Nothing, Target} # if nothing, will be whatever target is passed into the MCMC distribution.
end
mcmc(initial_distribution, kernel, kernel_args, target) = MCMC(initial_distribution, kernel, kernel_args, target)
mcmc(initial_distribution, kernel, kernel_args)         = MCMC(initial_distribution, kernel, kernel_args, nothing)

function random_weighted(alg::MCMC, target)
    alg_target = isnothing(alg.target) ? target : alg.target

    # Generate an initial proposal
    _, initial_weight, initial_choices = Gen.propose(alg.initial_distribution, (alg_target,))
    initial_trace = generate(alg_target, initial_choices)
    model_score = get_score(initial_trace)
    weight = model_score - initial_weight

    # Run the kernel
    moved_trace, = alg.kernel(initial_trace, alg.kernel_args...)
    moved_choices = get_choices(moved_trace)
    latents = get_latents(alg_target, moved_choices)

    # When alg.target and target are not the same, we need to 
    # adjust the weight to reflect the difference in the model score.
    if !isnothing(alg.target)
        # First, select just the latent (un-constrained) variables from moved_trace:
        # Then compute the score under the new target:
        final_trace = generate(target, latents)
        # Finally, compute the difference in the model score as a weight adjustment:
        return latents, weight + get_score(final_trace) - get_score(moved_trace)
    end

    # Otherwise, the weight is just the original weight
    return latents, weight
end

function estimate_logpdf(alg::MCMC, latents, target)
    alg_target = isnothing(alg.target) ? target : alg.target
    moved_trace = generate(alg_target, latents)
    initial_trace,  = Gen.reversal(alg.kernel)(moved_trace, alg.kernel_args...)
    initial_latents = get_latents(alg_target, initial_trace)
    initial_q,      = Gen.assess(alg.initial_distribution, alg.initial_args, ValueChoiceMap{ChoiceMap}(initial_latents))
    if isnothing(alg.target)
        return initial_q
    end
    final_trace = generate(target, latents)
    return initial_q + get_score(final_trace) - get_score(moved_trace)
end
