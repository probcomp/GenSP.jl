struct MCMC <: ProxDistribution{ChoiceMap}
    initial_distribution :: Distribution{ChoiceMap}
    initial_args :: Tuple
    kernel :: Function # Gen kernel DSL
    kernel_args :: Tuple
    target :: Target
end
mcmc(initial_distribution, initial_args, kernel, kernel_args, target) = MCMC(initial_distribution, initial_args, kernel, kernel_args, target)

function random_weighted(alg::MCMC, target)
    # Generate an initial proposal
    _, initial_weight, initial_choices = Gen.propose(alg.initial_distribution, alg.initial_args)
    initial_trace, model_score = generate(alg.target.p, alg.target.args, merge(initial_choices, alg.target.constraints))
    weight = model_score - initial_weight

    # Run the kernel
    moved_trace, = kernel(initial_trace, alg.kernel_args...)
    moved_choices = get_choices(moved_trace)

    # When alg.target and target are not the same, we need to 
    # adjust the weight to reflect the difference in the model score.
    # TODO: avoid this work when alg.target and target *are* the same.
    # First, select just the latent (un-constrained) variables from moved_trace:
    latents = Gen.get_selected(moved_choices, complement(selection_from_choicemap(alg.target.constraints)))
    # Then compute the score under the new target:
    final_trace = Gen.generate(target.p, target.args, merge(latents, target.constraints))
    # Finally, compute the difference in the model score as a weight adjustment:
    return latents, initial_weight + get_score(final_trace) - get_score(moved_trace)
end

function estimate_logpdf(alg::MCMC, latents, target)
    final_trace = Gen.generate(target.p, target.args, merge(latents, target.constraints))
    moved_trace = Gen.generate(alg.target.p, alg.target.args, merge(latents, alg.target.constraints))
    initial_trace,  = Gen.reversal(alg.kernel)(moved_trace, alg.kernel_args...)
    initial_latents = Gen.get_selected(get_choices(initial_trace), complement(selection_from_choicemap(alg.target.constraints)))
    initial_q,      = Gen.assess(alg.initial_distribution, alg.initial_args, ValueChoiceMap{ChoiceMap}(initial_latents))
    return initial_q + get_score(final_trace) - get_score(moved_trace)
end