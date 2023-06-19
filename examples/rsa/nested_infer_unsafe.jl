using Gen, GenSP

# A version of the `importance` primitive that pulls out an address of interest

struct ImportanceUnsafe <: SPDistribution{Any} end

const importance_unsafe = ImportanceUnsafe()

function normalized_probs(weights)
    ps = exp.(weights .- maximum(weights))
    return ps ./ sum(ps)
end

function GenSP.random_weighted(::ImportanceUnsafe, model, args, addr, constrained_selection, constraints, k)
    particles = [generate(model, args, constraints) for _ in 1:k]
    tr = particles[categorical(normalized_probs(map(last, particles)))][1]
    return tr[addr], get_score(tr) - logsumexp(map(last, particles)) + log(k)
end

function GenSP.estimate_logpdf(::ImportanceUnsafe, choice, model, args, addr, constrained_selection, constraints, k, verbose=false)
    # Generate k-1 particles from the proposal, recording weights
    particles = [generate(model, args, constraints) for _ in 1:k-1]
    weights = map(last, particles) #[last(generate(model, args, constraints)) for _ in 1:k-1]
    # Compute weight of the given trace
    if verbose
        println(weights)
    end
    new_constraints = merge(choicemap(addr => choice), constraints)
    if verbose
        display(new_constraints)
    end
    trace, w = generate(model, args, merge(choicemap(addr => choice), constraints))
    chosen_weight = project(trace, constrained_selection)
    if verbose 
        println(w)
        println(chosen_weight)
    end
    # Return the estimate of Q
    if all(isinf, weights) && isinf(chosen_weight)
        # TODO: model the fact that we will return SOMETHING,
        # i.e. a uniform-at-random choice from the particles.
        # But for now, just return -Inf.
        return -Inf
        # Estimate accept probability
        # n = 0
        # while isinf(get_score(trace))
        #     trace, = generate(model, args, constraints)
        #     n += 1
        # end
        # log_reject_prob_estimate = log1p(-1/n)
        # # Probability of generating K-1 rejects, and the given 
        # # trace, and then choosing it uniformly
        # (k-1)*log_accept_prob_estimate + 
    end
    return get_score(trace) - logsumexp([weights..., chosen_weight]) + log(k)
end