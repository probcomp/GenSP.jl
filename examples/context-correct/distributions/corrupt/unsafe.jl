# Primitive `corrupt_unsafe` distribution with estimated density.
# Currently, only supports `logpdf` and not `random`.

struct CorruptUnsafe <: SPDistribution{String} end
corrupt_unsafe = CorruptUnsafe()

function GenSP.random_weighted(::CorruptUnsafe, word)
    # word, weight = add_typos_sampler_with_weight(args...)
    error("Not implemented")
end

function add_typos_proposal(word, observed)
    proposal_logpdf = 0
    if word == observed
        if rand() < 0.995
            proposal_logpdf += log(0.995)
            return [], proposal_logpdf
        else
            proposal_logpdf += log(0.005)
        end
    end

    orig_word, orig_observed = word, observed

    possible_moves = enumerate_typos(word, observed)

    if length(possible_moves) == 0
        with_typo = add_single_typo_sampler(orig_word)
        proposal_logpdf += add_single_typo_logpdf(orig_word, with_typo)
    else
        if rand() < 0.99
            with_typo = rand(possible_moves)
        else
            with_typo = add_single_typo_sampler(orig_word)
        end
        if with_typo in possible_moves
            proposal_logpdf += logsumexp([log(0.99) - log(length(possible_moves)), log(0.01) + add_single_typo_logpdf(orig_word, with_typo)])
        else
            proposal_logpdf += log(0.01) + add_single_typo_logpdf(orig_word, with_typo)
        end
    end

    # Recursive call
    other_typos, p = add_typos_proposal(with_typo, orig_observed)
    return [with_typo, other_typos...], p + proposal_logpdf
end

function estimate_add_typos_density(word, observed)
    proposed, q = add_typos_proposal(word, observed)
    p = 0.0
    for w in proposed
        p += log(0.05)
        p += add_single_typo_logpdf(word, w)
        word = w
    end
    p += log(0.95)
    return p - q
end

function GenSP.estimate_logpdf(::CorruptUnsafe, obs, word)
    return estimate_add_typos_density(word, obs)
end