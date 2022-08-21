# Truncated `corrupt_trunc` distribution with exact density.

struct CorruptTruncatedExact <: Gen.Distribution{String} end
const corrupt_truncated_exact = CorruptTruncatedExact()

function Gen.random(::CorruptTruncatedExact, word, max_typos)
    local j = 0
    while (!bernoulli(j < max_typos ? 0.95 : 1))
        j += 1
        word = Gen.random(add_random_typo, word)
    end
    return word
end

function all_typos(word)
    # Enumerate all possible typos of a word: insertions, deletions, substitutions.
    typos = []

    # Insertion
    for insertion_loc in 0:length(word)
        for letter in CHARACTERS
            push!(typos, "$(word[1:insertion_loc])$(letter)$(word[insertion_loc+1:end])")
        end
    end

    # Deletion
    for deletion_loc in 1:length(word)
        push!(typos, "$(word[1:deletion_loc-1])$(word[deletion_loc+1:end])")
    end

    # Substitution
    for substitution_loc in 1:length(word)
        for letter in CHARACTERS
            push!(typos, "$(word[1:substitution_loc-1])$(letter)$(word[substitution_loc+1:end])")
        end
    end

    return typos
end

function Gen.logpdf(::CorruptTruncatedExact, corrupted, word, max_typos)
    # We need to enumerate all typo paths up to length max_typos,
    # and for each, compute the log probability of the typo path.
    # Then we need to take the logsumexp of all the log probabilities.
    logprobs = Float64[]
    # Store a queue of (word, path_logprob, j) triples.
    queue = [(word, 0.0, 0)]
    while !isempty(queue)
        (word, path_logprob, j) = pop!(queue)
        if word == corrupted
            # One way we could end up at `corrupted` is to stop now
            push!(logprobs, path_logprob + log(0.95))
        end
        if j < max_typos
            # We can add a typo to the word.
            # We need to consider all possible typos.
            for typo in all_typos(word)
                new_path_logprob = path_logprob + log(0.05) + logpdf(add_random_typo, typo, word)
                push!(queue, (typo, new_path_logprob, j+1))
            end
        end
    end
    return isempty(logprobs) ? -100 : Gen.logsumexp(logprobs)
end