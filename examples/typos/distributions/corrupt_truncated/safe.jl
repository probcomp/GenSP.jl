@gen function corrupt_model_trunc(word, max_len)
    j = 0
    while (!({(:leave_be, j)} ~ bernoulli(j < max_len ? 0.95 : 1.0)))
        j += 1
        word = {(:typo, j)} ~ add_random_typo(word)
    end
    corrupted ~ dirac(word)
end

@gen function corrupt_proposal_trunc()
    word, max_len = target.args
    corrupted     = target.constraints[:corrupted]
    j = 0
    while (!({(:leave_be, j)} ~ bernoulli(j < max_len ? (word==corrupted ? 0.995 : 0.0) : 1.0)))
        j += 1
        word = {(:typo, j)} ~ guess_next_typo(word, corrupted)
    end
end

corrupt_truncated_inference = custom_importance(ChoiceMapDistribution(corrupt_proposal_trunc), 1)

corrupt_truncated = Marginal{String}(corrupt_model_trunc, corrupt_truncated_inference, :corrupted)