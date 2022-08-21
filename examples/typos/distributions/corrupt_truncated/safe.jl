@gen function corrupt_model_trunc(word, max_len)
    j = 0
    while (!({(:leave_be, j)} ~ bernoulli(j < max_len ? 0.95 : 1.0)))
        j += 1
        word = {(:typo, j)} ~ add_random_typo(word)
    end
    corrupted ~ dirac(word)
end


function corrupt_inference_trunc(corrupted, word, max_len)
    @gen function corrupt_proposal()
        j = 0
        while (!({(:leave_be, j)} ~ bernoulli(j < max_len ? (word==corrupted ? 0.995 : 0.0) : 1.0)))
            j += 1
            word = {(:typo, j)} ~ guess_next_typo(word, corrupted)
        end
    end
    return custom_importance(corrupt_proposal, 1)
end

corrupt_truncated = Marginal{String}(corrupt_model_trunc, corrupt_inference_trunc, :corrupted)