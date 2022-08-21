@gen function corrupt_model(word)
    j = 0
    while (!({(:leave_be, j)} ~ bernoulli(0.95)))
        j += 1
        word = {(:typo, j)} ~ add_random_typo(word)
    end
    corrupted ~ dirac(word)
end

function corrupt_inference(corrupted, word)
    @gen function corrupt_proposal()
        j = 0
        while (!({(:leave_be, j)} ~ bernoulli(word==corrupted ? 0.995 : 0.0)))
            j += 1
            word = {(:typo, j)} ~ guess_next_typo(word, corrupted)
        end
    end
    return custom_importance(corrupt_proposal, 1)
end

corrupt = Marginal{String}(corrupt_model, corrupt_inference, :corrupted)
