@gen function corrupt_model(word)
    j = 0
    while (!({(:leave_be, j)} ~ bernoulli(0.95)))
        j += 1
        word = {(:typo, j)} ~ add_random_typo(word)
    end
    corrupted ~ dirac(word)
end

@gen function corrupt_proposal(target)
    word,     = target.args
    corrupted = target.constraints[:corrupted]

    j = 0
    while (!({(:leave_be, j)} ~ bernoulli(word==corrupted ? 0.995 : 0.0)))
        j += 1
        word = {(:typo, j)} ~ guess_next_typo(word, corrupted)
    end
end

corrupt_inference = custom_importance(ChoiceMapDistribution(corrupt_proposal), 1)

corrupt = Marginal{String}(corrupt_model, corrupt_inference, :corrupted)
