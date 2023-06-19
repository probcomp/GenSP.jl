using LinearAlgebra

noisy_walk_exact_enough = ChoiceMapDistribution(noisy_walk_model, select(:xs, :ys), importance(40))
