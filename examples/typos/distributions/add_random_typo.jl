# AddRandomTypo primitive distribution with exact density.

struct AddRandomTypo <: Gen.Distribution{String} end
const add_random_typo = AddRandomTypo()
Gen.random(::AddRandomTypo, word) = add_single_typo_sampler(word)
Gen.logpdf(::AddRandomTypo, obs, word) = add_single_typo_logpdf(word, obs)

function perform_typo_sampler(word, kind, loc)
    if kind == :insertion
        letter = rand(CHARACTERS)
        return "$(word[1:loc])$(letter)$(word[loc+1:end])"
    end
    if kind == :deletion
        return "$(word[1:loc-1])$(word[loc+1:end])"
    end
    if kind == :substitution
        letter = rand(CHARACTERS)
        return "$(word[1:loc-1])$(letter)$(word[loc+1:end])"
    end
end

function add_single_typo_sampler(word)
    if length(word) == 0
        edit_type = :insertion
    else
        edit_type = [:insertion, :deletion, :substitution][rand(1:3)]
    end
    first_loc = edit_type == :insertion ? 0 : 1
    last_loc  = length(word)
    edit_location = rand(first_loc:last_loc)
    return perform_typo_sampler(word, edit_type, edit_location)
end

function add_single_typo_logpdf(word, observed)
    L = length(word)
    M = length(observed)

    if L == 0 && M == 1
        return -log(length(CHARACTERS))
    end

    ways = 0
    if L == M
        # Substitution
        for i in 1:L
            if "$(observed[1:i-1])$(observed[i+1:end])" == "$(word[1:i-1])$(word[i+1:end])"
                ways += 1
            end
        end
        return log(ways) - log(L) - log(length(CHARACTERS)) - log(3)
    elseif L > M
        # Deletion
        for i in 1:L
            if observed == perform_typo_sampler(word, :deletion, i)
                ways += 1
            end
        end
        return log(ways) - log(L) - log(3)
    else
        # Insertion
        for i in 0:L
            if word == "$(observed[1:i])$(observed[i+2:end])"
                ways += 1
            end
        end
        return log(ways) - log(L+1) - log(length(CHARACTERS)) - log(3)
    end
end