
struct GuessNextTypo <: Gen.Distribution{String} end
const guess_next_typo = GuessNextTypo()
Gen.random(::GuessNextTypo, word, obs) = guess_next_typo_rand(word,obs)
Gen.logpdf(::GuessNextTypo, with_typo, word, obs) = guess_next_typo_logpdf(with_typo, word, obs)

distance_memo_table = Dict{Tuple{String, String}, Int}()
path_memo_table = Dict{Tuple{String, String}, Vector{Symbol}}()
function distance(s1, s2)
    get!(distance_memo_table, (s1, s2)) do
        length(s1) == 0 && return length(s2)
        length(s2) == 0 && return length(s1)
        s1_p = s1[1:end-1]
        s2_p = s2[1:end-1]
        options = [distance(s1_p, s2) + 1, distance(s1, s2_p) + 1, distance(s1_p, s2_p) + (s1[end] == s2[end] ? 0 : 1)]
        i = argmin(options)
        is = [j for j in 1:3 if options[i] == options[j]]
        path_memo_table[(s1, s2)] = [:deletion, :insertion, :substitution][is]
        return options[i]
    end
end

function enumerate_typos(word, observed)
    # Compute shortest path.
    d = distance(word, observed)

    possible_moves = []
    rest_of_word = ""

    queue = [(word, observed, rest_of_word)]
    while !isempty(queue)
        (word, observed, rest_of_word) = pop!(queue)
        if length(word) == 0 && length(observed) == 0
            continue
        end
        if length(word) == 0 || (length(observed) > 0 && :insertion in path_memo_table[(word, observed)])
            push!(possible_moves, "$(word)$(observed[end])$rest_of_word")
            push!(queue, (word, observed[1:end-1], rest_of_word))
        end
        if length(observed) == 0 || (length(word) > 0 && :deletion in path_memo_table[(word, observed)])
            push!(possible_moves, "$(word[1:end-1])$rest_of_word")
            push!(queue, (word[1:end-1], observed, "$(word[end])$rest_of_word"))
        end
        if length(word) > 0 && length(observed) > 0 && :substitution in path_memo_table[(word, observed)]
            if word[end] != observed[end]
                push!(possible_moves, "$(word[1:end-1])$(observed[end])$(rest_of_word)")
            end
            push!(queue, (word[1:end-1], observed[1:end-1], "$(word[end])$rest_of_word"))
        end
    end
    return possible_moves
end

function guess_next_typo_rand(word, observed)
    possible_moves = enumerate_typos(word, observed)
    return (!isempty(possible_moves) && rand() < 0.99) ? 
            rand(possible_moves) : 
            add_single_typo_sampler(word)
end

function guess_next_typo_logpdf(with_typo, word, observed)
    possible_moves = enumerate_typos(word, observed)
    if isempty(possible_moves)
        return add_single_typo_logpdf(word, with_typo)
    end

    if with_typo in possible_moves
        Gen.logsumexp([log(0.99) - log(length(possible_moves)), log(0.01) + add_single_typo_logpdf(word, with_typo)])
    else
        log(0.01) + add_single_typo_logpdf(word, with_typo)
    end
end
