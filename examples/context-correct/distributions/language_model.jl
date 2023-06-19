const unigram_freqs = Dict{String, Int}()
const bigram_freqs = Dict{String, Dict{String, Int}}()

function initialize_freqs()
    s = open("examples/context-correct/data/soap-text.txt") do f
        read(f, String);
    end;
    words = split(lowercase(s));
    prev = ""
    for word in words
        d = get!(bigram_freqs, prev) do
            Dict{String, Int}()
        end
        if haskey(d, word)
            d[word] += 1
        else
            d[word] = 1
        end
        if haskey(unigram_freqs, word)
            unigram_freqs[word] += 1
        else
            unigram_freqs[word] = 1
        end
        prev = word
    end;
    return s
end

normalize(weights) = weights ./ sum(weights)


const CHARACTERS = unique(lowercase(initialize_freqs()))
const UNIGRAM_FREQS_VEC = normalize(values(unigram_freqs))
const UNIGRAM_KEYS_VEC = collect(keys(unigram_freqs))

sample_uniform_freqs() = UNIGRAM_KEYS_VEC[categorical(UNIGRAM_FREQS_VEC)]
sample_dict(d) = collect(keys(d))[categorical(normalize(values(d)))]

function generate_word(last_word)
    if haskey(bigram_freqs, last_word) && rand() < 0.99
        options = bigram_freqs[last_word]
        return sample_dict(options)
    end
    return sample_uniform_freqs()
end

function generate_word_logpdf(last_word, this_word)
    uni = log(unigram_freqs[this_word])-log(sum(values(unigram_freqs)))
    if !haskey(bigram_freqs, last_word)
        return uni
    end
    bi = log(get(bigram_freqs[last_word], this_word, 0)) - log(sum(values(bigram_freqs[last_word])))
    return logsumexp([uni + log(0.01), bi + log(0.99)])
end

# Make generate_word its own distribution.
struct GenerateWord <: Gen.Distribution{String} end
gen_word = GenerateWord()

function Gen.random(::GenerateWord, previous_word)
    return generate_word(previous_word)
end

function Gen.logpdf(::GenerateWord, next_word, previous_word)
    return generate_word_logpdf(previous_word, next_word)
end