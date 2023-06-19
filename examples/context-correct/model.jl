using Gen, GenSP

include("../utils.jl")
include("distributions/language_model.jl")
include("distributions/distributions.jl")

# Truncated model
MAX_LEN = 2

# Exact:
@gen (static) function exact_truncated_step(i, w)
    word ~ gen_word(w)
    obs ~ corrupt_truncated_exact(word, MAX_LEN)
    return word
end

# Safe version:
@gen (static) function step(i, w)
    word ~ gen_word(w)
    obs ~ corrupt(word)
    return word
end

# Unsafe version:
@gen (static) function unsafe_step(i, w)
    word ~ gen_word(w)
    obs ~ corrupt_unsafe(word)
    return word
end

# Create the full models:
const exact_truncated_static_model = Unfold(exact_truncated_step)
const safe_static_model = Unfold(step)
const unsafe_static_model = Unfold(unsafe_step)

# Construct choicemap for observed sentence
function sentence_choicemap(sentence)
    words = split(sentence)
    choicemap([(i => :obs) => String(w) for (i, w) in enumerate(words)]...)
end

START_WORD = "."


# Inference

# Importance resampling
function typos_sir(model, sentence, K=10000)
    n = length(split(sentence))
    cm = sentence_choicemap(sentence)
    tr, w = importance_resampling(model, (n,START_WORD), cm, K)
    return [tr[i => :word] for i in 1:n], w
end


# Sequential Monte Carlo
function typos_smc(model, sentence, K=5000, return_state=false, resample=true)
    state = Gen.initialize_particle_filter(model, (0,START_WORD), choicemap(), K)
    #println(state)
    words = split(sentence)
    for (i, word) in enumerate(words)
        if !all(isinf, state.log_weights) && resample
            maybe_resample!(state)
        end
        particle_filter_step!(state, (i,START_WORD), (UnknownChange(),NoChange()), choicemap((i => :obs) => String(word)))
    end
    if return_state
        return state
    end
    if all(isinf, state.log_weights)
        tr = state.traces[1]
    else
        tr = sample_unweighted_traces(state, 1)[1]
    end
    return [tr[i => :word] for i in 1:length(words)], state.log_ml_est + logmeanexp(state.log_weights)
end

function run_smc_typos(num_particles, model_kind)
    model = begin
        if model_kind == :unsafe
            unsafe_static_model
        elseif model_kind == :safe
            safe_static_model
        elseif model_kind == :exact
            exact_truncated_static_model
        else
            error("Unknown model kind: $model_kind")
        end
    end

    last(typos_smc(model, "i kni that yo tink taht", num_particles))
end

typos_settings = [(1, 200), (2, 200), (3, 200), (4, 200), (5, 200), (10, 100), 
                  (25, 100), (50, 50), (100, 20), (200, 20), (300, 20), (400, 10), (500, 2)];

typos_settings = [(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (10, 5), 
    (25, 5), (50, 2), (100, 2), (200, 2)];


typos_model_config = Model("typos", run_smc_typos, typos_settings);
