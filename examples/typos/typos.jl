using Gen, GenProx

include("language_model.jl")
include("distributions/distributions.jl")

# Dynamic model -- no incremental computation
@gen function dynamic_model(n, init)
    word = init
    for i in 1:n
        word = {i => :word} ~ gen_word(word)
        {i => :obs} ~ corrupt(word)
    end
end

# Static model -- supports incremental computation
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

# Truncated model
MAX_LEN = 2

# Exact:
@gen (static) function exact_truncated_step(i, w)
    word ~ gen_word(w)
    obs ~ corrupt_truncated_exact(word, MAX_LEN)
    return word
end

# Safe:
@gen (static) function truncated_step(i, w)
    word ~ gen_word(w)
    obs ~ corrupt_truncated(word, MAX_LEN)
    return word
end

# Create the full models:
const static_model = Unfold(step)
const unsafe_static_model = Unfold(unsafe_step)
const exact_truncated_static_model = Unfold(exact_truncated_step)
const truncated_static_model = Unfold(truncated_step)

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

# Metropolis-Hastings
function typos_mh(model, sentence, steps=1000,verbose=true,init_trace=nothing)
    n = length(split(sentence))
    cm = sentence_choicemap(sentence)
    tr, = isnothing(init_trace) ? generate(model, (n,START_WORD), cm) : (init_trace,)
    accepted = 0
    lpdfs = Float64[Gen.get_score(init_trace)]
    times = Float64[0.0]
    accs = []
    for iter=1:steps
        push!(times, last(times))
        push!(accs, 0)
        for j=1:n
            this_step = @timed mh(tr, select(j => :word))
            times[end] += this_step.time
            tr, acc = this_step.value
            accepted += acc ? 1 : 0
            accs[end] += acc ? 1 : 0
        end
        # Estimate the score using `corrupt` to better handle
        # -Inf's from exact.
        push!(lpdfs, logsumexp([Gen.assess(unsafe_static_model, (n, START_WORD), get_choices(tr))[1] for _ in 1:10]) - log(10))
        if verbose && iter % 5 == 0
            println("Iter $iter / $steps")
            println([tr[i => :word] for i in 1:n])
        end
    end
    return (result=[tr[i => :word] for i in 1:n], scores=lpdfs, acc_rate=accepted / (steps * n), times=times, accs=accs)
end

# Sequential Monte Carlo
function typos_smc(model, sentence, K=5000, return_state=false)
    state = Gen.initialize_particle_filter(model, (0,START_WORD), choicemap(), K)
    words = split(sentence)
    for (i, word) in enumerate(words)
        particle_filter_step!(state, (i,START_WORD), (UnknownChange(),NoChange()), choicemap((i => :obs) => String(word)))
        maybe_resample!(state)
    end
    if return_state
        return state
    end
    tr = sample_unweighted_traces(state, 1)[1]
    return [tr[i => :word] for i in 1:length(words)], state.log_ml_est
end

