include("model.jl")

sentences = [
    ("i really appreciate your helping an old colleague", "i really apreciate your heling an odl colleague"),
    ("giving advice that he does n't take himself", "giving adsvice taht he does nt take hmself"),
    ("he was at a press conference", "he wdsa at a pres confferenc"),
    ("i know that you think that", "i kni that yo tink taht"),
    ("give us a good description", "givue us a godo ddescirition"),
    ("you help us find these killers","yusg help us find these kilerd"),
    ("i hate cell phones", "i hawt cell pohnes"),
]

# Test sentence
sentence = sentences[4][2]
n = length(split(sentence))
cm = sentence_choicemap(sentence)

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

# Initialize from the same state in each model
init_trace_safe,   = generate(static_model, (n,START_WORD), cm)
init_trace_unsafe, = generate(unsafe_static_model, (n,START_WORD), Gen.get_choices(init_trace_safe))
init_trace_exact,  = generate(exact_truncated_static_model, (n,START_WORD), Gen.get_choices(init_trace_safe))

# Generate 10 MH runs from the unsafe model
unsafe_results = [typos_mh(unsafe_static_model, sentence, 300, false, init_trace_unsafe) for _ in 1:10];
# Generate 10 MH runs from the safe model
safe_results   = [typos_mh(static_model, sentence, 300, false, init_trace_safe) for _ in 1:10];
# Generate 10 MH runs from the exact model
exact_results  = [typos_mh(exact_truncated_static_model, sentence, 300, true, init_trace_exact) for _ in 1:10];

function calculate_accept_rate_per_iteration(scores)
    acc = 0
    prev = first(scores)
    for score in scores[2:end]
       if !isapprox(score, prev, atol=1.0)
           acc += 1 
       end
       prev = score
    end
    return acc / length(scores)
end

# Print acceptance rates
println("Acceptance rate (estimated density): $(StatsBase.mean(calculate_accept_rate_per_iteration(unsafe_results[i].scores) for i in 1:10))")
println("Acceptance rate (exact density): $(StatsBase.mean(calculate_accept_rate_per_iteration(exact_results[i].scores) for i in 1:10))")