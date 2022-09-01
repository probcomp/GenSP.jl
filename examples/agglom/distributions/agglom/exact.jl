struct AgglomExact <: Distribution{Set{Set{Int}}} end
const agglom_exact = AgglomExact()

function Gen.random(::AgglomExact, data, alpha)
    # We begin with each datapoint in a separate cluster.
    clusters = Set{Set{Int}}()
    for i in 1:length(data)
        push!(clusters, Set(i))
    end

    # Repeatedly merge clusters, until deciding to stop
    while true
        merge = Gen.random(random_merge, clusters, data, alpha, m -> true)
        if merge == :stop
            break
        end
        apply_merge!(clusters, merge)
    end
    return clusters
end

function Gen.logpdf(::AgglomExact, clustering::Set{Set{Int}}, data, alpha)
    # To exactly compute the probability of a clustering, we need to 
    # enumerate all the sequences of merges that could have led to this 
    # clustering, and logsumexp their logprobs.
    logprobs = Float64[]

    # Compute the probability of stopping one time.
    merges = enumerate_merges(clustering)
    scores = [score_merge(m, data, alpha) for m in merges]
    stop_logprob = first(scores) - logsumexp(scores)
    
    # We create a queue of (clustering, logprob) pairs.
    # It begins with the initial clustering (every datapoint in its own cluster)
    initial_clustering = Set{Set{Int}}()
    for i in 1:length(data)
        push!(initial_clustering, Set(i))
    end
    queue = [(initial_clustering, 0.0)]

    while (length(queue) > 0)
        # Pop the next clustering from the queue.
        c, logprob = pop!(queue)
        
        if length(c) == length(clustering)
            # If we've reached the desired clustering, compute the stop
            # probability and add it to the logprobs array.
            push!(logprobs, logprob + stop_logprob)
            continue
        end

        # Otherwise consider all possible merges...
        merges = enumerate_merges(c)
        scores = [score_merge(m, data, alpha) for m in merges]
        merge_logprobs = scores .- logsumexp(scores)
        # ...and add any that are valid to the queue.
        for (m, lp) in zip(merges, merge_logprobs)
            if is_valid_merge(clustering, c, m)
                push!(queue, (apply_merge!(copy(c), m), logprob + lp))
            end
        end
    end

    # Return the logprob of the clustering.
    return logsumexp(logprobs)
end
