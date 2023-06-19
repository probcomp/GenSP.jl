struct AgglomUnsafe <: SPDistribution{Set{Set{Int}}} end
const agglom_unsafe = AgglomUnsafe()

# retained will be a sequence of merges.
function agglom_meta_inference_unsafe(observed_clustering, data, alpha, K, retained=nothing)
    particles = [initial_clustering(data) for _ in 1:K]
    weights = [0.0 for _ in 1:K]
    num_steps = length(data) - length(observed_clustering) + 1
    for step in 1:num_steps
        # Taking the actual step
        for i in 1:K
            pred = m -> is_valid_merge(observed_clustering, particles[i], m)
            if i == 1 && !isnothing(retained)
                merge = retained[step]
            else
                merge = Gen.random(random_merge, particles[i], data, alpha, pred)
            end
            weights[i] += Gen.logpdf(random_merge, merge, particles[i], data, alpha, m -> true) - Gen.logpdf(random_merge, merge, particles[i], data, alpha, pred)
            particles[i] = apply_merge!(particles[i], merge)
        end
        # Resampling
        total = logsumexp(weights)
        logprobs = weights .- total
        ancestor_indices = [(j == 1 && !isnothing(retained)) ? 1 : categorical(exp.(logprobs)) for j in 1:K]
        particles = [copy(particles[i]) for i in ancestor_indices]
        weights = [total - log(K) for _ in 1:K]
    end
    return (logsumexp(weights) - log(K))
end

function GenSP.random_weighted(::AgglomUnsafe, data, alpha)
    clustering = initial_clustering(data)
    L = length(clustering)
    merge_sequence = []
    while true
        merge = Gen.random(random_merge, clustering, data, alpha, m -> true)
        push!(merge_sequence, merge)
        clustering = apply_merge!(copy(clustering), merge)

        if length(clustering) == L
            break
        else
            L = length(clustering)
        end
    end

    # Now perform conditional Sequential Monte Carlo to get the score.
    score = agglom_meta_inference_unsafe(clustering, data, alpha, 25, merge_sequence)
    return clustering, score
end

function GenSP.estimate_logpdf(::AgglomUnsafe, clustering, data, alpha)
    return agglom_meta_inference_unsafe(clustering, data, alpha, 25)
end