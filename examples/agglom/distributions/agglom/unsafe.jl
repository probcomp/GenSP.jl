struct AgglomUnsafe <: ProxDistribution{Set{Set{Int}}} end
const agglom_unsafe = AgglomUnsafe()

# retained will be a sequence of merges.
function agglom_meta_inference_unsafe(observed_clustering, data, alpha, K, retained=nothing)
    particles = [initial_clustering(data) for _ in 1:K]
    weights = [0.0 for _ in 1:K]
    num_steps = length(data) - length(observed_clustering)
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

function GenProx.random_weighted(::AgglomUnsafe, data, alpha)
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
    score = agglom_meta_inference_unsafe(clustering, data, alpha, 5, merge_sequence)
    return clustering, score
end

function GenProx.estimate_logpdf(::AgglomUnsafe, clustering, data, alpha)
    return agglom_meta_inference_unsafe(clustering, data, alpha, 5)
end


# # Now for the approximate agglomerative clustering algorithm.
# initial_clustering(data) = Set([Set{Int}(i) for i in 1:length(data)])
# @gen function agglom_step(current_partition, data, alpha, predicate)
#     merge ~ random_merge(current_partition, data, alpha, predicate)
#     return apply_merge!(copy(current_partition), merge)
# end
# @gen function agglom_model(data, alpha)
#     clustering = initial_clustering(data)
#     L = length(clustering)
#     step = 1
#     while true
#         clustering = {step} ~ agglom_step(clustering, data, alpha, m -> true)
#         if length(clustering) == L
#             break
#         else
#             L = length(clustering)
#             step = step + 1
#         end
#     end
#     final_clustering ~ dirac(clustering)
#     return clustering
# end

# function agglom_meta_next_target(c, final_target)
#     Target(agglom_step, (c, final_target.args[1], final_target.args[2], m->true), choicemap())
# end

# @gen function agglom_meta_proposal(c, new_target, target)
#     data, alpha = new_target.args
#     clustering = target.constraints[:final_clustering]
#     {*} ~ agglom_step(c, data, alpha, m -> is_valid_merge(clustering, c, m))
# end

# function num_steps(target)
#     data, = target.args
#     clustering = target.constraints[:final_clustering]
#     length(data) - length(clustering) + 1
# end

# agglom_meta_inference = custom_smc(initial_clustering, agglom_meta_next_target, 
#                                     ChoiceMapDistribution(agglom_meta_proposal), num_steps, 5)

# agglom = Marginal{Set{Set{Int}}}(agglom_model, agglom_meta_inference, :final_clustering)
