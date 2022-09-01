# Now for the approximate agglomerative clustering algorithm.
# We first need to write `agglom` as a probabilistic program.
initial_clustering(data) = Set([Set{Int}(i) for i in 1:length(data)])
@gen function agglom_step(current_partition, data, alpha, predicate)
    merge ~ random_merge(current_partition, data, alpha, predicate)
    return apply_merge!(copy(current_partition), merge)
end
@gen function agglom_model(data, alpha)
    clustering = initial_clustering(data)
    L = length(clustering)
    step = 1
    while true
        clustering = {step} ~ agglom_step(clustering, data, alpha, m -> true)
        if length(clustering) == L
            break
        else
            L = length(clustering)
            step = step + 1
        end
    end
    final_clustering ~ dirac(clustering)
    return clustering
end

function agglom_meta_next_target(c, final_target)
    Target(agglom_step, (c, final_target.args[1], final_target.args[2], m->true), choicemap())
end

@gen function agglom_meta_proposal(c, new_target, target)
    data, alpha = new_target.args
    clustering = target.constraints[:final_clustering]
    {*} ~ agglom_step(c, data, alpha, m -> is_valid_merge(clustering, c, m))
end

function num_steps(target)
    data, = target.args
    clustering = target.constraints[:final_clustering]
    length(data) - length(clustering) + 1
end

agglom_meta_inference = custom_smc(initial_clustering, agglom_meta_next_target, 
                                    ChoiceMapDistribution(agglom_meta_proposal), num_steps, 5)

agglom = Marginal{Set{Set{Int}}}(agglom_model, agglom_meta_inference, :final_clustering)
