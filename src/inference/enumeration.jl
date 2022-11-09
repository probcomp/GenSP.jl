# Enumerative inference.
# WARNING: assumes that the target has a fixed number of latent variables,
# each of which is discrete with finite, static support per target.
struct Enumeration <: ProxDistribution{ChoiceMap} 
    addresses :: Function
end

enumeration(addresses) = Enumeration(addresses)

function get_subtrace(trace::Gen.DynamicDSLTrace, address)
    return get_subtrace_from_trie(trace.trie, address)
end

function get_subtrace_from_trie(trie::Trie, address)
    if Gen.has_leaf_node(trie, address)
        return trie[address].subtrace
    end
    @assert false "get_subtrace requires that the address be the address of a call"
end

function get_subtrace_from_trie(trie::Trie, address::Pair)
    first, rest = address

    if Gen.has_leaf_node(trie, first)
        return get_subtrace(trie[first].subtrace, rest)
    end

    if Gen.has_internal_node(trie, first)
        return get_subtrace_from_trie(trie.internal_nodes[first], rest)
    end

    throw(KeyError(trie, address))
end

function get_subtrace(trace::Gen.VectorTrace, address::Int)
    return trace.subtraces[address]    
end

function get_subtrace(trace::Gen.VectorTrace, address::Pair)
    return get_subtrace(trace.subtraces[address[1]], address[2])
end

function get_support(d::Gen.DistributionTrace{T, Gen.Categorical}) where {T}
    probs = get_args(d)[1]
    return collect(1:length(probs))
end

function get_support(d::Gen.DistributionTrace{T, Gen.UniformDiscrete}) where {T}
    a, b = get_args(d)
    return collect(a:b)
end

function get_support(::Gen.DistributionTrace{T, Gen.Bernoulli}) where {T}
    return [true, false]
end

function get_support(trace)
    @assert false "Enumerative inference currently supports only discrete latent variables, given choice with trace type $(tyepof(trace))."
end

function GenProx.random_weighted(e::Enumeration, target::Target)
    # Get latent addresses
    addresses = e.addresses(target)

    # Enumerate the support of each latent choice.
    initial_trace = GenProx.generate(target, choicemap())

    @assert initial_trace isa Gen.DynamicDSLTrace "Enumerative inference currently supports only DynamicDSL generative functions."

    # Enumerate the support of each latent choice.
    supports = [get_support(get_subtrace(initial_trace, choice)) for choice in addresses]
    
    # Enumerate all possible choices.
    choicemaps = []
    weights    = Float64[]
    for assignment in Iterators.product(supports...)
        choices = choicemap([address => assignment for (address, assignment) in zip(addresses, assignment)]...)
        push!(weights, get_score(GenProx.generate(target, choices)))
        push!(choicemaps, choices)
    end

    # Select one to return, based on normalized weights
    total_weight = logsumexp(weights)
    normalized_weights = weights .- total_weight
    selected_index = categorical(exp.(normalized_weights))
    selected_choices = choicemaps[selected_index]
    return selected_choices, normalized_weights[selected_index]
end

function GenProx.estimate_logpdf(e::Enumeration, choices, target)
    # Get latent addresses
    addresses = e.addresses(target)

    tr = GenProx.generate(target, choices)
    @assert tr isa Gen.DynamicDSLTrace "Enumerative inference currently supports only DynamicDSL generative functions."

    # Enumerate the support of each latent choice.
    supports = [get_support(get_subtrace(tr, choice)) for choice in addresses]
    
    # Enumerate all possible choices.
    weights    = Float64[]
    for assignment in Iterators.product(supports...)
        choices = choicemap([address => assignment for (address, assignment) in zip(addresses, assignment)]...)
        push!(weights, get_score(GenProx.generate(target, choices)))
    end

    # Normalize
    total_weight = logsumexp(weights)
    return get_score(tr) - total_weight
end

export enumeration
