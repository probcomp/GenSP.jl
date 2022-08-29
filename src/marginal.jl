# Create a new primitive distribution as 
# the marginal distribution of a particular address 
# in a larger model.
"""
    Marginal{T}(model, inference, address) <: ProxDistribution{T}

Given a `model` with a `T`-valued choice at address `address`, and a function `inference`
taking values of that choice to inference algorithms for inferring the model's latents,
produce a new `ProxDistribution{T}` representing the marginal distribution of `address`
within the generative model.
"""
struct Marginal{T} <: ProxDistribution{T}
    p :: GenerativeFunction # generative function representing a joint distribution
    q :: Distribution{ChoiceMap}
    addr
end

function random_weighted(m::Marginal{T}, args...) where T
    choices, weight, = Gen.propose(m.p, args)
    val = choices[m.addr]
    other_choices = Gen.get_selected(choices, complement(select(m.addr)))

    target = Target(m.p, args, choicemap(m.addr => val))
    q_weight, = Gen.assess(m.q, (target,), ValueChoiceMap{ChoiceMap}(other_choices))
    weight -= q_weight
    return val, weight
end

function estimate_logpdf(m::Marginal{T}, val, args...) where T
    target = Target(m.p, args, choicemap(m.addr => val))
    _, weight, choices = Gen.propose(m.q, (target,))
    choices = merge(choices, choicemap(m.addr => val))
    model_weight = Gen.assess(m.p, args, choices)[1]
    return isinf(model_weight) ? model_weight : model_weight - weight
end

export Marginal