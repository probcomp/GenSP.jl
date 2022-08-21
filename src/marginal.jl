# Create a new primitive distribution as 
# the marginal distribution of a particular address 
# in a larger model.
struct Marginal{T} <: ProxDistribution{T}
    p :: GenerativeFunction # generative function representing a joint distribution
    q :: Function # deterministic function mapping (val, args) to a Distribution{ChoiceMap}
    addr
end

function random_weighted(m::Marginal{T}, args...) where T
    choices, weight, = Gen.propose(m.p, args)
    val = choices[m.addr]
    other_choices = Gen.get_selected(choices, complement(select(m.addr)))

    target = Target(m.p, args, choicemap(m.addr => val))
    weight -= Gen.logpdf(m.q(val, args...), other_choices, target)
    return val, weight
end

function estimate_logpdf(m::Marginal{T}, val, args...) where T
    target = Target(m.p, args, choicemap(m.addr => val))
    _, weight, choices = Gen.propose(m.q(val, args...), (target,))
    choices = merge(choices, choicemap(m.addr => val))
    model_weight = Gen.assess(m.p, args, choices)[1]
    return isinf(model_weight) ? model_weight : model_weight - weight
end

export Marginal