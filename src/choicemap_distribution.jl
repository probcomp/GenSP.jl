"""
    dist::SPDistribution{ChoiceMap} = ChoiceMapDistribution(p::GenerativeFunction, selection::Selection, custom_q::Union{Nothing, Distribution{ChoiceMap}})

Construct a new distribution `dist` over choice maps, corresponding to the marginal distribution of `selection`
under the generative function `p`. If provided, `custom_q` is used to marginalize unselected choices when estimating
densities.
"""
struct ChoiceMapDistribution <: SPDistribution{ChoiceMap}
    p         :: GenerativeFunction
    selection :: Selection
    custom_q  :: Union{Nothing, Distribution{ChoiceMap}}
end
ChoiceMapDistribution(p) = ChoiceMapDistribution(p, AllSelection(), nothing)
ChoiceMapDistribution(p, selection) = ChoiceMapDistribution(p, selection, nothing)

function random_weighted(d::ChoiceMapDistribution, args...)
    trace = simulate(d.p, args)
    selected_choices = Gen.get_selected(get_choices(trace), d.selection)
    if isnothing(d.custom_q)
        weight = Gen.project(trace, d.selection)
    else
        unselected = Gen.get_selected(get_choices(trace), complement(d.selection))
        target = Target(d.p, args, selected_choices)
        weight = Gen.get_score(trace) - Gen.assess(d.custom_q, (target,), ValueChoiceMap{ChoiceMap}(unselected))[1]
    end
    return selected_choices, weight
end

function estimate_logpdf(d::ChoiceMapDistribution, choices, args...)
    if isnothing(d.custom_q)
        _, weight = Gen.generate(d.p, args, choices)
    else
        target = Target(d.p, args, choices)
        trace = Gen.simulate(d.custom_q, (target,))
        weight = Gen.get_score(generate(target, get_retval(trace))) - Gen.get_score(trace)
    end
    return weight
end

export ChoiceMapDistribution