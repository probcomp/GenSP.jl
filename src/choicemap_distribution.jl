struct ChoiceMapDistribution <: ProxDistribution{ChoiceMap}
    p         :: GenerativeFunction
    selection :: Selection
    custom_q  :: Union{Nothing, GenerativeFunction}
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
        weight = Gen.get_score(trace) - Gen.assess(d.custom_q, (selected_choices, args...), unselected)[1]
    end
    return selected_choices, weight
end

function estimate_logpdf(d::ChoiceMapDistribution, choices, args...)
    if isnothing(d.custom_q)
        _, weight = generate(d.p, args, choices)
    else
        trace = simulate(d.custom_q, (choices, args...))
        weight = Gen.get_score(trace) - Gen.assess(d.p, args, merge(choices, get_choices(trace)))[1]
    end
    return weight
end

export ChoiceMapDistribution