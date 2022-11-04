# Represents an unnormalized target distribution as a pairing of a 
# generative function (with specific arguments) and a choicemap of constraints.
struct Target
    p :: GenerativeFunction
    args :: Tuple
    constraints :: ChoiceMap
end

"""
    latent_sel::Selection = latent_selection(target::Target)

A selection that pulls out from any ChoiceMap of target.p only the unobserved choices.
"""
function latent_selection(target::Target)
    return complement(selection_from_choicemap(target.constraints))
end

"""
    latent_choices::ChoiceMap = get_latents(target::Target, choices::ChoiceMap) -> ChoiceMap

Given a `target` and a ChoiceMap `choices` of the target's generative function,
return a ChoiceMap of only the _unobserved_ choices in `choices`.
"""
function get_latents(target::Target, choices::ChoiceMap)
    return Gen.get_selected(choices, latent_selection(target))
end

"""
    latent_choices::ChoiceMap = get_latents(target::Target, trace::Trace)

Given a `target` and a Trace `trace` of the target's generative function,
return a ChoiceMap of only the _unobserved_ choices in `trace`.
"""
function get_latents(target::Target, trace::Trace)
    return get_latents(target, Gen.get_choices(trace))
end

"""
    trace::Trace = generate(target::Target, choices::ChoiceMap)

Given a `Target` and a choice map of latents, generate a trace of the target's generative function.
"""
function generate(target::Target, choices::ChoiceMap)
    return first(Gen.generate(target.p, target.args, merge(choices, target.constraints)))
end

export Target