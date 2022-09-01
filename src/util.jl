function selection_from_choicemap(choicemap)
    selection = select()

    key_to_submap = collect(Gen.get_submaps_shallow(choicemap))
    for (key, submap) in key_to_submap
        if submap isa ValueChoiceMap
            push!(selection, key)
        else
            subselection = selection_from_choicemap(submap)
            Gen.set_subselection!(selection, key, subselection)
        end
    end
    selection
end

function logmeanexp(weights)
    logsumexp(weights) - log(length(weights))
end

export logmeanexp