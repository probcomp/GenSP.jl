function apply_merge!(clustering, merge)
    if merge == :stop
        return clustering # No merge.
    end
    ci, cj = merge
    delete!(clustering, ci)
    delete!(clustering, cj)
    push!(clustering, union(ci, cj))
    return clustering
end

function is_valid_merge(target_clustering, current_clustering, merge)
    if merge == :stop 
        return length(target_clustering) == length(current_clustering)
    end
    # Otherwise, check if the merged clusters are together in the target clustering.
    sets = collect(target_clustering)
    findfirst(x -> first(merge[1]) in x, sets) == findfirst(x -> first(merge[2]) in x, sets)
end