function relative_variance(log_weights)
    z_estimate = logsumexp(log_weights) - log(length(log_weights))
    log_relative_weights = log_weights .- z_estimate
    return sqrt(sum((exp.(log_relative_weights) .- 1) .^ 2) / (length(log_weights) - 1))
end

function abs_variance(log_weights)
    z_estimate = logsumexp(log_weights) - log(length(log_weights))
    return sqrt(sum((exp.(log_weights) .- exp(z_estimate)) .^ 2) / (length(log_weights) - 1))
end