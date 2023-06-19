# The run_smc function takes in a model setting (:safe, :unsafe, :exact)
# and a number of particles, and returns a log likelihood estimate.
struct Model
    name    :: String
    run_smc :: Function
    computation_settings :: Vector{Tuple{Int, Int}} # (num_particles, num_replicates)
end

