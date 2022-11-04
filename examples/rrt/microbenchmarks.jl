using BenchmarkTools

include("model.jl")

NUM_SAMPLES = 500

# Arguments to the distribution
start = Point(0.1, 0.1)
speed = 0.3
noise = 0.1

for (name, dest) in [("likely", Point(0.9, 0.8)), ("unlikely", Point(0.3, 0.8))]
    println("$name destination")
    println("\tsafe")
    display(@benchmark GenProx.estimate_logpdf(noisy_walk, obs[:meas], start, dest, speed, noise, scene, planner_params, num_ticks, dt))
    
    println("\tunsafe")
    display(@benchmark GenProx.estimate_logpdf(noisy_walk_unsafe, obs[:meas], start, dest, speed, noise, scene, planner_params, num_ticks, dt))

    estimates = [GenProx.estimate_logpdf(noisy_walk_unsafe, obs[:meas], start, dest, speed, noise, scene, planner_params, num_ticks, dt) for _ in 1:NUM_SAMPLES]

    println("$name: $(relative_variance(estimates))")
    println("$name: $(abs_variance(estimates))")
end