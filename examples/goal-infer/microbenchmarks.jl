using BenchmarkTools, GenSP, Gen

include("model.jl")

NUM_SAMPLES = 500

# Arguments to the distribution
start = Point(0.1, 0.1)
speed = 0.3
noise = 0.02

measurements = [
    Point(0.0980245, 0.124775),
    Point(0.113734, 0.140773),
    Point(0.100412, 0.185499),
    Point(0.1794, 0.237386),
    Point(0.0957668, 0.267711),
    Point(0.140181, 0.31304),
    Point(0.124384, 0.326242),
    Point(0.122272, 0.414463),
    Point(0.124597, 0.482056),
    Point(0.126227, 0.498338)];
obs = choicemap(:meas => choicemap(:xs => map(p -> p.x, measurements), :ys => map(p -> p.y, measurements)))

for (name, dest) in [("likely", Point(0.9, 0.8)), ("unlikely", Point(0.3, 0.8))]
    println("$name destination")
    println("\tsafe")
    display(@benchmark GenSP.estimate_logpdf(noisy_walk, obs[:meas], start, dest, speed, noise, scene, planner_params, num_ticks, dt))
    
    println("\tunsafe")
    display(@benchmark GenSP.estimate_logpdf(noisy_walk_unsafe, obs[:meas], start, dest, speed, noise, scene, planner_params, num_ticks, dt))

    estimates = [GenSP.estimate_logpdf(noisy_walk_unsafe, obs[:meas], start, dest, speed, noise, scene, planner_params, num_ticks, dt) for _ in 1:NUM_SAMPLES]
end