using Gen, GenProx

include("geometry/primitives.jl")
include("geometry/planning.jl")
include("geometry/fixed_scene.jl")
include("distributions/noisy_walk/safe.jl")
include("distributions/noisy_walk/unsafe.jl")

start = Point(0.1, 0.1)
dest = Point(0.5, 0.5)
planner_params = PlannerParams(rrt_iters=600, rrt_dt=3.0,
                               refine_iters=3500, refine_std=1.)
dt = 0.1
num_ticks = 10

@gen function agent_model(
    scene::Scene, dt::Float64, num_ticks::Int, 
    planner_params::PlannerParams)

    # sample the start point of the agent from the prior
    start_x ~ uniform(0, 1)
    start_y ~ uniform(0, 1)
    start = Point(start_x, start_y)

    # sample the destination point of the agent from the prior
    dest_x ~ uniform(0, 1)
    dest_y ~ uniform(0, 1)
    dest = Point(dest_x, dest_y)

    # sample the speed from the prior
    speed ~ uniform(0.3, 1)

    noise = 0.01

    meas ~ noisy_walk(start, dest, speed, noise, scene, planner_params, num_ticks, dt)

    return dest
end;

measurements = [
    Point(0.0980245, 0.104775),
    Point(0.113734, 0.150773),
    Point(0.100412, 0.195499),
    Point(0.114794, 0.237386),
    Point(0.0957668, 0.277711),
    Point(0.140181, 0.31304),
    Point(0.124384, 0.356242),
    Point(0.122272, 0.414463),
    Point(0.124597, 0.462056),
    Point(0.126227, 0.498338)];

obs = choicemap(:meas => choicemap(:xs => map(p -> p.x, measurements), :ys => map(p -> p.y, measurements)))

