using Gen, GenSP
using Gen: select

include("../utils.jl")
include("geometry/primitives.jl")
include("geometry/planning.jl")
include("geometry/fixed_scene.jl")
include("distributions/noisy_walk/safe.jl")
include("distributions/noisy_walk/unsafe.jl")
include("distributions/noisy_walk/exact_enough.jl")

start = Point(0.1, 0.1)
dest = Point(0.5, 0.5)
planner_params = PlannerParams(rrt_iters=600, rrt_dt=3.0,
                               refine_iters=3500, refine_std=1.)
dt = 0.1
num_ticks = 10

function get_agent_model(model_kind)
    noisy_walk_dist = begin
        if model_kind == :safe
            noisy_walk
        elseif model_kind == :unsafe
            noisy_walk_unsafe
        elseif model_kind == :exact
            noisy_walk_exact_enough
        else
            error("Unknown model kind: $model_kind")
        end
    end
    return (@gen function agent_model(
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

        noise = 0.1

        meas ~ noisy_walk_dist(start, dest, speed, noise, scene, planner_params, num_ticks, dt)

        return dest
    end)
end

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

bad_measurements = [
    Point(rand(), rand()) for i in 1:3
]

obs = choicemap(:meas => choicemap(:xs => map(p -> p.x, bad_measurements), :ys => map(p -> p.y, bad_measurements)))

function run_rrt_smc(num_particles, model_kind)
    agent_model = get_agent_model(model_kind)
    ARGS = (scene, dt, 3, planner_params);
    _, wts, lml = importance_sampling(agent_model, ARGS, obs, num_particles)
    return exp.(wts .+ log(num_particles)), lml
    #return last(importance_sampling(agent_model, ARGS, obs, num_particles))
end

settings = [(1, 1000), (2, 1000), (3, 1000), (4, 1000), (5, 1000), (10, 1000), (25, 500), (50, 200), (100, 100), (1000, 10), (10000, 2)];
rrt_model = Model("rrt", run_rrt_smc, settings)