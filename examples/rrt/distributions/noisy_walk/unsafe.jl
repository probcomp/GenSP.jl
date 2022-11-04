struct NoisyWalkUnsafe <: ProxDistribution{ChoiceMap} end

const noisy_walk_unsafe = NoisyWalkUnsafe()

function GenProx.random_weighted(::NoisyWalkUnsafe, start, dest, speed, noise, scene, planner_params, num_ticks, dt)
    # plan a path that avoids obstacles in the scene
    maybe_path = plan_path(start, dest, scene, planner_params)
    planning_failed = maybe_path === nothing

    if planning_failed
        # path planning failed; assume agent stays at start location indefinitely
        locations = fill(start, num_ticks)
    else   
        # path planning succeeded; move along the path at constant speed
        locations = walk_path(maybe_path, speed, dt, num_ticks)
    end

    # generate noisy measurements of the agent's location at each time point
    noise_matrix = noise * I + zeros(num_ticks, num_ticks)
    x_locs, y_locs = map(p -> p.x, locations), map(p -> p.y, locations)
    xs = mvnormal(x_locs, noise_matrix)
    ys = mvnormal(y_locs, noise_matrix)
    return choicemap(:xs => xs, :ys => ys), logpdf(mvnormal, xs, x_locs, noise_matrix) + logpdf(mvnormal, ys, y_locs, noise_matrix)
end

function GenProx.estimate_logpdf(::NoisyWalkUnsafe, meas, start, dest, speed, noise, scene, planner_params, num_ticks, dt)
    maybe_path = plan_path(start, dest, scene, planner_params)
    planning_failed = maybe_path === nothing

    if planning_failed
        # path planning failed; assume agent stays at start location indefinitely
        locations = fill(start, num_ticks)
    else   
        # path planning succeeded; move along the path at constant speed
        locations = walk_path(maybe_path, speed, dt, num_ticks)
    end

    noise_matrix = noise^2 * I + zeros(num_ticks, num_ticks)
    xs, ys = meas[:xs], meas[:ys]
    x_locs, y_locs = map(p -> p.x, locations), map(p -> p.y, locations)
    return logpdf(mvnormal, xs, x_locs, noise_matrix) + logpdf(mvnormal, ys, y_locs, noise_matrix)
end