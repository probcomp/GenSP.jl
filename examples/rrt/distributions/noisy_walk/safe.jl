using LinearAlgebra

@gen function noisy_walk_model(start, dest, speed, noise, scene, planner_params, num_ticks, dt)

    # plan a path that avoids obstacles in the scene
    maybe_path ~ plan_path(start, dest, scene, planner_params)
    planning_failed = maybe_path === nothing

    if planning_failed   
        # path planning failed; assume agent stays at start location indefinitely
        locations = fill(start, num_ticks)
    else   
        # path planning succeeded; move along the path at constant speed
        locations = walk_path(maybe_path, speed, dt, num_ticks)
    end

    # generate noisy measurements of the agent's location at each time point
    xs ~ mvnormal(map(p -> p.x, locations), noise * I + zeros(num_ticks, num_ticks))
    ys ~ mvnormal(map(p -> p.y, locations), noise * I + zeros(num_ticks, num_ticks))
end

noisy_walk = ChoiceMapDistribution(noisy_walk_model, select(:xs, :ys), importance(1))
