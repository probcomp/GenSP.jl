struct ParticleCollection
    particles :: Vector{Trace}
    weights   :: Vector{Float64}
    lml_est   :: Float64
end

function log_marginal_likelihood(particles::ParticleCollection)
    return particles.lml_est + logsumexp(particles.weights) - log(length(particles.particles))
end


"""
An `SMCAlgorithm` supports three methods:
- `num_particles(alg)`: returns the number of particles this algorithm produces in its final particle collection
- `final_target(alg)`: returns the `Target` distribution for which particles are properly weighted
- `run_smc(alg, retained=nothing)`: runs the algorithm, returning a `ParticleCollection`.
  - `retained` is an optional argument that can be used to specify a ChoiceMap of latents for `final_target(alg)`.
    If `retained` is specified, the algorithm will run a *conditional* sequential Monte Carlo version of the algorithm,
    ensuring that the final particle in the collection agrees with the retained particle.

"""
abstract type SMCAlgorithm <: SPDistribution{ChoiceMap} end


"""
`SMCResample` runs an existing `SMCAlgorithm` but appends a multinomial resampling step.
The `how_many` parameter controls the number of particles in the resulting collection.
"""
struct SMCResample <: SMCAlgorithm
    previous :: SMCAlgorithm
    ess_threshold :: Float64 # a value of 1 will always resample
    how_many :: Int
end

num_particles(algorithm::SMCResample) = algorithm.how_many
final_target(algorithm::SMCResample) = final_target(algorithm.previous)
function run_smc(algorithm::SMCResample, retained=nothing)
    collection = run_smc(algorithm.previous, retained)
    num_particles = length(collection.particles)
    total_weight = logsumexp(collection.weights)
    log_normalized_weights = collection.weights .- total_weight
    if Gen.effective_sample_size(log_normalized_weights) > algorithm.ess_threshold * num_particles
        return collection
    end

    # Otherwise we have to resample
    normalized_weights = exp.(log_normalized_weights)
    selected_particle_indices = [categorical(normalized_weights) for _ in 1:algorithm.how_many]
    if !isnothing(retained)
        selected_particle_indices[end] = num_particles
    end
    particles = [collection.particles[i] for i in selected_particle_indices]
    weights   = zeros(algorithm.how_many)
    avg_weight = total_weight - log(num_particles)
    return ParticleCollection(particles, weights, avg_weight + collection.lml_est)
end

"""
`SMCClone` grows the number of particles in an existing `SMCAlgorithm` by a factor of `how_many`,
by cloning each particle produced by the existing algorithm `factor` times.
"""
struct SMCClone <: SMCAlgorithm
    previous :: SMCAlgorithm
    factor   :: Int
end

num_particles(algorithm::SMCClone) = algorithm.factor * num_particles(algorithm.previous)
final_target(algorithm::SMCClone) = final_target(algorithm.previous)
function run_smc(algorithm::SMCClone, retained=nothing)
    collection = run_smc(algorithm.previous, retained)
    particles = []
    weights = []
    for (particle, weight) in zip(collection.particles, collection.weights)
        for _ in 1:algorithm.factor
            push!(particles, particle)
            push!(weights, weight)
        end
    end
    return ParticleCollection(particles, weights, collection.lml_est)
end

"""
`SMCInit` initializes an SMC algorithm with a fixed number of particles, using 
a user-specified proposal distribution `q`. Particles are weighted for a user-specified
target distribution.
"""
struct SMCInit <: SMCAlgorithm
    q :: Distribution{ChoiceMap}
    target :: Target
    num_particles :: Int
end

num_particles(algorithm::SMCInit) = algorithm.num_particles
final_target(algorithm::SMCInit) = algorithm.target
function run_smc(algorithm::SMCInit, retained=nothing)
    traces = Vector{Trace}(undef, algorithm.num_particles)
    weights = Vector{Float64}(undef, algorithm.num_particles)
    Threads.@threads for i in 1:algorithm.num_particles
        if !isnothing(retained) && i == algorithm.num_particles
            proposed_trace, = Gen.generate(algorithm.q, (algorithm.target,), ValueChoiceMap{ChoiceMap}(retained))
        else
            proposed_trace = simulate(algorithm.q, (algorithm.target,))
        end
        model_trace, = Gen.generate(algorithm.target.p, algorithm.target.args, merge(get_retval(proposed_trace), algorithm.target.constraints))
        weights[i] = get_score(model_trace) - get_score(proposed_trace)
        traces[i] = model_trace
    end
    return ParticleCollection(traces, weights, 0.0)
end

"""
A `GeneralSMCStep` extends an existing algorithm with a Del-Moral SMC step, specified via 
a forward kernel `k`, a reverse kernel `l`, and a new target distribution `new_target`.
"""
struct GeneralSMCStep <: SMCAlgorithm
    previous :: SMCAlgorithm
    k :: Distribution{ChoiceMap}
    l :: Distribution{ChoiceMap}
    new_target :: Target
end

num_particles(algorithm::GeneralSMCStep) = num_particles(algorithm.previous)
final_target(algorithm::GeneralSMCStep) = algorithm.new_target
function run_smc(algorithm::GeneralSMCStep, retained=nothing)
    old_target = final_target(algorithm.previous)
    old_target_latents = latent_selection(old_target)

    if !isnothing(retained)
        new_retained_trace, = Gen.generate(algorithm.new_target.p, algorithm.new_target.args, merge(algorithm.new_target.constraints, retained))
        _, l_weight, old_retained_choices = propose(algorithm.l, (new_retained_trace, old_target))
        collection = run_smc(algorithm.previous, old_retained_choices)
        k_weight, = Gen.assess(algorithm.k, (last(collection.particles), algorithm.new_target), ValueChoiceMap{ChoiceMap}(retained))
        retained_weight = l_weight - k_weight + last(collection.weights) + get_score(new_retained_trace) - get_score(last(collection.particles))
    end
    
    particles = Trace[]
    weights = Float64[]
    idx = isnothing(retained) ? 0 : 1
    
    for (particle, weight) in zip(collection.particles[1:end-idx], collection.weights[1:end-idx])
        _, this_k_weight, new_choices = Gen.propose(algorithm.k, (particle, algorithm.new_target))
        new_trace, = Gen.generate(algorithm.new_target.p, algorithm.new_target.args, merge(algorithm.new_target.constraints, new_choices))
        old_choices = Gen.get_selected(Gen.get_choices(particle), old_target_latents)
        this_l_weight, = Gen.assess(algorithm.l, (new_trace, old_target), ValueChoiceMap{ChoiceMap}(old_choices))
        this_weight = this_l_weight - this_k_weight + weight + get_score(new_trace) - get_score(particle)
        push!(particles, new_trace)
        push!(weights, this_weight)
    end
    if !isnothing(retained)
        push!(particles, new_retained_trace)
        push!(weights, retained_weight)
    end
    return ParticleCollection(particles, weights, collection.lml_est)
end


"""
A `ChangeTargetSMCStep` extends an existing algorithm with a Del-Moral SMC step, where the new target
is over the same state space as the previous target, and both `k` and `l` are specialized to the identity 
kernel.
"""
struct ChangeTargetSMCStep <: SMCAlgorithm
    previous :: SMCAlgorithm
    new_target :: Target
end

num_particles(algorithm::ChangeTargetSMCStep) = num_particles(algorithm.previous)
final_target(algorithm::ChangeTargetSMCStep) = algorithm.new_target
function run_smc(algorithm::ChangeTargetSMCStep, retained=nothing)
    collection = run_smc(algorithm.previous, retained)
    old_target_latents = latent_selection(final_target(algorithm.previous))
    N = num_particles(algorithm.previous)
    particles = Vector{Trace}(undef, N)
    weights = Vector{Float64}(undef, N)
    Threads.@threads for i in 1:N 
        (particle, weight) = collection.particles[i], collection.weights[i]
        latents = Gen.get_selected(Gen.get_choices(particle), old_target_latents)
        new_trace, = Gen.generate(algorithm.new_target.p, algorithm.new_target.args, merge(algorithm.new_target.constraints, latents))
        this_weight = get_score(new_trace) - get_score(particle) + weight
        particles[i] = new_trace
        weights[i] = this_weight
    end
    return ParticleCollection(particles, weights, collection.lml_est)
end


"""
An `ExtendingSMCStep` extends an existing algorithm with a standard SMC step, where the new target
extends the previous target by changing its arguments and adding new constraints, and the proposal 
`k` samples new latent variables from the new target (but cannot delete or move preexisting variables).
"""
struct ExtendingSMCStep <: SMCAlgorithm
    previous :: SMCAlgorithm
    k :: Distribution{ChoiceMap}
    new_args :: Tuple
    argdiffs :: Tuple
    new_constraints :: ChoiceMap
end

num_particles(algorithm::ExtendingSMCStep) = num_particles(algorithm.previous)
final_target(algorithm::ExtendingSMCStep) = begin
    t = final_target(algorithm.previous)
    Target(t.p, algorithm.new_args, merge(t.constraints, algorithm.new_constraints))
end

function run_smc(algorithm::ExtendingSMCStep, retained=nothing)
    new_target = final_target(algorithm)

    if !isnothing(retained)
        old_target = final_target(algorithm.previous)
        # Create a retained trace.
        new_retained_trace, = Gen.generate(new_target.p, new_target.args, merge(new_target.constraints, retained))
        # Use Gen.update to revert to the previous target.
        # TODO: reusing the same argdiffs to go *backward* is not always correct.
        # We should probably apply some sort of 'inverse' transformation to each argdiff.
        # Luckily, for common argdiffs, which indicate *what* changed but not how, this is not a problem.
        previous_trace, _, _, discard = Gen.update(new_retained_trace, old_target.args, algorithm.argdiffs, EmptyChoiceMap())
        # The discard is the set of new choices added this step.
        # The previous retained map does not include them:
        previous_latents = Gen.get_selected(retained, complement(selection_from_choicemap(discard)))
        # Evaluate the `k` probability of the discard
        forward_weight_retained, = Gen.assess(algorithm.k, (previous_trace, new_target), ValueChoiceMap{ChoiceMap}(discard))
    end
    collection = run_smc(algorithm.previous, isnothing(retained) ? nothing : previous_latents)

    N = num_particles(algorithm.previous)
    particles = Vector{Trace}(undef, N)
    weights = zeros(N)
    idx = isnothing(retained) ? 0 : 1
    Threads.@threads for i in 1:(N-idx)
        particle, weight = collection.particles[i], collection.weights[i]
        extension = simulate(algorithm.k, (particle, new_target))
        extension, k_score = get_retval(extension), get_score(extension)
        new_trace, model_score_change, _, _ = update(particle, algorithm.new_args, algorithm.argdiffs, merge(extension, algorithm.new_constraints))
        particles[i] = new_trace
        weights[i] = weight - k_score + model_score_change
    end
    if !isnothing(retained)
        particles[end] = new_retained_trace
        weights[end] = last(collection.weights) + forward_weight_retained + get_score(new_retained_trace) - get_score(last(collection.particles))
    end 
    return ParticleCollection(particles, weights, collection.lml_est)
end

struct SMCRejuvenate <: SMCAlgorithm
    previous :: SMCAlgorithm
    kernel :: Function
end

num_particles(algorithm::SMCRejuvenate) = num_particles(algorithm.previous)
final_target(algorithm::SMCRejuvenate) = final_target(algorithm.previous)
function run_smc(alg::SMCRejuvenate, retained=nothing)
    if !isnothing(retained)
        new_retained_trace, = Gen.generate(final_target(alg).p, final_target(alg).args, merge(final_target(alg).constraints, retained))
        previous_trace, = Gen.reversal(alg.kernel)(new_retained_trace)
        previous_latents = get_latents(final_target(alg), previous_trace)
    end
    collection = run_smc(alg.previous, isnothing(retained) ? nothing : previous_latents)
    particles = Trace[]
    idx = isnothing(retained) ? 0 : 1
    for particle in collection.particles[1:end-idx]
        new_trace, = alg.kernel(particle)
        push!(particles, new_trace)
    end
    if !isnothing(retained)
        push!(particles, new_retained_trace)
    end
    return ParticleCollection(particles, collection.weights, collection.lml_est)
end

# Implementing the SP distribution interface for arbitrary SMC algorithms:
function random_weighted(g::SMCAlgorithm, target::Target)
    algorithm = ChangeTargetSMCStep(g, target)
    particle_collection = run_smc(algorithm)
    # Randomly select a particle according to its weight
    weights = particle_collection.weights
    total_weight = logsumexp(weights)
    probs = exp.(weights .- total_weight)
    particle_index = categorical(probs)
    particle = particle_collection.particles[particle_index]
    return Gen.get_selected(get_choices(particle), latent_selection(target)), 
           get_score(particle) - (particle_collection.lml_est + total_weight - log(length(particle_collection.particles)))
end

function estimate_logpdf(g::SMCAlgorithm, choices::ChoiceMap, target::Target)
    algorithm = ChangeTargetSMCStep(g.algorithm, target)

    # Perform a backward pass through the algorithm.
    collection = run_smc(algorithm, choices)
    
    return get_score(last(collection.particles)) - log_marginal_likelihood(collection)
end
