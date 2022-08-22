# An SMC algorithm is a GenerativeFunction with a ValueChoiceMap{ChoiceMap} as its trace
# and a collection of weighted traces as its return value. It takes as input another SMC
# algorithm.

struct ParticleCollection
    particles :: Vector{Trace}
    weights   :: Vector{Float64}
    lml_est   :: Float64
end

struct SMCAlgorithmTrace <: Trace
    algorithm :: GenerativeFunction
    value_choice_map :: ValueChoiceMap{ChoiceMap}
end

abstract type SMCAlgorithm <: GenerativeFunction{SMCAlgorithmTrace, ParticleCollection} end

struct SMCResample <: SMCAlgorithm
    previous :: SMCAlgorithm
    how_many :: Int
end

num_particles(algorithm::SMCResample) = algorithm.how_many
final_target(algorithm::SMCResample) = final_target(algorithm.previous)

struct SMCClone <: SMCAlgorithm
    previous :: SMCAlgorithm
    factor   :: Int
end

num_particles(algorithm::SMCClone) = algorithm.factor * num_particles(algorithm.previous)
final_target(algorithm::SMCClone) = final_target(algorithm.previous)

struct SMCInit <: SMCAlgorithm
    q :: GenerativeFunction
    target :: Target
    num_particles :: Int
end

num_particles(algorithm::SMCInit) = algorithm.num_particles
final_target(algorithm::SMCInit) = algorithm.target

struct GeneralSMCStep <: SMCAlgorithm
    previous :: SMCAlgorithm
    k :: GenerativeFunction
    l :: GenerativeFunction
    new_target :: Target
end

num_particles(algorithm::GeneralSMCStep) = num_particles(algorithm.previous)
final_target(algorithm::GeneralSMCStep) = algorithm.new_target

struct ChangeTargetSMCStep <: SMCAlgorithm
    previous :: SMCAlgorithm
    new_target :: Target
end

num_particles(algorithm::ChangeTargetSMCStep) = num_particles(algorithm.previous)
final_target(algorithm::ChangeTargetSMCStep) = algorithm.new_target

struct ExtendingSMCStep <: SMCAlgorithm
    previous :: SMCAlgorithm
    k :: GenerativeFunction
    new_args :: Tuple
    argdiffs :: Tuple
    new_constraints :: ChoiceMap
end

num_particles(algorithm::ExtendingSMCStep) = num_particles(algorithm.previous)
final_target(algorithm::ExtendingSMCStep) = begin
    t = final_target(algorithm.previous)
    Target(t.p, algorithm.new_args, merge(t.constraints, algorithm.new_constraints))
end

struct SMCRejuvenate <: SMCAlgorithm
    previous :: SMCAlgorithm
    kernel :: Function
end

num_particles(algorithm::SMCRejuvenate) = num_particles(algorithm.previous)
final_target(algorithm::SMCRejuvenate) = final_target(algorithm.previous)

struct SMCMaybeResample <: SMCAlgorithm
    previous :: SMCAlgorithm
    threshold :: Float64
end

num_particles(algorithm::SMCMaybeResample) = num_particles(algorithm.previous)
final_target(algorithm::SMCMaybeResample) = final_target(algorithm.previous)

struct GeneralSMC <: ProxDistribution{ChoiceMap}
    algorithm :: SMCAlgorithm
end

function random_weighted(g::GeneralSMC, target::Target)
    algorithm = ChangeTargetSMCStep(g.algorithm, target)
    particle_collection = forward(algorithm)
    # Randomly select a particle according to its weight
    weights = particle_collection.weights
    total_weight = logsumexp(weights)
    probs = exp.(weights .- total_weight)
    particle_index = categorical(probs)
    particle = particle_collection.particles[particle_index]
    # TODO: only the choices that are latent should be returned
    return get_choices(particle), particle_collection.lml_est + total_weight - log(length(particle_collection.particles))
end