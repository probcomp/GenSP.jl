# We encode our likelihood as a distribution over bags.
# First, the Gaussian likelihood: the probability of a bag is
# the marginal probability of the bag's contents' being generated 
# by a Gaussian with a particular mean and standard deviation,
# drawn from a conjugate (Normal-Inverse-Gamma) prior.
struct GaussianBag <: Distribution{Set{Float64}} end
const gaussian_bag = GaussianBag()

struct GaussianHypers
    mean :: Float64
    lambda :: Float64
    alpha :: Float64
    beta :: Float64
end

function Gen.random(::GaussianBag, N::Int, hypers::GaussianHypers)
    # Generate the parameters of the latent Gaussian
    variance = Gen.inv_gamma(hypers.alpha, hypers.beta)
    mean = Gen.normal(hypers.mean, sqrt(variance / hypers.lambda))
    samples = [Gen.normal(mean, sqrt(variance)) for _ in 1:N]
    return Set(samples)
end

function Gen.logpdf(::GaussianBag, bag::Set{Float64}, N::Int, hypers::GaussianHypers)
    # Compute the log probability of the bag.
    combined_mean = mean(bag)
    combined_sse  = sum((x - combined_mean)^2 for x in bag)

    lambda_tot = hypers.lambda + N
    alpha_tot  = hypers.alpha + N/2
    beta_tot   = hypers.beta + 0.5 * combined_sse + (hypers.lambda * N * (combined_mean - hypers.mean)^2) / (2*lambda_tot)

    loggamma(alpha_tot) - loggamma(hypers.alpha) + log(hypers.beta) * hypers.alpha - log(beta_tot) * alpha_tot + 0.5 * (log(hypers.lambda) - log(lambda_tot)) - (N/2)*log(2pi)
end
