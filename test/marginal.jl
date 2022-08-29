@testset "marginal (exact discrete)" begin
    @gen function mixture_model()
        z ~ bernoulli(0.5)
        x ~ normal(z ? -4 : 4, 1)
    end
    @gen function proposal(target)
        x = target.constraints[:x]
        log_prob_true = logpdf(normal, x, -4, 1)
        log_prob_false = logpdf(normal, x, 4, 1)
        Z = logsumexp([log_prob_true, log_prob_false])
        z ~ bernoulli(exp(log_prob_true - Z))
    end
    mixture_model_inference = custom_importance(ChoiceMapDistribution(proposal), 1)
    mixture_model_dist = Marginal{Float64}(mixture_model, mixture_model_inference, :x)
    true_logpdf = logsumexp([logpdf(normal, 1.0, -4, 1), logpdf(normal, 1.0, 4, 1)]) - log(2)
    @test isapprox(estimate_logpdf(mixture_model_dist, 1.0), true_logpdf)
    @test isapprox(estimate_logpdf(mixture_model_dist, 1.0), estimate_logpdf(mixture_model_dist, 1.0))
    x, w = random_weighted(mixture_model_dist)
    true_w = logsumexp([logpdf(normal, x, -4, 1), logpdf(normal, x, 4, 1)]) - log(2)
    @test isapprox(w, true_w)
end

@testset "marginal (exact continuous)" begin
    @gen function normal_normal_model()
        x ~ normal(3, 2)
        y ~ normal(x, 0.5)
    end
    @gen function proposal(target)
        y = target.constraints[:y]
        posterior_var = 1/(1/4 + 1/0.25)
        posterior_mean = posterior_var * (3/4 + y/0.25)    
        x ~ normal(posterior_mean, sqrt(posterior_var))
    end
    normal_normal_model_inference = custom_importance(ChoiceMapDistribution(proposal), 1)
    normal_normal_model_dist = Marginal{Float64}(normal_normal_model, normal_normal_model_inference, :y)
    normal_normal_true_logpdf(y) = logpdf(normal, y, 3, sqrt(4 + 0.25))
    @test isapprox(estimate_logpdf(normal_normal_model_dist, 1.0), normal_normal_true_logpdf(1.0))
    @test isapprox(estimate_logpdf(normal_normal_model_dist, 1.0), estimate_logpdf(normal_normal_model_dist, 1.0))
    y, w = random_weighted(normal_normal_model_dist)
    true_w = normal_normal_true_logpdf(y)
    @test isapprox(w, true_w)
end

@testset "marginal (approximate)" begin
    @gen function normal_normal_model()
        x ~ normal(3, 2)
        y ~ normal(x, 0.5)
    end
    @gen function proposal(target)
        x ~ normal(0, sqrt(20))
    end
    normal_normal_model_inference = custom_importance(ChoiceMapDistribution(proposal), 100000)
    normal_normal_model_dist = Marginal{Float64}(normal_normal_model, normal_normal_model_inference, :y)
    normal_normal_true_logpdf(y) = logpdf(normal, y, 3, sqrt(4 + 0.25))
    @test isapprox(estimate_logpdf(normal_normal_model_dist, 1.0), normal_normal_true_logpdf(1.0); atol=0.1)
    @test isapprox(estimate_logpdf(normal_normal_model_dist, 1.0), estimate_logpdf(normal_normal_model_dist, 1.0); atol=0.1)
    y, w = random_weighted(normal_normal_model_dist)
    true_w = normal_normal_true_logpdf(y)
    @test isapprox(w, true_w; atol=0.1) 
end