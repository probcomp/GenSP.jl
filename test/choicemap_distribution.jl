@testset "choicemap distribution" begin
    @gen function f()
        x ~ normal(0, 1)
        {:y => 1} ~ normal(0, 5)
        z ~ normal(x, 1)
    end
    cm = Gen.get_choices(simulate(f, ()))
    full_cm_dist = ChoiceMapDistribution(f)
    @test isapprox(estimate_logpdf(full_cm_dist, cm), first(assess(f, (), cm)))

    just_y_dist = ChoiceMapDistribution(f, select(:y))
    @test isapprox(estimate_logpdf(just_y_dist, choicemap((:y => 1) => 3.0)), logpdf(normal, 3.0, 0, 5))

    just_z_dist = ChoiceMapDistribution(f, select(:z), importance(1000))
    @test isapprox(estimate_logpdf(just_z_dist, choicemap((:z) => 1.0)), logpdf(normal, 1.0, 0, sqrt(2)); atol=0.05)
    (x, w) = random_weighted(just_z_dist)
    @test isapprox(w, logpdf(normal, x[:z], 0, sqrt(2)); atol=0.05)
end