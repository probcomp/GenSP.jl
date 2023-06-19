@testset "latent selections" begin
    choices = choicemap(:x => 4, :y => 2, (:z => 3) => 6)
    constraints = choicemap(:y => 1)
    # normal doesn't make sense here, 
    # but we just need to construct *some* Target object.
    # if we add 'trace type checking' in the future, this will error.
    target = Target(Gen.normal, (), constraints)
    selection = GenSP.latent_selection(target)
    @test :x in selection
    @test (:z => 3) in selection
    @test !(:y in selection)
    latents = GenSP.get_latents(target, choices)
    @test latents[:x] == 4
    @test latents[:z => 3] == 6
    @test !(Gen.has_value(latents, :y))
end