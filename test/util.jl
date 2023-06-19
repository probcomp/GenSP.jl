@testset "selection from choicemap" begin

    @gen function g()
    end

    @gen function f()
        x ~ normal(0, 1)
        {:y => 1} ~ normal(0, 5)
        z ~ g()
    end

    cm = Gen.get_choices(simulate(f, ()))
    selection = GenSP.selection_from_choicemap(cm)

    @test :x in selection
    @test (:y => 1) in selection
    @test !(:z in selection)
end