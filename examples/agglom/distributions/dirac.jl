struct Dirac{T} <: Distribution{T} end
Gen.random(::Dirac, x) = x
Gen.logpdf(::Dirac, val, x) = (x == val) ? 0.0 : -Inf
const dirac = Dirac{Any}()