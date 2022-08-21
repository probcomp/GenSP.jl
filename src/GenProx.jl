module GenProx

using Gen

include("util.jl")
include("distribution.jl")
include("target.jl")
include("marginal.jl")
include("inference/inference.jl")

end