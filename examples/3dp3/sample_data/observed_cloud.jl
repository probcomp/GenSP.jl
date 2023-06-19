using CSV
using DataFrames
using GenSP
using Gen

observed_cloud = CSV.File("examples/3dp3/sample_data/0048-001542-obs-pc.csv") |> DataFrame |> Matrix{Float64} |> transpose
observation_choicemap = choicemap([(:cloud => i) => observed_cloud[:, i] for i=1:size(observed_cloud, 2)]...);

