using CSV
using DataFrames
using NearestNeighbors 

latent_cloud = CSV.File("examples/3dp3/sample_data/0048-001542-gt-pc.csv") |> DataFrame |> Matrix{Float64} |> transpose
latent_tree = NearestNeighbors.KDTree(latent_cloud)
