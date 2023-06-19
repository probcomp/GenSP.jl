using Gen, GenSP, BenchmarkTools

include("distributions/render_point/whole_cloud.jl");

# Example 1: 3D point cloud
include("sample_data/latent_cloud.jl");
include("sample_data/observed_cloud.jl");
args_3d = (latent_cloud, 0.05, size(observed_cloud, 2))

display(@benchmark assess(render_point_cloud_unsafe, args_3d, observation_choicemap))
display(@benchmark assess(render_point_cloud_safe, args_3d, observation_choicemap))
display(@benchmark assess(render_point_cloud_exact_unsafe, args_3d, observation_choicemap))

# Example 2: 2D point cloud
include("sample_data/2d/preprocess.jl");
args_2d = (clouds[5], 0.01, size(cloud_obs, 2))
obs_2d = choicemap([(:cloud => i) => cloud_obs[:, i] for i=1:size(cloud_obs, 2)]...);

display(@benchmark assess(render_point_cloud_unsafe, args_2d, obs_2d))
display(@benchmark assess(render_point_cloud_safe, args_2d, obs_2d))
display(@benchmark assess(render_point_cloud_exact_unsafe, args_2d, obs_2d))
