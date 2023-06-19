# Code for generating data and plots for SMC.
# We begin by loading the latest serialized model.

using Serialization

function generate_data(model, rerun=false)
    # Load current data, using the Serialization package in Julia, if the file exists
    if isfile("results/$(model.name).jld2")
        results = Serialization.deserialize("results/$(model.name).jld2")
    else
        results = Dict()
    end
    # Generate new data
    for model_kind in [:safe, :unsafe, :exact]
        for (num_particles, num_replicates) in model.computation_settings
            if rerun || !haskey(results, (model_kind, num_particles))
                scores = [@timed last(model.run_smc(num_particles, model_kind)) for _ in 1:num_replicates]
                mean_time = mean(r.time for r in scores)
                mean_score = mean(r.value for r in scores)
                std_score = std(r.value for r in scores)
                lml_est = logmeanexp([r.value for r in scores])
                results[(model_kind, num_particles)] = (mean_time, mean_score, std_score, lml_est)
                print("$(model.name) $(model_kind) $(num_particles): $(results[(model_kind, num_particles)])\n")
            end
            # Save new data
            Serialization.serialize("results/$(model.name).jld2", results)
        end
    end
end

generate_data(agglom_model_config)
generate_data(rrt_model)
generate_data(typos_model_config)

using Plots

model_kind_names = Dict(:safe => "GenSP", :unsafe => "Handcoded", :exact => "Exact")

# We can now plot the results. We use the Plots package in Julia.
function generate_smc_results_plots(model_name)
    results = Serialization.deserialize("results/$(model_name).jld2")
    # Plot the results
    # pyplot()
    # Make a two-panel plot (panels arranged vertically).
    # The top panel shows number of particles vs. score (with yerr = std), with three different 
    # curves showing the results for the safe, unsafe, and exact models.
    # The bottom panel shows time vs. score, with three different curves showing the results for the
    # safe, unsafe, and exact models.
    # add a margin
    # change the precision of the axis ticks to show 1 decimal point only
    p = plot(layout=(1,2), size=(1000,300), legend=:bottomright, topmargin=5Plots.mm, 
                leftmargin=7Plots.mm, bottommargin=10Plots.mm,
                labelfontsize=15, legendfontsize=13, tickfontsize=15,
                yformatter=:plain, xformatter=:plain)
    for model_kind in [:safe, :unsafe, :exact]
        model_kind_name = model_kind_names[model_kind]
        num_particles = sort([k[2] for k in keys(results) if k[1] == model_kind])
        mean_scores = [results[(model_kind, k)][2] for k in num_particles]
        std_scores = [results[(model_kind, k)][3] for k in num_particles]
        mean_times = [results[(model_kind, k)][1] for k in num_particles]
        println("Plotting $num_particles, $mean_scores")
        plot!(p[1], num_particles, mean_scores, xscale=:log10, label="$(model_kind_name)", marker=:circle, linewidth=5, xlabel="Number of particles", ylabel="Log weight")
        println("Plotting $num_particles, $mean_times")
        plot!(p[2], mean_times, mean_scores, xscale=:log10, legend=nothing, label="$(model_kind_name)", marker=:circle, linewidth=5, xlabel="Time (s)", yticks=nothing)
    end
    savefig("results/$(model_name)_smc_results.pdf")
end