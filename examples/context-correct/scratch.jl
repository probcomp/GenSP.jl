
# Plot the trajectories in terms of number of iterations
unsafe_trajectories = [r.scores for r in unsafe_results]
safe_trajectories = [r.scores for r in safe_results]
exact_trajectories = [r.scores for r in exact_results]

plot(title="context-sensitive spelling correction",
     xlabel="iterations",
     ylabel="log unnormalized probability",
     fontsize=15,
     tickfontsize=15,
     labelfontsize=15,
     legendfontsize=15,
     titlefontsize=15,
     legend=:bottomright)

plot!([], label="Handcoded", color=1)
for trajectory in unsafe_trajectories
    plot!(trajectory, color=1, linewidth=3, alpha=0.3, label=nothing)
end
plot!([], label="GenSP", color="lightgreen")
for trajectory in safe_trajectories
    plot!(trajectory, color="lightgreen", linewidth=2, label=nothing)
end
plot!([], label="Exact", color=2)
for trajectory in exact_trajectories
    plot!([-180.2070833733241, trajectory[2:end]...], color=2, linewidth=3, alpha=0.2, label=nothing)
end
plot!()
savefig("10-runs-context-sensitive-spelling-correction-iterations.pdf")

function make_plot(title, xlabel, ylabel, legend=:bottomright)
    plot(title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend)
end
median_unsafe_trajectory = [StatsBase.median([traj[t] for traj in unsafe_trajectories]) for t in 1:301]
median_exact_trajectory = [StatsBase.median([traj[t] for traj in exact_trajectories]) for t in 1:301]
median_exact_trajectory[1] = median_unsafe_trajectory[1]
median_safe_trajectory = [StatsBase.median([traj[t] for traj in safe_trajectories]) for t in 1:301]

median_iters_plot = make_plot("", "", "context-correct");
plot!(median_iters_plot, median_safe_trajectory, color=3, label="Prox (safe)", linewidth=5)
plot!(median_iters_plot, median_unsafe_trajectory, color=1, label="Prox", linewidth=5)
plot!(median_iters_plot, median_exact_trajectory, color=2, label="Gen", linewidth=5)

median_timed_plot = make_plot("", "", "", nothing);
plot!(median_timed_plot, unsafe_results[1].times[2:301], median_unsafe_trajectory[2:301], label=nothing, color=1, linewidth=3, xscale=:log)
plot!(median_timed_plot, exact_results[1].times[2:301], median_exact_trajectory[2:301], label=nothing, color=2, linewidth=3, xscale=:log)
plot!(median_timed_plot, safe_results[1].times[2:301], median_safe_trajectory[2:301], label=nothing, color=3, linewidth=3, xscale=:log)

plot(median_iters_plot, median_timed_plot, link=:x)

savefig("median-mcmc-impact-plot.pdf")




res_ex, lpdfs_mh_ex, acc_rate_ex, times_ex = typos_mh(truncated_static_model, "i kni that yo tink taht", 100, true, init_trace_exact)
res_ex_2, lpdfs_mh_ex, acc_rate_ex = typos_mh(truncated_static_model, "i kni that yo tink taht", 100)
res_ex_3, lpdfs_mh_ex, acc_rate_ex = typos_mh(truncated_static_model, "i kni that yo tink taht", 100)

plot(times[1:100], lpdfs_mh, title="context-sensitive spelling correction", label="GenSP.jl (unsafe)", ylabel="log unnormalized probability", xlabel="time (seconds)", legend=:bottomright, tickfontsize=15, labelfontsize=15, legendfontsize=15, linewidth=3, titlefontsize=15, xscale=:log)
plot!(times_ex[1:100], lpdfs_mh_ex, label="Gen.jl (exact)", linewidth=3)
plot!(safe_times[1:100], safe_lpdfs_mh, label="GenSP.jl (safe)", linewidth=3)
savefig("context-sensitive-spelling-correction-mcmc-convergence-wall-clock.pdf")
plot(lpdfs_mh[1:100], title="context-sensitive spelling correction", label="Prox", ylabel="log unnormalized probability", xlabel="# iterations", legend=:bottomright, labelfontsize=15, legendfontsize=15, linewidth=3, tickfontsize=15, titlefontsize=15)
plot!(lpdfs_mh_ex, label="Gen", linewidth=3)
savefig("context-sensitive-spelling-correction-mcmc-convergence.pdf")




function smc_time(model, truth, sentence, N=1)
    # We continually increase particle count by 50%
    truth = split(truth)
    results = []
    for _ in 1:N
        K = 100
        run = nothing
        while true
            run = @timed typos_smc(model, sentence, K, true)
            state = run.value
            K = Int(floor(K * 1.5))
            println(K) 
            println([state.traces[1][i => :word] for i in 1:length(truth)])
            if any(tr -> all(((i,w),) -> tr[i => :word] == w, enumerate(truth)), state.traces)
                break
            end
        end
        push!(results, run.time)
    end
    return results
end


function mh_time(model, truth, sentence, N=1)
    n = length(split(sentence))
    cm = sentence_choicemap(sentence)
    words = map(String, split(truth))
    results = []
    for _ in 1:N
        tr, = generate(model, (n,START_WORD), cm)
        iter = 0
        t = @timed while any(((i, w),) -> tr[i => :word] != w, enumerate(words))
            for j=1:n
                tr, = mh(tr, select(j => :word))
            end
            if iter % 2500 == 0
                println("Iter $iter")
                println([tr[i => :word] for i in 1:n])
            end
            iter += 1
        end
        push!(results, t.time)
    end
    return results
end


baseline_timings = []
for (truth, obs) in sentences
    println(truth)
    push!(baseline_timings, mh_time(static_model, truth, obs, 10))
end

smc_timings = []
for (truth, obs) in sentences
    println(truth)
    push!(smc_timings, smc_time(static_model, truth, obs, 10))
end









safe_3d = deserialize("../results/results_10_safe_300_iters_without_traces_3_15_2_05am.jld");
unsafe_3d = deserialize("../results/mcmc-runs-unsafe-test03220230.jld");
exact_3d = deserialize("../results/mcmc-runs-exact-test03231215.jld");


# Means for typos
mean_unsafe_trajectory_typos = [logmeanexp([r.scores[t] for r in unsafe_results]) for t in 1:length(unsafe_results[1].scores)];
mean_safe_trajectory_typos = [logmeanexp([r.scores[t] for r in safe_results]) for t in 1:length(safe_results[1].scores)];
mean_exact_trajectory_typos = [logmeanexp([r.scores[t] for r in exact_results]) for t in 1:length(exact_results[1].scores)];
mean_unsafe_times_typos = [mean([r.times[t] for r in unsafe_results]) for t in 1:length(unsafe_results[1].times)];
mean_safe_times_typos = [mean([r.times[t] for r in safe_results]) for t in 1:length(safe_results[1].times)];
mean_exact_times_typos = [mean([r.times[t] for r in exact_results]) for t in 1:length(exact_results[1].times)];
std_unsafe_trajectory_typos = [(std(exp.([r.scores[t] for r in unsafe_results]))) for t in 1:length(unsafe_results[1].scores)];
high_ribbon_unsafe_typos = [logsumexp(mean_unsafe_trajectory_typos[i], log(std_unsafe_trajectory_typos[i])) - mean_unsafe_trajectory_typos[i] for i in 1:length(std_unsafe_trajectory_typos)];
low_ribbon_unsafe_typos = [mean_unsafe_trajectory_typos[i] - log(max(1e-80, exp(mean_unsafe_trajectory_typos[i]) - (std_unsafe_trajectory_typos[i]))) for i in 1:length(std_unsafe_trajectory_typos)];
logspace_stds(stds, means) = [log(stds[i]) - (means[i]) for i in 1:length(stds)]

using Plots
mean_iters_plot_typos = make_plot("", "", "context-correct");
plot!(mean_iters_plot_typos, mean_safe_trajectory_typos, color=1, label="GenSP estimator", linewidth=5,labelfontsize=15, legendfontsize=10,xtickfontsize=10, ytickfontsize=10)
plot!(mean_iters_plot_typos, mean_unsafe_trajectory_typos, color=2, label="Handcoded estimator", linewidth=5)
plot!(mean_iters_plot_typos, mean_exact_trajectory_typos, color=3, label="Exact density", linewidth=5)

mean_time_plot_typos = make_plot("", "", "", nothing);
plot!(mean_time_plot_typos, mean_unsafe_times_typos[2:301], mean_unsafe_trajectory_typos[1:300], label=nothing, color=2, linewidth=5, xscale=:log, labelfontsize=15, legendfontsize=10,xtickfontsize=10, ytickfontsize=10)
plot!(mean_time_plot_typos, mean_exact_times_typos[2:301], mean_exact_trajectory_typos[1:300], label=nothing, color=3, linewidth=5, xscale=:log)
plot!(mean_time_plot_typos, mean_safe_times_typos[2:301], mean_safe_trajectory_typos[1:300], label=nothing, color=1, linewidth=5, xscale=:log)

plot(mean_iters_plot_typos, mean_time_plot_typos, link=:x)

# Means for 3d
mean_unsafe_trajectory_3d = [logmeanexp([r[1][t] for r in unsafe_3d]) for t in 1:length(unsafe_3d[1][1])];
mean_safe_trajectory_3d = [logmeanexp([r[1][t] for r in safe_3d]) for t in 1:length(safe_3d[1][1])];
mean_exact_trajectory_3d = [logmeanexp([r[1][t] for r in exact_3d]) for t in 1:length(exact_3d[1][1])];
mean_unsafe_times_3d = [mean([r[2][t] for r in unsafe_3d]) for t in 1:length(unsafe_3d[1][2])];
mean_safe_times_3d = [mean([r.times[t] for r in safe_3d]) for t in 1:length(safe_3d[1].times)];
mean_exact_times_3d = [mean([r[2][t] for r in exact_3d]) for t in 1:length(exact_3d[1][2])];

# Plots for 3d
mean_iters_plot_3d = make_plot("", "", "3DP3");
plot!(mean_iters_plot_3d, mean_safe_trajectory_3d, color=1, label="GenSP estimator", linewidth=5,labelfontsize=15, legendfontsize=10,xtickfontsize=10, ytickfontsize=10)
plot!(mean_iters_plot_3d, mean_unsafe_trajectory_3d, color=2, label="Handcoded estimator", linewidth=5)
plot!(mean_iters_plot_3d, mean_exact_trajectory_3d, color=3, label="Exact density", linewidth=5)

mean_time_plot_3d = make_plot("", "", "", nothing);
plot!(mean_time_plot_3d, mean_unsafe_times_3d[2:301], mean_unsafe_trajectory_3d[1:300], label=nothing, color=2, linewidth=5, xscale=:log, labelfontsize=15, legendfontsize=10,xtickfontsize=10, ytickfontsize=10)
plot!(mean_time_plot_3d, mean_exact_times_3d[2:301], mean_exact_trajectory_3d[1:300], label=nothing, color=3, linewidth=5, xscale=:log)
plot!(mean_time_plot_3d, mean_safe_times_3d[2:301], mean_safe_trajectory_3d[1:300], label=nothing, color=1, linewidth=5, xscale=:log)

plot(mean_iters_plot_3d, mean_time_plot_3d, link=:x)

