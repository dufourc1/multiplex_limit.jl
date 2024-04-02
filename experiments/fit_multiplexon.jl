using multiplex_limit

using NetworkHistogram
using DelimitedFiles
using Plots
using Statistics
using LinearAlgebra
using MVBernoulli

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

include("utils.jl")


model = multiplex_limit.random_multiplexon(5, 2)
n = 200
latents = orderered_latents(n, model.π)
@assert length(latents) == n


A, ξ = rand(model, n, latents)
@assert size(A) == (n, n, 2)
@assert length(ξ) == n

estimated, history = graphhist(
    A; starting_assignment_rule = NetworkHistogram.RandomStart(), maxitr = Int(1e6),
    stop_rule = PreviousBestValue(1000))

moments, indices = NetworkHistogram.get_moment_representation(estimated)


plots_pij = []
plots_truth = []
for (i, index) in enumerate(indices)
    p_true = get_p_matrix(multiplex_limit.get_matrix_moment(model, i), ξ)
    p_hat = get_p_matrix(moments[:, :, i], estimated.node_labels)
    push!(plots_pij,
        heatmap(
            p_hat,
            clims = (0, 1),
            xformatter = _ -> "",
            yformatter = _ -> "",
            xlabel = "Estimated moment $index",
            aspect_ratio = :equal,
            axis = ([], false)
        )
    )
    push!(plots_truth,
        heatmap(
            p_true,
            clims = (0, 1),
            xformatter = _ -> "",
            yformatter = _ -> "",
            xlabel = "True moment $index",
            aspect_ratio = :equal,
        )
    )
    println("MSE between true and estimated moment $index: ", mean((p_true .- p_hat).^2 ./mean(p_true.^2)))
end

for i in 1:length(indices)
    display(plot(plots_pij[i], plots_truth[i], layout = (1, 2), size = (1200, 600)))
end


corrs = MVBernoulli.correlation_matrix.(model.θ)
index_one = MVBernoulli.binary_vector_to_index(ones(Int, length(A[1,1,:])))
index_zero = MVBernoulli.binary_vector_to_index(zeros(Int, length(A[1, 1, :])))

proba_ones = vcat([vcat([model.θ[i, j].tabulation.p[index_one] for j in i:size(model.θ, 1)]...)
                for i in 1:size(model.θ, 1)]...)
proba_zeros = vcat([vcat([model.θ[i, j].tabulation.p[index_zero] for j in i:size(model.θ, 1)]...)
                for i in 1:size(model.θ, 1)]...)

corr_12 = [c[1,2] for c in vec(corrs)]
plot_corr_vs_proba = scatter(corr_12, proba_zeros, label = "probability of 0")
scatter!(plot_corr_vs_proba, corr_12, proba_ones, label = "probability of 1")
plot!(plot_corr_vs_proba, ylabel = "Probability", xlabel = "Correlation x_1 and x_2", legend = :topright)
display(plot_corr_vs_proba)
