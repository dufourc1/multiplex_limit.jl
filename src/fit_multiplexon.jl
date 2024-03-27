using multiplex_limit

using NetworkHistogram
using DelimitedFiles
using Plots
using Statistics
using LinearAlgebra

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

include("utils.jl")

n = 1000

model = multiplex_limit.random_multiplexon(10, 3)
latents = orderered_latents(n, model.π)
A, ξ = rand(model, n, latents)

estimated, history = graphhist(
    A; starting_assignment_rule = RandomStart(), maxitr = Int(1e6),
    stop_rule = PreviousBestValue(1000))

moments, indices = NetworkHistogram.get_moment_representation(estimated)
plots_truth = []
plots_estimates = []

function get_matrix_moment(model::multiplex_limit.Multiplexon, moment_index)
    k = length(model.π)
    moment = Matrix{Float64}(undef, k, k)
    for i in 1:k
        for j in 1:k
            moment[i, j] = model.θ[i, j].ordinary_moments[moment_index + 1]
        end
    end
    return moment
end

plots_pij = []
plots_truth = []
for (i, index) in enumerate(indices)
    push!(plots_pij,
        heatmap(
            get_p_matrix(moments[:, :, i], estimated.node_labels),
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
            get_p_matrix(get_matrix_moment(model, i), ξ),
            clims = (0, 1),
            xformatter = _ -> "",
            yformatter = _ -> "",
            xlabel = "True moment $index",
            aspect_ratio = :equal,
        )
    )
end

for i in 1:length(indices)
    display(plot(plots_pij[i], plots_truth[i], layout = (1, 2), size = (1200, 600)))
end
