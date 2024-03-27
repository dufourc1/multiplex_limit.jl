## analysis of malaria data from https://doi.org/10.1371/journal.pcbi.1003268
#  conclusion: not enough data ? I think I get confused between community estimation
# and graphon estimation without realizing it.

using NetworkHistogram
using DelimitedFiles
using Plots
using Statistics
using LinearAlgebra

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

include("utils.jl")

function edge_list_to_adjs(edge_lists = [
        "data/malaria_data/HVR_1.txt",
        "data/malaria_data/HVR_5.txt",
        "data/malaria_data/HVR_6.txt"])
    M = length(edge_lists)

    edges = [readdlm(e, ',', Int, '\n') for e in edge_lists]
    n = maximum([maximum(edges[i]) for i in eachindex(edges)])

    A = Array{Int, 3}(undef, n, n, M)
    A .= 0
    for i in 1:M
        for e in eachrow(edges[i])
            A[e[1], e[2], i] = 1
            A[e[2], e[1], i] = 1
        end
    end
   return A
end

path_to_data = "data/malaria_data/"
nums_to_analyze = collect(1:9) #
edge_lists = [joinpath(path_to_data, "HVR_$i.txt") for i in nums_to_analyze]

A = edge_list_to_adjs(edge_lists)

# manually test correlation
corr_matrix = zeros(9, 9)
for i in 1:9
    for j in 1:9
        if i == j
            corr_matrix[i, j] = 1
        else
            corr_matrix[i, j] = cor(vec(A[:, :, i]), vec(A[:, :, j]))
        end
    end
end

highest_corr_indices, highest_corr = getci(corr_matrix - I, 2*6)


from_paper = [
    rectangle_around_cell(1,5)
    rectangle_around_cell(1,6)
    rectangle_around_cell(5,6)
]

top_corr = vcat([rectangle_around_cell(i[1],i[2]) for i in highest_corr_indices if i[1] < i[2]]...)

size_plot = 600
p = heatmap(corr_matrix, xlabel = "Network", ylabel = "Network", aspect_ratio = :equal,
    size = (size_plot-5, size_plot+1), xlims = (0.5, size(corr_matrix, 1) + 0.5),
    ylims = (0.5, size(corr_matrix, 2) + 0.5))
plot!(p, some_rects[:, 1], some_rects[:, 2], label = "Zhang et al., 2024",
    line = :green, lw = 2,)
plot!(p, top_corr[:, 1], top_corr[:, 2],
    label = "top $(size(top_corr,1)÷6)", line = :red, lw = 2)
plot!(p, legend = :outerbottom, legendcolumns = 1, dpi=600, title = "Pseudo correlation between networks")
display(p)

estimated, history = graphhist(A; starting_assignment_rule = EigenStart(), maxitr = Int(1e8), stop_rule = PreviousBestValue(10000))
plot(history.history)

moments, indices = NetworkHistogram.get_moment_representation(estimated)


function get_p_matrix(θ, node_labels)
    n = length(node_labels)
    pij = zeros(n, n)
    for i in 1:n
        for j in 1:n
            pij[i, j] = θ[node_labels[i], node_labels[j]]
        end
    end
    return pij
end

p_moms = []
plots_pij = []
for (i, index) in enumerate(indices)
    push!(p_moms,
        heatmap(moments[:, :, i], xlabel = "Moment $index", xformatter = _ -> "",
            yformatter = _ -> "", aspect_ratio = :equal, axis = (
                [], false)))
    push!(plots_pij,
        heatmap(
            get_p_matrix(moments[:, :, i], estimated.node_labels), clims = (0, 1), xformatter = _ -> "",
            yformatter = _ -> "", xlabel = "Moment $index", aspect_ratio = :equal, axis = (
                [], false)))
end

p_networks = []
for i in 1:length(nums_to_analyze)
    push!(p_networks,
        heatmap(A[:, :, i], clims = (0, 1), legend = :none,
            xlabel = "Network $i", xformatter = _ -> "",
            yformatter = _ -> ""))
end

list_plots = []
for (i,mom) in enumerate(indices)
    if length(mom) == 1
        push!(list_plots,
            heatmap(
                get_p_matrix(moments[:, :, i], estimated.node_labels),
                clims = (0, 1),
                xformatter = _ -> "",
                yformatter = _ -> "",
                xlabel = "Moment $mom",
                legend = :none,
                axis = ([], false),
                )
            )
        push!(list_plots, p_networks[mom[1]])
    end
end

display(plot(list_plots..., layout = Plots.grid(3,6), size = (1200, 600)))



plots_pij_unscaled = []
for (i, index) in enumerate(indices)
    push!(plots_pij_unscaled,
        heatmap(
            get_p_matrix(moments[:, :, i], estimated.node_labels),
            xlabel = "Moment $index",
            xformatter=_->"",
            yformatter=_->"",
            aspect_ratio = :equal,
            axis = ([], false),
            #clims = (0, 1),
            )
        )
end

indices_interesting_moments = []
for i in eachindex(indices)
    if length(indices[i])>1 && maximum(moments[:,:,i]) > 0.2
        push!(indices_interesting_moments, i)
    end
end
println(length(indices_interesting_moments))



per_fig = 6
layout_tuple = (2, 3)
@assert layout_tuple[1] * layout_tuple[2] == per_fig
size_one_plot = 400
num_plots = length(indices_interesting_moments) ÷ per_fig
if length(indices_interesting_moments) % per_fig != 0
    num_plots += 1
end

for i in 1:num_plots
    display(plot(
        [plots_pij[k]
         for k in indices_interesting_moments[((i - 1) * per_fig + 1):min(
            i * per_fig, length(indices_interesting_moments))]]...,
        layout = Plots.grid(layout_tuple...),
        size = (layout_tuple[2]*(size_one_plot+1), layout_tuple[1]*size_one_plot),
        )
    )
    display(plot(
        [plots_pij_unscaled[k]
         for k in indices_interesting_moments[((i - 1) * per_fig + 1):min(
            i * per_fig, length(indices_interesting_moments))]]...,
        layout = Plots.grid(layout_tuple...),
        size = (layout_tuple[2]*(size_one_plot+1), layout_tuple[1]*size_one_plot),
        )
    )
end



moment_indices_to_test = [[1],[5],[6], [1,5], [1,6], [5,6], [1,5,6] ]
indices_commonly_used_networks_cross_moments = findall(
    x -> x in moment_indices_to_test, indices)
display(plot(
        [plots_pij_unscaled[i] for i in indices_commonly_used_networks_cross_moments]...,
        layout = Plots.grid(2,4),
        size = (1200, 600)
        )
)

l = @layout [a b c]
indices_commonly_used_networks = findall(x -> x in [[1],[5],[6]], indices)
width_plot = 400
display(plot([plots_pij_unscaled[i] for i in indices_commonly_used_networks]..., layout = l, size = (3*(width_plot+1), width_plot), margin = 5Plots.mm))
display(plot([p_moms[i] for i in indices_commonly_used_networks]...,
    layout = l, size = (3 * (width_plot + 1), width_plot), margin = 5Plots.mm))
