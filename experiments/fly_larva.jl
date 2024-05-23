using NetworkHistogram
using CairoMakie
using DataFrames
using CSV
using Statistics
using StatsBase

include("utils.jl")


data_dir = joinpath(@__DIR__, "../data/fly_larva/")
path_to_data = joinpath(data_dir, "edges.csv")


##
# read the clusters
path_to_clusters = joinpath(data_dir, "nodes.csv")
df_nodes = CSV.read(path_to_clusters, DataFrame)
df_nodes."# index" .+= 1


# Load the data
n = maximum(df_nodes."# index")
num_layers = 3
A = Array{Int}(undef, n, n, num_layers)

for i in 1:n
    for j in 1:n
        A[i, j, :] .= zeros(Int, num_layers)
    end
end

etype_to_layers = Dict(
    "aa" => 2,
    "da" => 3,
    "dd" => 1,
    "ad" => 3,
)

for line in readlines(path_to_data)
    if occursin("#", line)
        continue
    end
    i, j, count, type = split(line, ',')
    i = parse(Int, i)+1
    j = parse(Int, j)+1
    l = etype_to_layers[type]
    if i!=j
        A[i, j, l] = 1
        A[j, i, l] = 1
    end
end

## preprocess data

degrees = dropdims(sum(A, dims = (2)),dims=2)
# only keep nodes with at least 1 connection in each layer
threshold = 1
nodes_deg_geq_1 = findall(x -> all(x .≥ threshold), eachrow(degrees))

A = A[nodes_deg_geq_1, nodes_deg_geq_1, :]
n = size(A, 1)
## eplore the data
hemishphere = filter(x -> x."# index" ∈ nodes_deg_geq_1, df_nodes)." hemisphere"
hemishphere_map = Dict("left" => 1, "right" => 2, missing => 3)
hemishphere = [hemishphere_map[h] for h in hemishphere]


# cell types ordering
cell_types = filter(x -> x."# index" ∈ nodes_deg_geq_1, df_nodes)." cell_type"
replace!(cell_types, missing => "missing")


cell_types_degrees = [mean(A[findall(cell_types .== type), :, :])
                      for type in unique(cell_types)]

cell_type_order = unique(cell_types)[sortperm(cell_types_degrees, rev = true)]
tuple_cell_type_degree = [(findfirst(s -> s == cell_types[i], cell_type_order), degrees[i])
                    for i in 1:size(A, 1)]
sorting_by_cell_type = sortperm(tuple_cell_type_degree,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] > y[2]))

# cluster from paper ordering
categories = filter(x -> x."# index"  ∈ nodes_deg_geq_1, df_nodes)." cluster"
categories_degree = [mean(A[findall(categories .== cat), :, :])
                     for cat in unique(categories)]

categories_order = unique(categories)[sortperm(categories_degree, rev = true)]
degrees_flat = vec(sum(A, dims = (2, 3)))
tuple_cat_degree = [(findfirst(s -> s == categories[i], categories_order), degrees[i])
                    for i in 1:size(A, 1)]

sorting_by_category = sortperm(tuple_cat_degree,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] > y[2]))

sorting_by_degree = sortperm(degrees_flat, rev = true)


## Plot of data

fig = Figure()
for i in 1:num_layers
    ax = Axis(fig[1, i], aspect = 1, title = "Layer $i")
    heatmap!(ax, A[:, :, i], colormap = :binary)
    hidedecorations!(ax)
end
display(fig)

fig = Figure()
for i in 1:num_layers
    ax = Axis(fig[1, i], aspect = 1, title = "Layer $i")
    heatmap!(ax, A[sorting_by_category, sorting_by_category, i], colormap = :binary)
    hidedecorations!(ax)
end
display(fig)

fig = Figure()
for i in 1:num_layers
    ax = Axis(fig[1, i], aspect = 1, title = "Layer $i")
    heatmap!(ax, A[sorting_by_degree, sorting_by_degree, i], colormap = :binary)
    hidedecorations!(ax)
end
display(fig)

## fit histogram estimator
using NetworkHistogram

estimator, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e8),
    stop_rule = PreviousBestValue(1_000_0))

fig = Figure()
best_ll = round(NetworkHistogram.get_bestitr(history)[2], sigdigits = 4)
ax = Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Log-likelihood",
    title = "Log-likelihood: $(best_ll)")
lines!(ax, get(history.history, :best_likelihood)...)
display(fig)

estimmated_block = NetworkHistogram.GraphShapeHist(estimator)

max_shapes = length(unique(estimmated_block.θ))
min_shapes = size(estimmated_block.θ, 1)

##

max_shapes = 100
estimated, bic = NetworkHistogram.get_best_smoothed_estimator(estimator,A; n_max = max_shapes, n_min = min_shapes)


## extract the marginals and correlations
using MVBernoulli

mvberns = MVBernoulli.from_tabulation.(estimated.θ)
marginals = MVBernoulli.marginals.(mvberns)
correlation = [c[3] for c in MVBernoulli.correlation_matrix.(mvberns)]


sorting_by(x) =  (marginals[x, x][2], x)

sorted_groups = sortperm(1:length(unique(estimated.node_labels)), rev = true,
    by = x -> sorting_by(x)
)

sorted_nodes = sortperm(
    estimated.node_labels,
    rev = true,
    by = x -> sorting_by(x)
)

for i in eachindex(estimated.node_labels)
    estimated.node_labels[i] = findfirst(x -> x == estimated.node_labels[i], sorted_groups)
end
estimated.θ .= estimated.θ[sorted_groups, sorted_groups]

tuple_hemisphere_label = [(hemishphere[i], estimated.node_labels[i])
                          for i in 1:size(A, 1)]
sorted_labels_hemisphere = sortperm(tuple_hemisphere_label,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] < y[2]))



tuple_hemisphere_cell_type = [(hemishphere[i], cell_types[i])
                          for i in 1:size(A, 1)]
sorted_types_hemisphere = sortperm(tuple_hemisphere_cell_type,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] < y[2]))


sorted_labels = sortperm(estimated.node_labels, rev = false)

@assert sorted_labels == sorted_nodes

# check that the ordering is correct
@assert mvberns[sorted_groups, sorted_groups] == MVBernoulli.from_tabulation.(estimated.θ)

##


fig = Figure(size=(800,800))
ax = Axis(fig[1, 1], aspect = 1, title = "a, cat")
ax2 = Axis(fig[1, 2], aspect = 1, title = "d, cat")
ax3 = Axis(fig[2, 1], aspect = 1, title = "a fit")
ax4 = Axis(fig[2, 2], aspect = 1, title = "d fit")
heatmap!(ax, A[sorting_by_category, sorting_by_category, 1], colormap = :binary)
heatmap!(ax2, A[sorting_by_category, sorting_by_category, 2], colormap = :binary)
heatmap!(ax3, A[sorted_labels, sorted_labels, 1], colormap = :binary)
heatmap!(ax4, A[sorted_labels, sorted_labels, 2], colormap = :binary)
display(fig)


##


mvberns_sorted = mvberns[sorted_groups, sorted_groups]
marginals_sorted = marginals[sorted_groups, sorted_groups]
correlation_sorted = correlation[sorted_groups, sorted_groups]

P = zeros(n, n, 3)
P[:, :, 1] = get_p_matrix([m[1] for m in marginals_sorted], estimated.node_labels)
P[:, :, 2] = get_p_matrix([m[2] for m in marginals_sorted], estimated.node_labels)
P[:, :, 3] = get_p_matrix([m[3] for m in marginals_sorted], estimated.node_labels)


A_plot_big = zeros(n, n)
A_plot_big[findall(A[:,:,1] .== 1)] .= 1
A_plot_big[findall(A[:,:,2] .== 1)] .= 2
A_plot_big[findall(A[:,:,3] .== 1)] .= 3
A_plot = A_plot_big# dropdims(sum(A_plot_big, dims = 3), dims = 3)
colornames = ["None", findfirst(x -> x == 1, etype_to_layers), findfirst(x -> x == 2, etype_to_layers), findfirst(x -> x == 3, etype_to_layers)]


##

A_plot_updated = deepcopy(A_plot) .* 1


reverse_colormaps = true
colormap = :okabe_ito

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 700))
    if reverse_colormaps
        cmap_in_heatmap = Makie.Categorical(Reverse(colormap))
        cmap_color_bar = Reverse(cgrad(colormap, 4, categorical = true))
    else
        cmap_in_heatmap = Makie.Categorical(colormap)
        cmap_color_bar = cgrad(colormap, 4, categorical = true)
    end
    ax = Axis(fig[1, 1], aspect = 1, title = "Sorted by histogram")
    ax2 = Axis(fig[1, 2], aspect = 1, title = "Sorted by cell type")
    ax3 = Axis(fig[2, 1], aspect = 1, title = "Sorted by histogram and hemisphere")
    ax4 = Axis(fig[2, 2], aspect = 1, title = "Sorted by cell type and hemisphere")

    heatmap!(ax, A_plot_updated[sorted_labels, sorted_labels],
        colormap = cmap_in_heatmap)
    heatmap!(ax2, A_plot_updated[sorting_by_cell_type, sorting_by_cell_type],
        colormap = cmap_in_heatmap)
    pl = heatmap!(ax3, A_plot_updated[sorted_labels_hemisphere, sorted_labels_hemisphere],
        colormap = cmap_in_heatmap)
    heatmap!(ax4, A_plot_updated[sorted_types_hemisphere, sorted_types_hemisphere],
        colormap = cmap_in_heatmap)
    hidedecorations!.([ax, ax2, ax3, ax4], label = false)

    # add lines for hemisphere
    counts = countmap(hemishphere)
    for ax in [ax3, ax4]
        vlines!(ax, cumsum([counts[1], counts[2]]), color = :white, linewidth = 0.5)
        hlines!(ax, cumsum([counts[1], counts[2]]), color = :white, linewidth = 0.5)
    end



    cb = Colorbar(fig[:, end+1];
        colormap = cmap_color_bar,
        limits = (0, 4),
        label = "Type of connection",
        vertical = true, height = Relative(0.7), flipaxis = true,
        ticks = ([0.5, 1.5, 2.5, 3.5], colornames))
    display(fig)
    save("fly_larva.png", fig, px_per_unit = 1)
end


##


P_power = P .^1

with_theme(theme_latexfonts()) do
    fig = Figure(size = (850, 400), fontsize = 16)
    colormap = :lipari
    ax = Axis(fig[1, 1], aspect = 1, title = findfirst(x -> x == 1, etype_to_layers))
    ax1 = Axis(fig[1, 2], aspect = 1, title = findfirst(x -> x == 2, etype_to_layers))
    ax2 = Axis(fig[1, 3], aspect = 1, title = findfirst(x -> x == 3, etype_to_layers))
    hidedecorations!.([ax, ax1, ax2])

    for (i, ax) in enumerate([ax, ax1, ax2])
        heatmap!(ax, P_power[sorted_labels, sorted_labels, i],
            colormap = colormap, colorrange = (0, 1))
    end
    Colorbar(fig[2, 2], colorrange = (0, 1),
        colormap = colormap, vertical = false, flipaxis = false, width = Relative(0.7), label = "Probability")
    display(fig)
end

using StatsBase

with_theme(theme_latexfonts()) do
    fig = Figure(size = (850, 400), fontsize = 16)
    colormap = :lipari
    ax = Axis(fig[1, 1], aspect = 1, title = findfirst(x -> x == 1, etype_to_layers))
    ax1 = Axis(fig[1, 2], aspect = 1, title = findfirst(x -> x == 2, etype_to_layers))
    ax2 = Axis(fig[1, 3], aspect = 1, title = findfirst(x -> x == 3, etype_to_layers))
    hidedecorations!.([ax, ax1, ax2])
    counts = countmap(hemishphere)
    for (i, ax) in enumerate([ax, ax1, ax2])
        heatmap!(ax, P_power[sorted_labels_hemisphere, sorted_labels_hemisphere, i],
            colormap = colormap, colorrange = (0, 1))
        vlines!(ax, cumsum([counts[1], counts[2]]), color = :white, linewidth = 0.5)
        hlines!(ax, cumsum([counts[1], counts[2]]), color = :white, linewidth = 0.5)
    end
    Colorbar(fig[2, 2], colorrange = (0, 1),
        colormap = colormap, vertical = false, flipaxis = false, width = Relative(0.7), label = "Probability")
    display(fig)
end
