using NetworkHistogram
using CSV, DataFrames
using Dates
using CairoMakie
using MVBernoulli
using StatsBase

using Random
Random.seed!(1234)

include("utils.jl")


# if true, the incomplete observations are used
# otherwise, only the students that appeared in the facebook and friendship survey layers are considered
use_incomplete_observation = false


if !use_incomplete_observation
    postfix_file_name = "_complete"
else
    postfix_file_name = ""
end

# Load data
nodes_data = CSV.read("data/sp_high_school/diaries.csv/nodes.csv", DataFrame; normalizenames = true)
diaries = CSV.read("data/sp_high_school/diaries.csv/edges.csv", DataFrame; normalizenames = true)
facebook = CSV.read("data/sp_high_school/facebook.csv/edges.csv", DataFrame; normalizenames = true)
survey = CSV.read("data/sp_high_school/survey.csv/edges.csv", DataFrame; normalizenames = true)
contacts = CSV.read("data/sp_high_school/proximity.csv/edges.csv", DataFrame; normalizenames = true)
contacts.timestamps = unix2datetime.(contacts.time)
nb_time_points = length(unique(contacts.timestamps))


# indices start at 1
nodes_data._index .+= 1
for (i, df) in enumerate([diaries, facebook, survey, contacts])
    df._source .+= 1
    df.target .+= 1
end

# Create a multiplex network
n_nodes = size(nodes_data, 1)
A_all = zeros(n_nodes, n_nodes, 4)
layer_names_all = ["Contact diaries", "Facebook", "Friendship survey", "Face-to-face contact"]
layer_short_names_all = ["diaries", "facebook", "survey", "contact"]

for (i, df) in enumerate([diaries, facebook, survey, contacts])
    for row in eachrow(df)
        A_all[row._source, row.target, i] += 1
        A_all[row.target, row._source, i] += 1
    end
end


nodes_in = []
for i in 1:size(A_all, 3)
    nodes_inter = [x[1] for x in findall(sum(A_all[:, :, i], dims = (2, 3)) .> 0)]
    println("Layer $(layer_names_all[i]), number of nodes: $(length(nodes_inter))")
    push!(nodes_in, nodes_inter)
end


nodes_observed = []
for i in 1:n_nodes
    if i ∈ nodes_in[2] && i ∈ nodes_in[3]
        push!(nodes_observed, i)
    end
end


for i in 1:4
    println("Layer $(layer_names_all[i]), number of unique edge decorations: $(length(unique(A_all[:,:,i])))")
    println(" ", countmap(A_all[:,:,i]))
end


A_all[:, :, 2:3] = A_all[:, :, 2:3] .> 0
A_all[:, :, 1] = A_all[:, :, 1] .> 0

threshold = 100 ÷ 20
A_all[:, :, 4] = A_all[:, :, 4] .≥ threshold


layers_picked = [2,3,4]
layer_names = layer_names_all[layers_picked]
layer_short_names = layer_short_names_all[layers_picked]
A = deepcopy(A_all[:,:, layers_picked])
L = size(A, 3)

# find disconnected nodes
disconnected = [Tuple(coord)[1] for coord in findall(sum(A, dims = (2,3)) .== 0)]

if use_incomplete_observation
    to_drop = disconnected
else
    to_drop = unique([disconnected; setdiff(1:n_nodes, nodes_observed)])
end


# update indices
function update_indices(index, disconnected=to_drop)
    if index ∈ disconnected
        return 0
    end
    return index - sum(index .≥ disconnected)
end

transform!(nodes_data, :_index => ByRow(update_indices) => :_index)

# remove disconnected nodes
deleteat!(nodes_data, findall(nodes_data._index .== 0))
to_keep = setdiff(1:n_nodes, to_drop)
A = A[to_keep, to_keep, :]

@assert size(A, 1) == n_nodes - length(to_drop)
n = size(A, 1)

function get_type_of_class(class)
    result = ""
    if occursin("MP", class)
        result = "MP"
    elseif occursin("PC", class)
        result = "PC"
    elseif occursin("PSI", class)
        result = "PSI"
    elseif occursin("BIO", class)
        result = "2BIO"
    end
    return result
end




class_types = map(get_type_of_class, nodes_data.class)
all_classes = deepcopy(nodes_data.class)

function get_lines_for_labels(labels)
    classes_count = countmap(labels)
    classes_ordered = sort(collect(keys(classes_count)))
    println("Classes ordered: ", classes_ordered)
    lines = zeros(Int, length(classes_ordered) + 1)
    for (i, c) in enumerate(classes_ordered)
        lines[i+1] = lines[i] + classes_count[classes_ordered[i]]
    end
    return lines
end

lines = get_lines_for_labels(class_types)
lines_all = get_lines_for_labels(all_classes)

## Visualise the layers

function make_fig_layers(A, layer_names = ["Layer $i" for i in size(A, 3)], lines = :nothing)
    fig = Figure()
    axes = [Axis(fig[1, i], aspect = 1, title = layer_names[i]) for i in 1:size(A, 3)]
    for i in 1:size(A, 3)
        heatmap!(axes[i], A[:, :, i], colormap = :binary)
        if lines != :nothing
            for l in lines
                hlines!(axes[i], l, color = :red, linewidth = 0.5)
                vlines!(axes[i], l, color = :red, linewidth = 0.5)
            end
        end
    end
    hidedecorations!.(axes)
    return fig
end

fig = make_fig_layers(A, layer_names)
display(fig)

# sort by class
sorted_by_class = sortperm(all_classes)

fig_sorted = make_fig_layers(A[sorted_by_class, sorted_by_class, :], layer_names, lines)
display(fig_sorted)


heatmap(A[sorted_by_class, sorted_by_class, end], colormap = :binary)

## Fit the model

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

estimated, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A;
                                                                      max_iterations_stalled = 20,
                                                                      n_min =1)
block_estimated = NetworkHistogram.GraphShapeHist(estimator)


estimated_original = deepcopy(estimated)
block_estimated_original = deepcopy(block_estimated)


## sort the estimated labels
estimated = deepcopy(estimated_original)
block_estimated = deepcopy(block_estimated_original)

marginals_not_sorted = MVBernoulli.marginals.(MVBernoulli.from_tabulation.(estimated.θ))


if use_incomplete_observation
    order_sort = [L,L-2,L-1]
else
    order_sort = [L]
end

 function_to_sort(x, order_sort = order_sort) = marginals_not_sorted[x, x][order_sort]
    sort_by(x) = (function_to_sort(x)..., x)

sorted_groups = sortperm(1:length(unique(estimated.node_labels)), rev = true,
    by = sort_by)
for i in eachindex(estimated.node_labels)
    estimated.node_labels[i] = findfirst(x -> x == estimated.node_labels[i], sorted_groups)
    block_estimated.node_labels[i] = findfirst(x -> x == block_estimated.node_labels[i], sorted_groups)
end
estimated.θ .= estimated.θ[sorted_groups, sorted_groups]
block_estimated.θ .= block_estimated.θ[sorted_groups, sorted_groups]


mvberns = MVBernoulli.from_tabulation.(estimated.θ)
marginals = MVBernoulli.marginals.(mvberns)



P = zeros(n, n, L)
for i in 1:L
    P[:, :, i] = get_p_matrix([m[i] for m in marginals], estimated.node_labels)
end

sorted_by_fit = sortperm(estimated.node_labels)


fit_and_class = [(estimated.node_labels[i], identity(nodes_data.class[i]))
                 for i in 1:n]
sorted_by_class_then_fit = sortperm(fit_and_class,
    lt = (x, y) -> x[2] < y[2] ||
        (x[2] == y[2] && x[1] < y[1]))


sortings = [sorted_by_fit, sorted_by_class_then_fit, sorted_by_class]
names = ["Sorted by fit", "Sorted by class then fit", "Sorted by class"]

for (k,sorting_indices) in enumerate(sortings)
    with_theme(theme_latexfonts()) do
        fig = Figure(size = (130+200*L, 250), fontsize = 14)
        colormap = :binary
        color_minor_lines = :blue
        color_major_lines = :black
        axes = [Axis(fig[1, i],
                    aspect = 1,
                    title = layer_names[i],
                    xminorticks = lines_all[2:end-1],
                    xticks = lines[2:(end - 1)],
                    yminorticks = lines_all[2:(end - 1)],
                    yticks = lines[2:(end - 1)],
                    xminorticksvisible = true,
                    yminorticksvisible = true,
                    xminorticksize = 3,
                    yminorticksize = 3,
                    xticksize = 5,
                    yticksize = 5,)
                for i in 1:L]
        hidedecorations!.(axes; ticks = false, minorticks = false, grid = false, minorgrid=false)
        for i in 1:L
            hm = heatmap!(axes[i], P[sorting_indices, sorting_indices, i], colormap = colormap, colorrange = (0, 1))
            if occursin("class", names[k])
                for l in lines_all
                    hlines!(axes[i], l, color = color_minor_lines, linewidth = 0)
                    vlines!(axes[i], l, color = color_minor_lines, linewidth = 0)
                end
                for l in lines
                    hlines!(axes[i], l, color = color_major_lines, linewidth = 0.6, alpha = 0.5, linestyle = :dash)
                    vlines!(axes[i], l, color = color_major_lines, linewidth = 0.6, alpha = 0.5, linestyle = :dash)
                end
            end
        end
        Colorbar(fig[1, end+1], colorrange = (0, 1),
            colormap = colormap, vertical = true, flipaxis = true, height = Relative(0.7))
        display(fig)
        if names[k] == "Sorted by class then fit"
            save(joinpath(@__DIR__, "sp_high_school$postfix_file_name.png"), fig, px_per_unit = 2)
        elseif names[k] == "Sorted by class"
            save(joinpath(@__DIR__, "sp_high_school_sorted_by_class$postfix_file_name.png"), fig, px_per_unit = 2)
        end
    end
end

##

with_theme(theme_latexfonts()) do
    fig = Figure(size = (130 + 200 * L, 250), fontsize = 14)
    colormap = :binary
    color_minor_lines = :blue
    color_major_lines = :black
    axes = [Axis(fig[1, i],
                aspect = 1,
                title = layer_names[i],
                xminorticks = lines_all[2:(end - 1)],
                xticks = lines[2:(end - 1)],
                yminorticks = lines_all[2:(end - 1)],
                yticks = lines[2:(end - 1)],
                xminorticksvisible = true,
                yminorticksvisible = true,
                xminorticksize = 3,
                yminorticksize = 3,
                xticksize = 5,
                yticksize = 5)
            for i in 1:L]
    hidedecorations!.(
        axes; ticks = false, minorticks = false, grid = false, minorgrid = false)
    for i in 1:L
        hm = heatmap!(axes[i], A[sorted_by_class_then_fit, sorted_by_class_then_fit, i],
            colormap = colormap, colorrange = (0, 1))
    end
    display(fig)
end



# Commented block
k = NetworkHistogram.get_num_blocks(estimator)
indices_shapes = CartesianIndex.((i, j) for j in 1:k for i in 1:k if i ≥ j)
shape_community = zeros(Int, k,k)
shapes_unique = unique(estimated.θ)
shape_ordered = sort(shapes_unique, lt = (x, y) -> x[1] < y[1], rev = true)
for (i, index) in enumerate(indices_shapes)
    label = findfirst(x -> x == estimated.θ[index], shape_ordered)
    shape_community[index] = label
    shape_community[index[2], index[1]] = label
end


proba_not_connected = get_p_matrix([shapes_unique[i][1] for i in shape_community], estimated.node_labels)

community = get_p_matrix(shape_community, estimated.node_labels)

for sorting in sortings[2:2]
    fig,ax,hm = heatmap(community[sorting, sorting],
        colormap = Makie.Categorical(:Paired_11), show_axis = false)
    for l in lines_all
        hlines!(ax, l, color = :black, linewidth = 0.5)
        vlines!(ax, l, color = :black, linewidth = 0.5)
    end
    Colorbar(fig[:, end + 1], hm, )
    display(fig)
end



for sorting in sortings[2:2]
    fig, ax, hm = heatmap(proba_not_connected[sorting, sorting],
        colormap = :binary, show_axis = false, colorrange = (0, 1))
    for l in lines_all
        hlines!(ax, l, color = :red, linewidth = 0.5)
        vlines!(ax, l, color = :red, linewidth = 0.5)
    end
    Colorbar(fig[:, end + 1], hm)#, colorrange = (0, 1))
    display(fig)
end






## conditional probabilties

layers_to_consider = [L-2,L-1,L]
colormap = :binary
sorting_cond_proba = sorted_by_class_then_fit
mean_cond_proba = ones(length(layers_to_consider), length(layers_to_consider))

with_theme(theme_latexfonts()) do
    fig = Figure(size = (600, 600))

    for (index_1, layer_1) in enumerate(layers_to_consider)
        for (index_2,layer_2) in enumerate(layers_to_consider)
            if layer_1 == layer_2
                continue
            end
            x = zeros(Union{Bool,Missing}, L)
            y = zeros(Union{Bool,Missing}, L)
            x .= missing
            y .= missing
            x[layer_1] = true
            y[layer_2] = true
            cond_proba = get_p_matrix(
                [MVBernoulli.conditional_proba(d, x, y)
                for d in MVBernoulli.from_tabulation.(estimated.θ)],
                estimated.node_labels)
            ax = Axis(fig[index_1, index_2], aspect = 1, title = "$(layer_short_names[layer_1]) | $(layer_short_names[layer_2])")
            heatmap!(ax, cond_proba[sorting_cond_proba, sorting_cond_proba],
                colormap = colormap, colorrange = (0, 1))
            hidedecorations!(ax)
            mean_cond_proba[index_1,index_2] = mean(cond_proba[findall(!isnan, cond_proba)])
        end
    end
    Colorbar(fig[2, end + 1], colorrange = (0, 1),
        colormap = colormap, vertical = true, flipaxis = true, height = Relative(1.2))
    rowgap!(fig.layout, Relative(0.01))
    colgap!(fig.layout, Relative(0.01))
    display(fig)
    save(joinpath(@__DIR__, "sp_high_school_cond_proba$postfix_file_name.png"), fig, px_per_unit = 2)
end



# xaxis given y axis for my conditional probabilities

mean_cond_proba_paper = [1.0 0.96 0.69; 0.31 1.0 0.3; 0.66 0.91 1.0]
fig = Figure(size = (600, 300))

# use transpose to get yaxis in term of xaxis as in the original paper (and more natural)
for (i,cond) in enumerate(transpose.([mean_cond_proba, mean_cond_proba_paper]))
    ax = Axis(fig[1, i], aspect = 1, title = i== 1 ? "Mean estimated" : "Original paper",
        xticks = (1:length(layers_to_consider), layer_short_names[layers_to_consider]),
        yticks = (1:length(layers_to_consider), layer_short_names[layers_to_consider])
    )
    heatmap!(ax, cond, colormap = colormap, colorrange = (0, 1))
    for i in 1:length(layers_to_consider), j in 1:length(layers_to_consider)
        txtcolor = cond[i, j] < 0.55 ? :black : :white
        text!(ax, "$(round(cond[i,j], digits = 2))", position = (i, j),
            color = txtcolor, align = (:center, :center))
    end
end
#ax.xticklabelalign = (:center, :center)
#ax.yticklabelalign = (:right, :center)
display(fig)
save(joinpath(@__DIR__, "sp_high_school_mean_cond_proba$postfix_file_name.png"),
    fig, px_per_unit = 2)

## Comparison with fit only on the contact layer

A_contact = deepcopy(A[:,:,end])

# find disconnected nodes
disconnected = [Tuple(coord)[1] for coord in findall(sum(A_contact, dims = (2)) .== 0)]

# update indices
function update_indices(index, disconnected = disconnected)
    if index ∈ disconnected
        return 0
    end
    return index - sum(index .≥ disconnected)
end

nodes_data_contact = deepcopy(nodes_data)

transform!(nodes_data_contact, :_index => ByRow(update_indices) => :_index)

# remove disconnected nodes
deleteat!(nodes_data_contact, findall(nodes_data_contact._index .== 0))
A_contact = NetworkHistogram.drop_disconnected_components(A_contact)

classes_contact = deepcopy(nodes_data_contact.class)
class_types_contact = map(get_type_of_class,classes_contact)
lines_contact = get_lines_for_labels(class_types_contact)
lines_all_contact = get_lines_for_labels(classes_contact)


estimator_contact, history_contact = graphhist(A_contact;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e8),
    stop_rule = PreviousBestValue(1_000_0))

fig = Figure()
best_ll = round(NetworkHistogram.get_bestitr(history_contact)[2], sigdigits = 4)
ax = Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Log-likelihood",
    title = "Log-likelihood: $(best_ll)")
lines!(ax, get(history_contact.history, :best_likelihood)...)
display(fig)


estimated_contact, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator_contact, A_contact;
    max_iterations_stalled = 20,
    n_min = 1)
estimated_contact = NetworkHistogram.GraphShapeHist(estimator_contact)

estimated_original_contact = deepcopy(estimated_contact)

## sort the estimated labels
estimated_contact = deepcopy(estimated_original_contact)

marginals_not_sorted = [m[1] for m in estimated_contact.θ]

sort_by(x) = (marginals_not_sorted[x, x], x)

sorted_groups = sortperm(1:length(unique(estimated_contact.node_labels)), rev = true,
    by = sort_by)
for i in eachindex(estimated_contact.node_labels)
    estimated_contact.node_labels[i] = findfirst(x -> x == estimated_contact.node_labels[i], sorted_groups)
end
estimated_contact.θ .= estimated_contact.θ[sorted_groups, sorted_groups]


P_contact = get_p_matrix([m[1] for m in estimated_contact.θ], estimated_contact.node_labels)



fit_and_class_contact = [(estimated_contact.node_labels[i], classes_contact[i])
                 for i in 1:size(estimated_contact.node_labels,1)]
sorted_by_class_then_fit_contact = sortperm(fit_and_class_contact,
    lt = (x, y) -> x[2] < y[2] ||
        (x[2] == y[2] && x[1] < y[1]))
sorted_by_class_contact = 1:size(estimated_contact.node_labels,1)

#sorted_by_class_then_fit_contact = setdiff(sorted_by_class_then_fit, isolated_contact)
#slide_index(x) = x - sum(x .≥ isolated_contact)
#sorted_by_class_then_fit_contact = slide_index.(sorted_by_class_then_fit_contact)
sorted_obs = A[sorted_by_class_then_fit, sorted_by_class_then_fit, end]

sorted_obs_contact = A_contact[
    sorted_by_class_then_fit_contact, sorted_by_class_then_fit_contact]


titles = ["Wearables only", "All layers"]

with_theme(theme_latexfonts()) do
    fig = Figure(size = (600, 300), fontsize = 16)
    colormap = :lipari
    color_minor_lines = :blue
    color_major_lines = :black
    axes = [Axis(fig[1, i],
                aspect = 1,
                title = titles[i],
                xminorticks = lines_all[2:(end - 1)],
                xticks = lines[2:(end - 1)],
                yminorticks = lines_all[2:(end - 1)],
                yticks = lines[2:(end - 1)],
                xminorticksvisible = true,
                yminorticksvisible = true,
                xminorticksize = 3,
                yminorticksize = 3,
                xticksize = 5,
                yticksize = 5)
            for i in 1:2]
    hidedecorations!.(
        axes; ticks = false, minorticks = false, grid = false, minorgrid = false)
    for i in 1:2
        if i == 2
            hm = heatmap!(
                axes[i], sorted_obs,
                colormap = :binary)
        elseif i == 1
            hm = heatmap!(
                axes[i], sorted_obs_contact,
                colormap = :binary)
        end
    end
    display(fig)
    save(joinpath(@__DIR__, "sp_high_school_contact_sorted_obs$postfix_file_name.png"), fig, px_per_unit = 2)
end

##

titles = [ "Only face-to-face", "All layers","",""]
y_labels = ["Estimated proba","", "Recorded contacts",""]
to_plot = [P_contact[sorted_by_class_then_fit_contact, sorted_by_class_then_fit_contact],
    P[sorted_by_class_then_fit, sorted_by_class_then_fit, end], sorted_obs_contact, sorted_obs]
indices = [(1, 1), (1, 2), (2, 1), (2, 2)]
with_theme(theme_latexfonts()) do
    fig = Figure(size = (600, 550), fontsize = 16)
    colormap = :binary
    color_minor_lines = :blue
    color_major_lines = :black
    axes = [Axis(fig[indices[i]...],
                aspect = 1,
                title = titles[i],
                ylabel = y_labels[i],
                xminorticks = lines_all[2:(end - 1)],
                xticks = lines[2:(end - 1)],
                yminorticks = lines_all[2:(end - 1)],
                yticks = lines[2:(end - 1)],
                xminorticksvisible = true,
                yminorticksvisible = true,
                xminorticksize = 3,
                yminorticksize = 3,
                xticksize = 5,
                yticksize = 5)
            for i in 1:4]
    hidedecorations!.(
        axes; ticks = false, minorticks = false, grid = false, minorgrid = false, label = false)
    for i in 1:4
        heatmap!(axes[i], to_plot[i], colormap = colormap, colorrange = (0, 1))
        if i == 1
            for l in lines_contact
                hlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
                vlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
            end
        elseif  i == 2
            for l in lines
                hlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
                vlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
            end
        end
    end
    Colorbar(fig[1, end + 1], colorrange = (0, 1),
        colormap = colormap, vertical = true, flipaxis = true, height = Relative(0.7))
    display(fig)
    save(joinpath(@__DIR__, "sp_high_school_contact$postfix_file_name.png"), fig, px_per_unit = 2)
end


##

titles = ["Only face-to-face", "All layers"]

with_theme(theme_latexfonts()) do
    fig = Figure(size = (130 + 200 * 2, 250), fontsize = 16)
    colormap = :binary
    color_minor_lines = :blue
    color_major_lines = :black
    axes = [Axis(fig[1, i],
                aspect = 1,
                title = titles[i],
                xminorticks = lines_all[2:(end - 1)],
                xticks = lines[2:(end - 1)],
                yminorticks = lines_all[2:(end - 1)],
                yticks = lines[2:(end - 1)],
                xminorticksvisible = true,
                yminorticksvisible = true,
                xminorticksize = 3,
                yminorticksize = 3,
                xticksize = 5,
                yticksize = 5)
            for i in 1:2]
    hidedecorations!.(
        axes; ticks = false, minorticks = false, grid = false, minorgrid = false)
    for i in 1:2
        if i == 1
            hm = heatmap!(axes[i],
                P_contact[
                    sorted_by_class_then_fit_contact, sorted_by_class_then_fit_contact],
                colormap = colormap, colorrange = (0, 1))
            for l in lines_contact
                hlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
                vlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
            end
        else
            hm = heatmap!(
                axes[i], P[sorted_by_class_then_fit, sorted_by_class_then_fit, end],
                colormap = colormap, colorrange = (0, 1))
            for l in lines
                hlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
                vlines!(axes[i], l, color = color_major_lines,
                    linewidth = 0.6, alpha = 0.5, linestyle = :dash)
            end
        end

    end
    Colorbar(fig[1, end + 1], colorrange = (0, 1),
        colormap = colormap, vertical = true, flipaxis = true, height = Relative(0.7))
    display(fig)
    save(joinpath(@__DIR__, "sp_high_school_contact_just_fit$postfix_file_name.png"), fig, px_per_unit = 2)
end

# Commented block
# k = NetworkHistogram.get_num_blocks(estimator_contact)
# indices_shapes = CartesianIndex.((i, j) for j in 1:k for i in 1:k if i ≥ j)
# shape_community = zeros(Int, k, k)
# shape_ordered = sort(unique(estimated_contact.θ), lt = (x, y) -> x[1] < y[1], rev = false)
# for (i, index) in enumerate(indices_shapes)
#     label = findfirst(x -> x == estimated_contact.θ[index], shape_ordered)
#     shape_community[index] = label
#     shape_community[index[2], index[1]] = label
# end

# community = get_p_matrix(shape_community, estimated_contact.node_labels)

# fig, ax, hm = heatmap(community[sorted_by_class_then_fit_contact, sorted_by_class_then_fit_contact],
#     colormap = Makie.Categorical(:Paired_11), show_axis = false)
# Colorbar(fig[:, end + 1], hm)
# display(fig)
