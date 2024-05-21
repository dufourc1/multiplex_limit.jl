using CSV, JLD
using DataFrames
using Statistics
using CairoMakie
using Random
Random.seed!(1234)

##

# Load the data
path_to_data_folder = joinpath(@__DIR__, "../data/", "MultiplexDiseasome-master")

omim_dataset = CSV.read(
    joinpath(path_to_data_folder, "Datasets", "DSG_network_from_OMIM.csv"), DataFrame)


function get_genotype_and_symptoms_cooccurrence_matrix(dataset::DataFrame, file::String)
    if isfile(file)
        loaded = load(file)
        return loaded["node_names"], loaded["adj_genotype"], loaded["adj_symptoms"]
    end
    n = length(unique(dataset.disorder))
    diseases_names = sort(unique(dataset.disorder))

    adj_genotype = zeros(n, n)
    adj_symptoms = zeros(n, n)

    for i in 1:(n - 1)
        df_i = filter(x -> x.disorder == diseases_names[i], dataset)
        for j in (i + 1):n
            df_j = filter(x -> x.disorder == diseases_names[j], dataset)
            adj_genotype[i, j] = count_matches(df_i.gene_symb, df_j.gene_symb)
            adj_symptoms[i, j] = count_matches(df_i.symptom, df_j.symptom)
            adj_genotype[j, i] = adj_genotype[i, j]
            adj_symptoms[j, i] = adj_symptoms[i, j]
        end
    end

    save(file, "node_names", diseases_names, "adj_genotype",
        adj_genotype, "adj_symptoms", adj_symptoms)
    return diseases_names, adj_genotype, adj_symptoms
end

diseases_names, adj_genotype, adj_symptoms = get_genotype_and_symptoms_cooccurrence_matrix(
    omim_dataset, joinpath(path_to_data_folder, "adj_omim_2.jld"))

##

# build the multiplex network
A_all_ = zeros(length(diseases_names), length(diseases_names), 2)
A_all_[:, :, 1] = adj_genotype
A_all_[:, :, 2] = adj_symptoms

# binarize the adjacency matrices
A_all = Int.(A_all_ .> 0)
# compute the degrees on each layers
degrees = dropdims(sum(A_all, dims = (2)), dims = 2)

# only keep nodes with at least 2 connections in each layer
threshold = 2
nodes_deg_geq_2 = findall(x -> x[1] ≥ threshold && x[2] ≥ threshold, eachrow(degrees))

A_inter = A_all[nodes_deg_geq_2, nodes_deg_geq_2, :]
diseases_names_inter = diseases_names[nodes_deg_geq_2]

# drop isolated nodes
non_isolated_nodes = findall(x -> x ≥ 1, vec(sum(A_inter, dims = (2, 3))))

A = A_inter[non_isolated_nodes, non_isolated_nodes, :]
diseases_names = diseases_names_inter[non_isolated_nodes]

n = size(A, 1)
#slow but can't be bothered to optimize
categories = Array{String}(undef, n)
for (i, name) in enumerate(diseases_names)
    category_raw = filter(x -> x.disorder == name, omim_dataset).disorder_cat
    if isempty(category_raw) || ismissing(category_raw[1])
        categories[i] = "z-Unknown"
    else
        categories[i] = category_raw[1]
    end
end

categories_degree = [mean(A[findall(categories .== cat), :, :])
                     for cat in unique(categories)]
categories_order = unique(categories)[sortperm(categories_degree, rev = true)]

degrees = vec(sum(A, dims = (2, 3)))
tuple_cat_degree = [(findfirst(s -> s == categories[i], categories_order), degrees[i])
                    for i in 1:size(A, 1)]

sorting_by_category = sortperm(tuple_cat_degree,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] > y[2]))

sorting_by_degree = sortperm(degrees, rev = true)


## Plot of data

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype, threshold = $threshold")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms, threshold = $threshold")
heatmap!(ax, A[:, :, 1], colormap = :binary)
heatmap!(ax2, A[:, :, 2], colormap = :binary)
display(fig)


fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype, threshold = $threshold")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms, threshold = $threshold")
heatmap!(ax, A[sorting_by_category,sorting_by_category, 1], colormap = :binary)
heatmap!(ax2, A[sorting_by_category, sorting_by_category, 2], colormap = :binary)
display(fig)

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype, threshold = $threshold")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms, threshold = $threshold")
heatmap!(ax, A[sorting_by_degree, sorting_by_degree, 1], colormap = :binary)
heatmap!(ax2, A[sorting_by_degree, sorting_by_degree, 2], colormap = :binary)
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

estimated, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)


## extract the marginals and correlations
using MVBernoulli


mvberns = MVBernoulli.from_tabulation.(estimated.θ)
marginals = MVBernoulli.marginals.(mvberns)
correlation = [c[3] for c in MVBernoulli.correlation_matrix.(mvberns)]


sorted_groups = sortperm(1:length(unique(estimated.node_labels)), rev = true,
    by = x -> (marginals[x, x][2], correlation[x,x], x))

sorted_nodes = sortperm(estimated.node_labels, rev = true, by = x -> (marginals[x, x][2], correlation[x,x], x))

for i in eachindex(estimated.node_labels)
    estimated.node_labels[i] = findfirst(x->x==estimated.node_labels[i], sorted_groups)
end
estimated.θ .= estimated.θ[sorted_groups, sorted_groups]
sorted_labels = sortperm(estimated.node_labels, rev = false)

@assert sorted_labels == sorted_nodes

# check that the ordering is correct
@assert mvberns[sorted_groups, sorted_groups] ==  MVBernoulli.from_tabulation.(estimated.θ)

mvberns_sorted = mvberns[sorted_groups, sorted_groups]
marginals_sorted = marginals[sorted_groups, sorted_groups]
correlation_sorted = correlation[sorted_groups, sorted_groups]

include("utils.jl")
P = zeros(n, n, 3)
P[:, :, 1] = get_p_matrix([m[1] for m in marginals_sorted], estimated.node_labels)
P[:, :, 2] = get_p_matrix([m[2] for m in marginals_sorted], estimated.node_labels)
P[:, :, 3] = get_p_matrix(correlation_sorted, estimated.node_labels)

## main plot
A_plot_big = deepcopy(A)
A_plot_big[:, :, 1] .*= 1
A_plot_big[:, :, 2] .*= 2
A_plot = dropdims(sum(A_plot_big, dims = 3), dims = 3)

dict_name = Dict([0 => "None", 1 => "Genotype", 2 => "Phenotype", 3 => "Both"])
A_plot_string = [dict_name[a] for a in A_plot]


function make_box_corrs(corr_value,P)
    first_node_block = Tuple(findfirst(
        x -> x == corr_value, P[sorted_labels, sorted_labels, 3]))
    last_node_block = Tuple(findlast(
        x -> x == corr_value, P[sorted_labels, sorted_labels, 3]))
    return BBox(
        first_node_block[1], last_node_block[1], first_node_block[2], last_node_block[2])
end

##

with_theme(theme_latexfonts()) do
    fig = Figure(size = (930, 360), fontsize = 16)
    colormap = :lipari
    colormap_corr = :balance
    corrs_values = sort([c for c in correlation_sorted if !isnan(c)], rev = true)
    max_abs_corr = max(0.5,maximum(abs.(corrs_values)))
    corr_range = (-max_abs_corr, max_abs_corr)
    ax = Axis(fig[1, 1], aspect = 1, title = "Observed network")
    ax1 = Axis(fig[1, 2], aspect = 1, title = "Fitted genotype layer")
    ax2 = Axis(fig[1, 3], aspect = 1, title = "Fitted phenotype layer")
    ax3 = Axis(fig[1, 4], aspect = 1, title = "Fitted correlation")
    hidedecorations!.([ax, ax1, ax2, ax3])

    heatmap!(ax, A_plot[sorted_labels, sorted_labels],
        colormap = Makie.Categorical(Reverse(:okabe_ito)))
    heatmap!(
        ax1, P[sorted_labels, sorted_labels, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(
        ax2, P[sorted_labels, sorted_labels, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax3, P[sorted_labels, sorted_labels, 3],
        colormap = colormap_corr, colorrange = corr_range)

    for c in corrs_values[1:1]
        box_corrs = make_box_corrs(c, P)
        wireframe!(ax, box_corrs, color = :red)
    end
    cb = Colorbar(fig[2, 1];
        colormap = Reverse(cgrad(:okabe_ito, 4, categorical = true)),
        limits = (0, 4),
        label = "Type of connection",
        ticklabelsize = 12,
        vertical = false, width = Relative(1.0),
        flipaxis = false, ticks = (
            [0.5, 1.4, 2.6, 3.5], ["None", "Genotype", "Phenotype", "Both"]))
    Colorbar(fig[2, 2:3], colorrange = (0, 1),
        colormap = colormap, vertical = false, flipaxis = false, width = Relative(0.7), label = "Probability")
    Colorbar(fig[2, 4], colorrange = corr_range, label = "Correlation",
        colormap = colormap_corr, vertical = false, width = Relative(1.0), flipaxis = false)
    #colgap!(fig.layout, 0)
    display(fig)
    save(joinpath(@__DIR__, "diseasome_all_in_one.pdf"), fig)
    save(joinpath(@__DIR__, "diseasome_all_in_one.png"), fig, px_per_unit = 2)
end

## Additional figures

with_theme(theme_latexfonts()) do
    fig = Figure(size = (850, 400))
    colormap = :okabe_ito
    ax = Axis(fig[1, 1], aspect = 1, title = "Sorted by histogram",)
    ax2 = Axis(fig[1, 2], aspect = 1, title = "Sorted by disease category",)
    ax3 = Axis(fig[1, 3], aspect = 1, title = "Sorted by degree",)

    heatmap!(ax, A_plot[sorted_labels, sorted_labels],
        colormap = Makie.Categorical(Reverse(colormap)))
    heatmap!(ax2, A_plot[sorting_by_category, sorting_by_category],
        colormap = Makie.Categorical(Reverse(colormap)))
    pl = heatmap!(ax3, A_plot[sorting_by_degree, sorting_by_degree],
        colormap = Makie.Categorical(Reverse(colormap)))
    hidedecorations!.([ax, ax2, ax3], label = false)

    cb = Colorbar(fig[2, :];
        colormap = Reverse(cgrad(colormap, 4, categorical = true)),
        limits = (0, 4),
        label = "Type of connection",
        vertical = false, width = Relative(0.5), flipaxis = false,
        ticks = ([0.5, 1.5, 2.5, 3.5], ["None", "Genotype", "Phenotype", "Both"]))
    display(fig)

    save(joinpath(@__DIR__, "diseasom_multiple_ordering.pdf"), fig)
    save(joinpath(@__DIR__, "diseasom_multiple_ordering.png"), fig, px_per_unit = 2)
end

## get table of diseases
using Latexify
k = length(unique(estimated.node_labels))

df = DataFrame(Community = 1:k,Diseases = [join(diseases_names[findall(estimated.node_labels .== i)], ", ") for i in 1:k])
latexify(df; env = :table, latex = false)

include(joinpath(path_to_data_folder,"communities.jl"))
##

corrs_values = sort([c for c in correlation_sorted if !isnan(c)],
    rev = true)
for c in corrs_values[1:3]
    println("correlation value: ", c)

    indices_group = (findall(x -> x == c, correlation_sorted))
    indices_node_group = [x[1] for x in indices_group]
    indices_group = filter(x -> Tuple(x)[1] <= Tuple(x)[2], indices_group)
    diseases_highest_correlation = vcat(split.(filter(x -> x.Community ∈ indices_node_group, df).Diseases, ", ")...)
    println(length(diseases_highest_correlation))
    println(count_matches.(Ref(diseases_highest_correlation), communities_original_paper))
end
##
nodes_with_marginal_1 = (findall(x -> x[2] == 1, marginals_sorted))
indices_node_group_marginal_1 = unique([x[1] for x in nodes_with_marginal_1])
diseases_marginal_1 = vcat(split.(
    filter(x -> x.Community ∈ indices_node_group_marginal_1, df).Diseases, ", ")...)


count_matches.(Ref(diseases_marginal_1), communities_original_paper)
