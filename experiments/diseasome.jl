using NetworkHistogram, Statistics, CSV
using DataFrames
using ProgressMeter
using Makie, CairoMakie
using JLD
using MVBernoulli
using LinearAlgebra

Makie.inline!(true)
using Random
Random.seed!(123354472456192348634612304864326748)
include("utils.jl")
## Load the data

path_to_data_folder = joinpath(@__DIR__, "../data/", "MultiplexDiseasome-master")

gwas_dataset = CSV.read(joinpath(path_to_data_folder, "Datasets", "DSG_network_from_GWAS.csv"), DataFrame)
omim_dataset = CSV.read(joinpath(path_to_data_folder, "Datasets", "DSG_network_from_OMIM.csv"), DataFrame)

## GWAS dataset

function get_genotype_and_symptoms_cooccurrence_matrix(gwas_dataset::DataFrame, file::String = joinpath(path_to_data_folder,"adj_gwas.jld"))
    if isfile(file)
        loaded =  load(file)
        return loaded["node_names"], loaded["adj_genotype"], loaded["adj_symptoms"]
    end
    n = length(unique(gwas_dataset.disorder))
    diseases_names = sort(unique(gwas_dataset.disorder))

    adj_genotype = zeros(n, n)
    adj_symptoms = zeros(n, n)

    @showprogress for i in 1:n-1
        df_i = filter(x -> x.disorder == diseases_names[i], gwas_dataset)
        for j in i+1:n
            df_j = filter(x -> x.disorder == diseases_names[j], gwas_dataset)
            adj_genotype[i, j] = count_matches(df_i.gene_symb, df_j.gene_symb)
            adj_symptoms[i, j] = count_matches(df_i.symptom, df_j.symptom)
            adj_genotype[j, i] = adj_genotype[i, j]
            adj_symptoms[j, i] = adj_symptoms[i, j]
        end
    end

    save(file, "node_names" ,diseases_names, "adj_genotype" ,adj_genotype, "adj_symptoms" ,adj_symptoms)
    return diseases_names, adj_genotype, adj_symptoms
end

diseases_names, adj_genotype, adj_symptoms = get_genotype_and_symptoms_cooccurrence_matrix(omim_dataset, joinpath(path_to_data_folder,"adj_omim.jld"))




A_all = zeros(length(diseases_names), length(diseases_names), 2)
A_all[:, :, 1] =  adj_genotype
A_all[:, :, 2] = adj_symptoms



## Remove isolated diseases
A_all_binary = Int.(A_all .> 0)
degrees = dropdims(sum(A_all_binary, dims = (2)), dims = 2)

threshold = 2
#result with threshold = 2 and non_isolated_layer = findall(x -> x[1] ≥ threshold && x[2] ≥ threshold, eachrow(degrees))
non_isolated_layer = findall(x -> x[1] ≥ threshold && x[2] ≥ threshold, eachrow(degrees))

A_inter = A_all_binary[non_isolated_layer, non_isolated_layer, :]
A_weight_inter = A_all[non_isolated_layer, non_isolated_layer, :]
names = diseases_names[non_isolated_layer]
non_isolated_diseases = findall(x -> x ≥ threshold,
    vec(sum(A_inter, dims = (2, 3))))


A_weight = A_weight_inter[non_isolated_diseases, non_isolated_diseases, :]
names = names[non_isolated_diseases]
n = length(names)


#slow but can't be bothered to optimize
categories = Array{String}(undef, n)
for (i,name) in enumerate(names)
    category_raw = filter(x -> x.disorder == name, omim_dataset).disorder_cat
    if isempty(category_raw) || ismissing(category_raw[1])
        categories[i] = "z-Unknown"
    else
        categories[i] = category_raw[1]
    end
end

A = Int.(A_weight .> 0)



categories_degree = [mean(A[findall(categories .== cat), :, :]) for cat in unique(categories)]
categories_order = unique(categories)[sortperm(categories_degree, rev = true)]

degrees = vec(sum(A, dims = (2,3)))
tuple_cat_degree = [(findfirst(s -> s == categories[i], categories_order), degrees[i])
                    for i in 1:size(A, 1)]

sorting_by_category = sortperm(tuple_cat_degree,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] > y[2]))

## Convert to binary

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype, threshold = $threshold")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms, threshold = $threshold")
heatmap!(ax, A[:, :, 1], colormap = :binary)
heatmap!(ax2, A[:, :, 2], colormap = :binary)
display(fig)



## Fit the model
estimator, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e7),
    stop_rule = PreviousBestValue(10000))

fig = Figure()
best_ll = round(NetworkHistogram.get_bestitr(history)[2], sigdigits=2)
ax = Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Log-likelihood", title = "Log-likelihood: $(best_ll)")
lines!(ax, get(history.history, :best_likelihood)...)
display(fig)


##
n_group_nodes = length(unique(estimator.node_labels))
max_shapes = n_group_nodes * (n_group_nodes + 1) ÷ 2
estimator_ = NetworkHistogram.GraphShapeHist(max_shapes, estimator)
for i in 1:n_group_nodes
    for j in i:n_group_nodes
        @assert all(estimator.θ[i,j,:] .== estimator_.θ[i,j])
    end
end


best_smoothed, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)
display(lines(bic_values, legend = false, xlabel = "Number of shapes", ylabel = "BIC", title = "BIC values"))
##


estimated = best_smoothed
moments, indices = NetworkHistogram.get_moment_representation(estimated)

mvberns = MVBernoulli.from_tabulation.(estimated.θ)
marginals = MVBernoulli.marginals.(mvberns)
corrs = MVBernoulli.correlation_matrix.(mvberns)

mvberns_block = MVBernoulli.from_tabulation.(estimator_.θ)
marginals_block = MVBernoulli.marginals.(mvberns_block)
corrs_block = MVBernoulli.correlation_matrix.(mvberns_block)

@assert length(estimated.node_labels) == n
P_block = zeros(n, n, 3)
P_block[:, :, 1] = get_p_matrix([m[1] for m in marginals_block], estimator_.node_labels)
P_block[:, :, 2] = get_p_matrix([m[2] for m in marginals_block], estimator_.node_labels)
P_block[:, :, 3] = get_p_matrix([m[3] for m in corrs_block], estimator_.node_labels)

P = zeros(n, n, 3)
P[:, :, 1] = get_p_matrix([m[1] for m in marginals], estimated.node_labels)
P[:, :, 2] = get_p_matrix([m[2] for m in marginals], estimated.node_labels)
P[:, :, 3] = get_p_matrix([m[3] for m in corrs], estimated.node_labels)


#P[:,:,1:2] .= P[:,:,1:2] .^0.5
#P_block[:,:,1:2] .= P_block[:,:,1:2] .^0.5

function display_approx_and_data(P, A, sorting; label = "")
    fig = Figure(size = (800, 500))
    colormap = :lipari
    ax = Axis(fig[1, 1], aspect = 1, title = "Genotype layer", ylabel = "Histogram")
    ax2 = Axis(fig[1, 2], aspect = 1, title = "Phenotype layer")
    ax3 = Axis(fig[1, 3], aspect = 1, title = "Correlation")
    ylabel = label == "" ? "Adjacency matrix" : "Adjacency matrix (sorted by $label)"
    ax4 = Axis(fig[2, 1], aspect = 1, ylabel = ylabel)
    ax5 = Axis(fig[2, 2], aspect = 1)
    ax6 = Axis(fig[2, 3], aspect = 1)
    heatmap!(ax, P[sorting,sorting, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax2, P[sorting,sorting, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax3, P[sorting,sorting, 3], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax4, A[sorting, sorting, 1], colormap = :binary)
    heatmap!(ax5, A[sorting, sorting, 2], colormap = :binary)
    heatmap!(
        ax6, A[sorting, sorting, 1] .* A[sorting, sorting, 2],
        colormap = :binary)
    Colorbar(fig[1:2, end + 1], colorrange = (0, 1),
        colormap = colormap, vertical = true, height = Relative(0.8))
    #Colorbar(fig[1:2, end + 1], colorrange = (-1, 1), label = "Correlation",
    #    colormap = :balance, vertical = true)
    hidedecorations!.([ax2, ax3, ax5, ax6])
    hidedecorations!.([ax, ax4], label=false)
    return fig
end


sorted_degree = sortperm(vec(sum(A, dims = (2, 3))), rev = true)
sorted_labels = sortperm(estimated.node_labels, rev = false)



fig_fit = display_approx_and_data(P, A, sorted_labels, label = "")
save(joinpath(@__DIR__, "diseasome_fit.pdf"), fig_fit)
display(fig_fit)

##
A_plot_big = deepcopy(A)
A_plot_big[:, :, 1] .*= 1
A_plot_big[:, :, 2] .*= 2
A_plot = dropdims(sum(A_plot_big, dims = 3), dims = 3)

dict_name = Dict([0 => "None", 1 => "Genotype", 2 => "Phenotype", 3 => "Both"])
A_plot_string = [dict_name[a] for a in A_plot]

fig = Figure(size = (800, 400))
#titlelayout = GridLayout(fig[0, 2], halign = :center, tellwidth = false)
#Label(titlelayout[1,:], "Flattened multiplex adjacency matrix", halign = :center,
#    fontsize = 20)
#rowgap!(titlelayout, 0)

ax = Axis(fig[1, 1], aspect = 1, title = "Sorted by histogram clusters", titlesize = 14)
ax2 = Axis(fig[1, 2], aspect = 1, title = "Sorted by disease category", titlesize = 14)
ax3 = Axis(fig[1, 3], aspect = 1, title = "Sorted by degree", titlesize = 14)

heatmap!(ax, A_plot[sorted_labels, sorted_labels],
    colormap = Makie.Categorical(Reverse(:okabe_ito)))
heatmap!(ax2, A_plot[sorting_by_category, sorting_by_category],
    colormap = Makie.Categorical(Reverse(:okabe_ito)))
pl = heatmap!(ax3, A_plot[sorted_degree, sorted_degree],
    colormap = Makie.Categorical(Reverse(:okabe_ito)))
hidedecorations!.([ax, ax2, ax3], label = false)
#cb = Colorbar(fig[2, :],pl;
#    label = "Type of connection", vertical = false, width = Relative(0.5), flipaxis=false)
cb = Colorbar(fig[2, :];
    colormap = Reverse(cgrad(:okabe_ito, 4, categorical = true)),
    limits = (0,4),
    label = "Type of connection",
    labelsize = 14,
    ticklabelsize = 12,
    vertical = false, width = Relative(0.5), flipaxis=true, ticks = ([0.5, 1.5, 2.5, 3.5], ["None", "Genotype", "Phenotype", "Both"]))
display(fig)
save(joinpath(@__DIR__, "diseasome_adjacency.pdf"), fig)

##


display(display_approx_and_data(P, A, sorted_labels, label = "fit"))
display(display_approx_and_data(P, A, 1:n; label= "index"))
display(display_approx_and_data(P, A, sorting_by_category, label ="category"))
display(display_approx_and_data(P, A, sorted_degree, label = "degree"))
display(display_approx_and_data(P_block, A, 1:n; label = "index, block model"))
display(display_approx_and_data(P_block, A, sorting_by_category, label = "category, block model"))
display(display_approx_and_data(P_block, A, sorted_degree, label = "degree, block model"))
display(display_approx_and_data(P_block, A, sorted_labels, label = "fit, block model"))

## find interesting correlations

indices_group = (findall(x -> x[3] > 0.4, corrs))
indices_node_group = [x[1] for x in indices_group]
indices_group = filter(x -> Tuple(x)[1] <= Tuple(x)[2], indices_group)

fitted_dists = unique(mvberns[indices_group])

indices_node_group
nodes = [findall(estimated.node_labels .== i) for i in indices_node_group]
names_corr = [names[i] for i in nodes]
