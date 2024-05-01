using NetworkHistogram, Statistics, CSV
using DataFrames
using ProgressMeter
using Makie, CairoMakie
using JLD

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
A_all[:, :, 1] = adj_genotype
A_all[:, :, 2] = adj_symptoms


## Remove isolated diseases
degrees = vec(sum(A_all, dims = (2, 3)))
threshold = 1
non_isolated_diseases = findall(x -> x ≥ threshold , degrees)

A_weight = A_all[non_isolated_diseases, non_isolated_diseases, :]
names = diseases_names[non_isolated_diseases]
n = length(names)


#slow but can't be bothered to optimize
categories = Array{String}(undef, n)
for (i,name) in enumerate(names)
    category_raw = filter(x -> x.disorder == name, gwas_dataset).disorder_cat
    if isempty(category_raw) || ismissing(category_raw[1])
        categories[i] = "z-Unknown"
    else
        categories[i] = category_raw[1]
    end
end

sorting_by_category = sortperm(categories)
A_weight = A_weight[sorting_by_category, sorting_by_category, :]
names = names[sorting_by_category]

## Convert to binary
A = Int.(A_weight .> 0)
fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms")
heatmap!(ax, A[:, :, 1], colormap = :binary)
heatmap!(ax2, A[:, :, 2], colormap = :binary)
display(fig)



## Fit the model
estimator, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e7),
    stop_rule = PreviousBestValue(10000))

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Log-likelihood")
lines!(ax, get(history.history, :best_likelihood)...)
display(fig)


##
best_smoothed, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)

estimated = best_smoothed
moments, indices = NetworkHistogram.get_moment_representation(estimated)


##

P = zeros(n, n, 3)
for i in 1:3
    P[:, :, i] = get_p_matrix(moments[:, :, i], estimated.node_labels)
end


function display_approx_and_data(P, A, sorting; label = "p")
    fig = Figure(size = (800, 400))
    colormap = :lipari
    ax = Axis(fig[1, 1], aspect = 1, title = "Genotype")
    ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms")
    ax3 = Axis(fig[1, 3], aspect = 1, title = "Interactions")
    ax4 = Axis(fig[2, 1], aspect = 1)
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
    Colorbar(fig[1:2, end + 1], colorrange = (0, 1), label = label,
        colormap = colormap, vertical = true)
    hidedecorations!.([ax, ax2, ax3, ax4, ax5, ax6])
    return fig
end

sorted_degree = sortperm(vec(sum(A, dims = (2, 3))), rev = true)
sorted_labels = sortperm(estimated.node_labels, rev = false)


display(display_approx_and_data(moments, moments, 1:size(moments,1); label= "moments"))
display(display_approx_and_data(P, A, 1:n; label= "ordered by index"))
display(display_approx_and_data(P, A, sorting_by_category, label ="ordered by category"))
display(display_approx_and_data(P, A, sorted_labels, label = "ordered by fit"))
display(display_approx_and_data(P, A, sorted_degree, label = "ordered by degree"))

fig = Figure(size=(800,400))
colormap = :lipari
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms")
ax3 = Axis(fig[1, 3], aspect = 1, title = "Interactions")
heatmap!(ax, P[:, :, 1], colormap = colormap, colorrange = (0, 1))
heatmap!(ax2, P[:, :, 2], colormap = colormap, colorrange = (0, 1))
heatmap!(ax3, P[:, :, 3], colormap = colormap, colorrange = (0, 1))
Colorbar(fig[end + 1,:], colorheatmap!(ax3, P[:, :, 3], colormap = colormap, colorrange = (0, 1))
range = (0, 1), label = "ordered by category", colormap = colormap, vertical = false)
hidedecorations!.([ax, ax2, ax3])
display(fig)

P_sorted_degree = P[sorted_degree, sorted_degree, :]


fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms")
heatmap!(ax, A[sorted_degree, sorted_degree, 1], colormap = :binary)
heatmap!(ax2, A[sorted_degree, sorted_degree, 2], colormap = :binary)
display(fig)

fig = Figure(size = (800, 400))
colormap = :lipari
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms")
ax3 = Axis(fig[1, 3], aspect = 1, title = "Interactions")
heatmap!(ax, P_sorted_degree[:, :, 1], colormap = colormap, colorrange = (0, 1))
heatmap!(ax2, P_sorted_degree[:, :, 2], colormap = colormap, colorrange = (0, 1))
heatmap!(ax3, P_sorted_degree[:, :, 3], colormap = colormap, colorrange = (0, 1))
Colorbar(fig[end + 1, :], colorrange = (0, 1), label = "ordered by degree",
    colormap = colormap, vertical = false)
hidedecorations!.([ax, ax2, ax3])
display(fig)

P_sorted_labels = P[sorted_labels, sorted_labels, :]

##
