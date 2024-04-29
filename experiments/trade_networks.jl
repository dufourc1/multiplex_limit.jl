using NetworkHistogram
using Plots

include("utils.jl")

data_dir = joinpath(@__DIR__, "../data/FAO_Multiplex_Trade/Dataset/")
path_to_data = joinpath(data_dir, "fao_trade_multiplex.edges")
path_to_layer_names = joinpath(data_dir, "fao_trade_layers.txt")
path_to_node_names = joinpath(data_dir, "fao_trade_nodes.txt")

# Load the data
n_all = 214
max_edges_per_layer = n_all * (n_all - 1) ÷ 2
num_layers = 364
A_all = Array{Int}(undef, n_all, n_all, num_layers)
A_weights_all = Array{Float64}(undef, n_all, n_all, num_layers)
for i in 1:n_all
    for j in 1:n_all
        A_all[i, j, :] .= zeros(Int, num_layers)
    end
end


list_names_all = Vector{String}(undef, num_layers)
for (i, line) in enumerate(readlines(path_to_layer_names))
    if i > 1
        num, name = split(line, ' ')
        list_names_all[i - 1] = name
    end
end

node_names_all = Vector{String}(undef, n_all)
for (i, line) in enumerate(readlines(path_to_node_names))
    if i == 1
        continue
    end
    num, name = split(line, ' ')
    name = replace(name, "_" => " ")
    node_names_all[i-1] = name
end

for line in readlines(path_to_data)
    layer, i, j, weight = split(line, ' ')
    i = parse(Int, i)
    j = parse(Int, j)
    layer = parse(Int, layer)
    weight = parse(Float64, weight)
    if i != j && weight > 0
        A_all[i, j, layer] = 1
        A_all[j, i, layer] = 1
        A_weights_all[i, j, layer] = weight
        A_weights_all[j, i, layer] = weight
    end
end


# preprocessing as in "Latent space models for multiplex networks"
layers_dense = findall(vec(sum(A_all, dims = (1, 2))) ./ (max_edges_per_layer * 2) .≥ 0.1)
nodes_dense = findall(vec(sum(A_all[:,:, layers_dense], dims = (2,3))) .≥5*length(layers_dense))
nodes_dense = filter(x -> x!= 93, nodes_dense) # remove the node 93: unspecified country


A = A_all[nodes_dense, nodes_dense, layers_dense]
A_weights = A_weights_all[nodes_dense, nodes_dense, layers_dense]
list_names = list_names_all[layers_dense]
node_names = node_names_all[nodes_dense]
n = size(A, 1)

# fit the model
estimated, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e7),
    stop_rule = PreviousBestValue(1000))
#display(plot(history.history))

moments, indices = NetworkHistogram.get_moment_representation(estimated)




permutation = sortperm(estimated.node_labels)

white_lines = []
permuted_node_labels = estimated.node_labels[permutation]
for (index, label) in enumerate(permuted_node_labels[2:end])
    if label != permuted_node_labels[index]
        push!(white_lines, index)
    end
end

#get parameters matrices permuted
P = zeros(n,n, size(moments,3))
for i in 1:size(moments,3)
    P[:,:,i] = get_p_matrix(moments[:,:,i], estimated.node_labels)
end
A_permuted = A[permutation, permutation, :]
P_permuted = P[permutation, permutation, :]


function plot_pairs_adjacency_probs(A,P)
    p_networks = []
    p_probs = []
    for i in 1:size(A, 3)
        plot_graph = heatmap(
            A[:, :, i], clims = (0, 1), legend = :none,
            xformatter = _ -> "",
            yformatter = _ -> "")
        plot_probs = heatmap(
            P[:, :, i], clims = (0, 1), legend = :none,
            xformatter = _ -> "",
            yformatter = _ -> "")
        push!(p_networks, plot_graph)
        push!(p_probs, plot_probs)
        #display(plot(plot_graph, plot_probs, layout = (1, 2),
        #    size = (800, 400), bottom_margin = 4Plots.mm))
    end
    return p_networks, p_probs
end


#plot_networks, plot_probs = plot_pairs_adjacency_probs(A,P)
#plot_networks_permuted, plot_probs_permuted = plot_pairs_adjacency_probs(A_permuted,P_permuted)


#for i in 1:size(A, 3)
#    display(plot(plot_networks[i], plot_probs[i], plot_networks_permuted[i],
#                plot_probs_permuted[i], layout = (2, 2),
#            size = (800, 800), bottom_margin = 4Plots.mm, suptitle = list_names[i]))
#end


n_groups = length(unique(estimated.node_labels))
for group in sort(unique(estimated.node_labels))
    println("Group $group")
    println(node_names[estimated.node_labels .== group])
end

using DataFrames,Countries
df_all_countries = DataFrame(all_countries())
labels_not_found = Dict([
    "China, Hong Kong SAR" => "Hong Kong",
    "China, mainland" => "China",
    "China, Taiwan Province of" => "Taiwan, Province of China",
    "Iran (Islamic Republic of)" => "Iran, Islamic Republic of",
    "Republic of Korea" => "Korea, Republic of",
    "United States of America" => "United States",
    "CÃ´te d'Ivoire" => "Côte d'Ivoire",
    "Czech Republic" => "Czechia",
    "Republic of Moldova" => "Moldova, Republic of",
    "The former Yugoslav Republic of Macedonia" => "North Macedonia",
    "Venezuela (Bolivarian Republic of)" => "Venezuela, Bolivarian Republic of",
    "Bolivia (Plurinational State of)" => "Bolivia, Plurinational State of",
    "United Republic of Tanzania" => "Tanzania, United Republic of",
    "China, Macao SAR" => "Macao",
    "Turkey" => "Türkiye",
    ]
)

for name in node_names
    if name ∉ df_all_countries.name
        println(name)
        println(labels_not_found[name])
    end
end

nodes_name_common = replace.(node_names, labels_not_found...)
@assert length(nodes_name_common[(!in).(nodes_name_common, Ref(df_all_countries.name))]) ==
        0


country_codes = [df_all_countries[df_all_countries.name .== name, :].alpha3[1] for name in nodes_name_common]
country_cluster = estimated.node_labels

##
# Plot countries based on clustering
using Makie, CairoMakie, GeoMakie
import Downloads
using GeoMakie.GeoJSON
using GeometryBasics
using GeoMakie.GeoInterface
worldCountries = GeoJSON.read(read(
    Downloads.download("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"),
    String))

lons = -180:180
lats = -90:90
field = [exp(cosd(l)) + 3(y / 90) for l in lons, y in lats]

fig = Figure(size = (1200, 800), fontsize = 22)

ax = GeoAxis(
    fig[1, 1];
    dest = "+proj=wintri",
    title = "World Countries",
    tellheight = true
)

hm1 = Makie.surface!(ax, lons, lats, field; shading = NoShading)
Makie.translate!(hm1, 0, 0, -10)

poly!(
    ax, worldCountries;
    color = :white,
    strokecolor = :black,
    strokewidth = 0.25
)

countries_to_plot = filter(x -> x.ISO_A3 ∈ country_codes, worldCountries.features)
colors =  ones(Int,length(countries_to_plot))
for i in eachindex(countries_to_plot)
    country_code = countries_to_plot[i].ISO_A3
    colors[i] = country_cluster[country_codes .== country_code][1]
end

hm2 = poly!(
    ax, GeoJSON.FeatureCollection(features=countries_to_plot);
    color = colors,
    colormap = :tab20c,
    strokecolor = :black,
    strokewidth = 0.25
)

cb = Colorbar(fig[1, 2]; colorrange = (1, n_groups),
    colormap = cgrad(:tab20c, n_groups, categorical = true),
    label = "Group", height = Relative(0.65))

fig
