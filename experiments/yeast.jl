using NetworkHistogram
using Plots

include("utils.jl")

data_dir = joinpath(@__DIR__, "../data/YeastLandscape_Multiplex_Genetic/Dataset/")
path_to_data = joinpath(data_dir, "yeast_landscape_multiplex.edges")
path_to_layer_names = joinpath(data_dir, "yeast_landscape_layers.txt")


# Load the data
n = 4458
num_layers = 4
A = Array{Int}(undef, n, n, num_layers)

for i in 1:n
    for j in 1:n
        A[i, j, :] .= zeros(Int, num_layers)
    end
end

for line in readlines(path_to_data)
    layer, i, j, weight = split(line, ' ')
    i = parse(Int, i)
    j = parse(Int, j)
    layer = parse(Int, layer)
    weight = parse(Float64, weight)
    if i != j && abs(weight) > 0.1
        A[i, j, layer] = 1
        A[j, i, layer] = 1
    end
end

list_names = Vector{String}(undef, num_layers)
for (i, line) in enumerate(readlines(path_to_layer_names))
    if i > 1
        num, name = split(line, ' ')
        list_names[i - 1] = name
    end
end

for i in 1:num_layers
    println("Layer $i: $(list_names[i]), with density $(sum(A[:, :, i])/(n*(n-1)/2)).")
end


indices_studied = collect(1:num_layers)
A_to_study = A[:, :, indices_studied]
println(list_names[indices_studied])

estimated, history = graphhist(
    A_to_study; starting_assignment_rule = NetworkHistogram.RandomStart(), maxitr = Int(1e6),
    stop_rule = PreviousBestValue(500))

display(plot(history.history))
moments, indices = NetworkHistogram.get_moment_representation(estimated)

p_moms = []
plots_pij = []
for (i, index) in enumerate(indices)
    push!(p_moms,
        heatmap(moments[:, :, i], xlabel = "Moment $index", xformatter = _ -> "",
            yformatter = _ -> "", aspect_ratio = :equal, axis = (
                [], false)))
    push!(plots_pij,
        heatmap(
            get_p_matrix(moments[:, :, i], estimated.node_labels), xformatter = _ -> "",
            yformatter = _ -> "", xlabel = "Moment $index", aspect_ratio = :equal, axis = (
                [], false)))
end

p_networks = []
for i in 1:size(A_to_study, 3)
    push!(p_networks,
        heatmap(A_to_study[:, :, i], clims = (0, 1), legend = :none,
            xlabel = "Network $i", xformatter = _ -> "",
            yformatter = _ -> ""))
end

indices_interesting_moments = []
for i in eachindex(indices)
    if maximum(moments[:, :, i]) > 0
        push!(indices_interesting_moments, i)
    end
end
println(length(indices_interesting_moments))

for i in indices_interesting_moments
    display(plot(p_moms[i]))
end
