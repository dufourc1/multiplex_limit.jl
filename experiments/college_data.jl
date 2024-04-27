using NetworkHistogram
using Plots

include("utils.jl")


data_dir = joinpath(@__DIR__, "../data/CS-Aarhus_Multiplex_Social/Dataset/")
path_to_data = joinpath(data_dir, "CS-Aarhus_multiplex.edges")
path_to_layer_names = joinpath(data_dir, "CS-Aarhus_layers.txt")

# Load the data
n = 61
num_layers = 5
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
    if i!=j
        A[i, j, layer] = 1
        A[j, i, layer] = 1
    end
end


list_names = Vector{String}(undef, num_layers)
for (i,line) in enumerate(readlines(path_to_layer_names))
    if i > 1
        num, name = split(line, ' ')
        list_names[i-1] = name
    end
end



indices_studied = collect(1:num_layers)
A_to_study = A[:, :, indices_studied]
println(list_names[indices_studied])

estimated, history = graphhist(
    A_to_study; starting_assignment_rule = NetworkHistogram.RandomStart(), maxitr = Int(1e6),
    stop_rule = PreviousBestValue(500), h=1/4)

display(plot(history.history))
moments, indices = NetworkHistogram.get_moment_representation(estimated)

permutation = sortperm(estimated.node_labels)

white_lines = []
permuted_node_labels = estimated.node_labels[permutation]
for (index,label) in enumerate(permuted_node_labels[2:end])
    if label != permuted_node_labels[index]
        push!(white_lines, index)
    end
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
            get_p_matrix(moments[:, :, i], estimated.node_labels), clims = (
                0, 1), xformatter = _ -> "",
            yformatter = _ -> "", xlabel = "Moment $index", aspect_ratio = :equal, axis = (
                [], false)))
end

p_networks = []
for i in 1:size(A_to_study, 3)
    p = heatmap(
        A_to_study[:, :, i][permutation, permutation], clims = (0, 1), legend = :none,
        xlabel = "Network $i : $(list_names[i])", xformatter = _ -> "",
        yformatter = _ -> "")
    for line in white_lines
        plot!(p, [line, line], [0, n], color = :white)
        plot!(p, [0, n], [line, line], color = :white)
    end
    plot!(p, xlims=(0.5, n+0.5), ylims=(0.5, n+0.5))
    push!(p_networks,p)
end

display(plot(p_networks...))


indices_interesting_moments = []
for i in eachindex(indices)
    if maximum(moments[:, :, i]) > 0.2
        push!(indices_interesting_moments, i)
    end
end
println(length(indices_interesting_moments))


for i in indices_interesting_moments
    display(plot(plots_pij[i]))
end
