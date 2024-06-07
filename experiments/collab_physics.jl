using NetworkHistogram
using CSV, DataFrames
using Dates
using CairoMakie
using MVBernoulli
using StatsBase

include("utils.jl")

## Load data
nodes_data = CSV.read("data/collab_physics/pierreAuger.csv/nodes.csv",
    DataFrame, normalizenames = true, drop = ["_pos"])
edges = CSV.read("data/collab_physics/pierreAuger.csv/edges.csv",
    DataFrame, normalizenames = true)
edges._source = edges._source .+ 1
edges.target = edges.target .+ 1

n_nodes = size(nodes_data, 1)
L = length(unique(edges.layer))
A = zeros(Bool, n_nodes, n_nodes, L)
for row in eachrow(edges)
    A[row._source, row.target, row.layer] = 1
    A[row.target, row._source, row.layer] = 1
end

for i in 1:L
    display(heatmap(A[:, :, i], colormap = :binary))
end

A_sum = dropdims(sum(A, dims = 3), dims = 3)
countmap(A_sum)
A_sum[A_sum .≥ 2] .= 2

A_multiplex = zeros(Int, n_nodes, n_nodes, 2)
for j in 1:size(A_sum, 1)
    for i in 1:size(A_sum, 2)
       if A_sum[j, i] == 1
            A_multiplex[j, i, 1] = 1
        elseif A_sum[j, i] == 2
            A_multiplex[j, i, 2] = 1
        end
    end
end


fig = Figure()
axes = [Axis(fig[1, 1], aspect=1, xlabel = "Node", ylabel = "Node"),
    Axis(fig[1, 2], aspect = 1, xlabel = "Node", ylabel = "Node")]

for i in 1:2
    heatmap!(axes[i], A_multiplex[:, :,i], colormap = :binary)
end

display(fig)
