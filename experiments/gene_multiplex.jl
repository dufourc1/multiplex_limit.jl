using NetworkHistogram
using CSV, DataFrames
using Dates
using CairoMakie
using MVBernoulli
using StatsBase

include("utils.jl")

## Load data
nodes_data = CSV.read("data/genetic_multiplex/Celegans.csv/nodes.csv",
    DataFrame, normalizenames = true, drop = ["_pos"])
edges = CSV.read("data/genetic_multiplex/Celegans.csv/edges.csv",
    DataFrame, normalizenames = true)
edges._source = edges._source .+ 1
edges.target = edges.target .+ 1


n_nodes = 3879
L = length(unique(edges.layer))
A = zeros(Bool,n_nodes, n_nodes, L)
for row in eachrow(edges)
    A[row._source, row.target, row.layer] = 1
    A[row.target, row._source, row.layer] = 1
end

A_agg = sum(A, dims = 3)[:,:,1]


findall(sum(A_agg,dims=1) .≤ 1)

heatmap(sum(A, dims = 3)[:,:,1], colormap = :binary)

for i in 1:L
    display(heatmap(A[:,:,i], colormap = :binary))
end

sum(A)
