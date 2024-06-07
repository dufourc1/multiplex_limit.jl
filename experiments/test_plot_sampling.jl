using CairoMakie
using Graphs
using LinearAlgebra
using GraphMakie
using NetworkHistogram
using Distributions



# Define the Graphon function, example
W(u, v) =  u*v

# Creating a heat map of the Graphon over a grid
u = v = LinRange(0, 1, 100)
W_values = [W(ui, vi) for ui in u, vi in v]


u = v = LinRange(0, 1, 30)
A = [W(ui, vi) > rand() for ui in u, vi in v]
A = triu(A) + triu(A, 1)'
A = A .- diagm(diag(A))

A = NetworkHistogram.drop_disconnected_components(A)

# Sampled matrix, for example
# Example adjacency matrix for the graph, assuming symmetric and binary
G = A
# Generate the plots
fig = Figure(size=(900, 300))

ax1 = Axis(fig[1, 1],aspect = 1)
heatmap!(ax1, u, v, W_values, colormap = :binary, colorrange = (0,1))

# Adding an arrow and additional axis on the right of ax1
#arrows!(ax1, [0.9], [0.5], [1.1], [0.5], arrow_size=15, color=:red)
#ax_extra = Axis(fig[1, 1], ylabel="W(U₁, U₂)", xgridvisible=false, ygridvisible=false)

ax2 = Axis(fig[1, 2], aspect = 1)
heatmap!(ax2, u,v, A, colormap=:binary)
hidedecorations!(ax2)

# Adding an arrow between ax1 and ax2
#arrows!(fig, [0.33, 0.4], [0.5, 0.5], [0.66, 0.6], [0.5, 0.5], arrow_size=15, color=:red)

ax3 = Axis(fig[1, 3])
graph = Graphs.SimpleGraphs.SimpleGraph(G)

graphplot!(ax3, graph, color = :black, markersize = 0.1, edge_width=0.3)
hidespines!(ax3)
hidedecorations!(ax3)
ax3.aspect = DataAspect()



display(fig)

##

colors = [:white, :red, :black]


W_decorated(u,v) = [1-u*v, u*v/2, u*v/2]
function get_color_sampled(u, v)
    return rand(Categorical(W_decorated(u, v)))
end

A = Matrix{Int}(undef, 30, 30)
for i in 1:30, j in i:30
    if i == j
        A[i,j] = 0
        continue
    end
    A[i,j] = get_color_sampled(u[i], v[j])-1
    A[j,i] = A[i,j]
end

A = NetworkHistogram.drop_disconnected_components(A)

graph = Graphs.SimpleGraphs.SimpleGraph(A)
edges_colors = [colors[A[e.src, e.dst]+1] for e in edges(graph)]

W_values = [[W_decorated(ui, vi)[k] for ui in u, vi in v] for k in 1:3]


fig = Figure(size = (900, 300))

ax1 = Axis(fig[1, 1], aspect = 1)
heatmap!(ax1, u, v, W_values[3], colormap = :binary, colorrange = (0, 1))

# Adding an arrow and additional axis on the right of ax1
#arrows!(ax1, [0.9], [0.5], [1.1], [0.5], arrow_size=15, color=:red)
#ax_extra = Axis(fig[1, 1], ylabel="W(U₁, U₂)", xgridvisible=false, ygridvisible=false)

ax2 = Axis(fig[1, 2], aspect = 1)
heatmap!(ax2, u, v, A, colormap = colors)
hidedecorations!(ax2)

# Adding an arrow between ax1 and ax2
#arrows!(fig, [0.33, 0.4], [0.5, 0.5], [0.66, 0.6], [0.5, 0.5], arrow_size=15, color=:red)

ax3 = Axis(fig[1, 3])
graph = Graphs.SimpleGraphs.SimpleGraph(A)

graphplot!(ax3, graph, markersize = 0.1, edge_width = 0.3, edge_color = edges_colors)
hidespines!(ax3)
hidedecorations!(ax3)
ax3.aspect = DataAspect()

display(fig)
