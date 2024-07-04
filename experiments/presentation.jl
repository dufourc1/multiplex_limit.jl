using CairoMakie
using Graphs
using LinearAlgebra
using GraphMakie
using NetworkHistogram
using Distributions

using LaTeXStrings

run_all = true
using Random
Random.seed!(12345)
##

# Define the Graphon function, example
W(u, v) =  u*v

# Creating a heat map of the Graphon over a grid
u = v = LinRange(0, 1, 100)
W_values = [W(ui, vi) for ui in u, vi in v]


n = 30
u = v = LinRange(0, 1, n)
A = [W(ui, vi) > rand() for ui in u, vi in v]
A = triu(A) + triu(A, 1)'
A = A .- diagm(diag(A))

A = NetworkHistogram.drop_disconnected_components(A)

# Sampled matrix, for example
# Example adjacency matrix for the graph, assuming symmetric and binary
G = A
# Generate the plots


with_theme(theme_latexfonts()) do
    fig = Figure(size=(850, 250), fontsize = 16)

    ax1 = Axis(fig[1, 1],aspect = 1, xticks = [0,1], yticks = [0,1], title = L"f:[0,1]^2 \mapsto [0,1]")
    hm = heatmap!(ax1, u, v, W_values, colormap = :binary, colorrange = (0,1))

    # Adding an arrow and additional axis on the right of ax1
    #arrows!(ax1, [0.9], [0.5], [1.1], [0.5], arrow_size=15, color=:red)
    #ax_extra = Axis(fig[1, 1], ylabel="W(U₁, U₂)", xgridvisible=false, ygridvisible=false)

    ax2 = Axis(fig[1, 2], aspect = 1, title = "Adjacency matrix")
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
    save("experiments/simple_graphon_sampling.png", fig, px_per_unit = 3)
end

##

function W_decorated(u, v)
    scale = 4 + (1-u)*(1-v)
    proba_1 = (1-u)*(1-v)/scale
    proba_2 = u*v/scale
    proba_0 = 1 - proba_1 - proba_2
    p = [proba_0, proba_1, proba_2]
    return p ./sum(p)
end

function get_color_sampled(u, v)
    return rand(Categorical(W_decorated(u, v)))
end

n = 30
u,v = LinRange(0.001, 0.999, n), LinRange(0.001, 0.999, n)
A = Matrix{Int}(undef, n, n)
for i in 1:n, j in i:n
    if i == j
        A[i,j] = 0
        continue
    end
    A[i,j] = get_color_sampled(u[i], v[j])-1
    A[j,i] = A[i,j]
end

A = NetworkHistogram.drop_disconnected_components(A)
u = v = LinRange(0, 1, 100)
W_values = [[W_decorated(ui, vi)[k] for ui in u, vi in v] for k in 1:3]

##

colormaps = [:Oranges, :Blues]
colors = [:white, to_colormap(colormaps[1])[6], to_colormap(colormaps[2])[7]]

graph = Graphs.SimpleGraphs.SimpleGraph(A)



edges_colors = [colors[A[e.src, e.dst]+1] for e in edges(graph)]




with_theme(theme_latexfonts()) do
    fig = Figure(size = (850, 250))


    xs = u



    ax1 = Axis3(fig[1, 1], aspect = (1, 1, 1.3*1/3),
                elevation = π / 8, xticks = ([0, 1],["0", "1"]),
                yticks = ([0, 1], ["0", "1"]),
                zticks = ([0.1, 0.8], [L"w^{(1)}", L"w^{(2)}"]),
                xlabel = "",
                ylabel = "",
                #zticksvisible = false,
                xticksvisible = false,
                yticksvisible = false,
                zgridvisible = false,
                #xticklabelsvisible = false,
                #yticklabelsvisible = false,
                #zticklabelsvisible = false,
                zspinesvisible = false,
                #azimuth = 1.4π,
                title = L"W(x,y)"
                )


    scale = 5
    hm = heatmap!(ax1, xs, xs, W_values[2] .* scale , colorrange = (0, 1), colormap = colormaps[1])
    translate!(hm, 0, 0, 0.1)

    hm = heatmap!(ax1, xs, xs, W_values[3] .* scale, colorrange = (0, 1), colormap = colormaps[2])
    translate!(hm, 0, 0, 0.8)

    zlims!(ax1, (0,1))
    #xlims!(ax1, (0,1))
    #ylims!(ax1, (0,1))
    hidespines!(ax1)



    # Adding an arrow and additional axis on the right of ax1
    #arrows!(ax1, [0.9], [0.5], [1.1], [0.5], arrow_size=15, color=:red)
    #ax_extra = Axis(fig[1, 1], ylabel="W(U₁, U₂)", xgridvisible=false, ygridvisible=false)

    ax2 = Axis(fig[1, 2], aspect = 1, title = "Decorated adjacency matrix")
    heatmap!(ax2, u, v, A, colormap = colors)
    hidedecorations!(ax2)

    # Adding an arrow between ax1 and ax2
    #arrows!(fig, [0.33, 0.4], [0.5, 0.5], [0.66, 0.6], [0.5, 0.5], arrow_size=15, color=:red)

    ax3 = Axis(fig[1, 3])
    graph = Graphs.SimpleGraphs.SimpleGraph(A)

    graphplot!(ax3, graph, markersize = 1, edge_width = 1, edge_color = edges_colors)
    hidespines!(ax3)
    hidedecorations!(ax3)
    ax3.aspect = DataAspect()
    resize_to_layout!(fig)
    display(fig)
    save("experiments/decorated_graphon_sampling.png", fig, px_per_unit = 3)
end

##
if run_all
    include("utils.jl")
    using MVBernoulli

    function get_tabulation(x, y)
        tabulation = zeros(4)
        tabulation[1] = 3 * x * y
        tabulation[2] = 3 * sin(2 * π * x) * sin(2 * π * y)
        tabulation[3] = exp(-3 * ((x - 0.5)^2 + (y - 0.5)^2))
        tabulation[4] = 2 - 3 * (x + y)
        #tabulation[1] *= 1.5
        softmax!(tabulation)
        return tabulation
    end

    function get_ground_truth_and_adj(n, get_theta = get_tabulation)
        w = zeros((n, n, 4))
        for i in 1:n
            for j in 1:n
                w[i, j, :] = get_theta(i / n, j / n)
            end
        end

        theta_matrix = [MVBernoulli.from_tabulation(w[i, j, :]) for i in 1:n, j in 1:n]
        correlations = MVBernoulli.correlation_matrix.(theta_matrix)
        marginals = MVBernoulli.marginals.(theta_matrix)

        P_true = zeros(n, n, 3)
        P_true[:, :, 1] = [x[1] for x in marginals]
        P_true[:, :, 2] = [x[2] for x in marginals]
        P_true[:, :, 3] = [x[3] for x in correlations]

        A = Array{Int}(undef, n, n, 2)
        for i in 1:n
            for j in i:n
                if i == j
                    A[i, j, :] .= 0
                else
                    A[i, j, :] .= rand(theta_matrix[i, j])
                    A[j, i, :] .= A[i, j, :]
                end
            end
        end
        return theta_matrix, P_true, A
    end



    n_small = 400
    n = n_small
    _, P_true, A = get_ground_truth_and_adj(n, get_tabulation)
    estimator, history = graphhist(A;
        starting_assignment_rule = EigenStart(),
        maxitr = Int(1e7),
        stop_rule = PreviousBestValue(10000))

    estimated, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)
    estimated_block = NetworkHistogram.GraphShapeHist(estimator)

    mvberns_hat = MVBernoulli.from_tabulation.(estimated.θ)
    marginals_hat = MVBernoulli.marginals.(mvberns_hat)
    correlations_hat = MVBernoulli.correlation_matrix.(mvberns_hat)

    P = zeros(n, n, 3)
    P[:, :, 1] = get_p_matrix([m[1] for m in marginals_hat], estimated.node_labels)
    P[:, :, 2] = get_p_matrix([m[2] for m in marginals_hat], estimated.node_labels)
    P[:, :, 3] = get_p_matrix([m[3] for m in correlations_hat], estimated.node_labels)

    ##

    sorted_labels = sortperm(estimated.node_labels, rev = false)
    sorted_labels_block = sortperm(estimated_block.node_labels, rev = true)

    mvberns_block = MVBernoulli.from_tabulation.(estimated_block.θ)


    P_tabulation_true = zeros(n, n, 4)
    for i in 1:n
        for j in 1:n
            P_tabulation_true[i, j, :] = get_tabulation(i / n, j / n)
        end
    end

    P_tabulation = zeros(n, n, 4)
    P_tabulation_block = zeros(n, n, 4)
    for i in 1:4
        P_tabulation[:, :, i] = get_p_matrix(
            [m.tabulation.p[i] for m in mvberns_hat], estimated.node_labels)
        P_tabulation_block[:, :, i] = get_p_matrix(
            [m.tabulation.p[i] for m in mvberns_block], estimated_block.node_labels)[
            sorted_labels_block, sorted_labels_block]
    end



    with_theme(theme_latexfonts()) do
        colormap = :lipari
        fig = Figure(size = (600, 300), fontsize = 16)
        ax11 = Axis(fig[1, 1], aspect = 1, title = L"w^{(1)}", ylabel = "Ground truth")
        ax12 = Axis(fig[1, 2], aspect = 1, title = L"w^{(2)}")
        ax13 = Axis(fig[1, 3], aspect = 1, title = L"w^{(3)}")
        ax14 = Axis(fig[1, 4], aspect = 1, title = L"w^{(4)}")
        ax21 = Axis(fig[2, 1], aspect = 1, ylabel = "Estimated")
        ax22 = Axis(fig[2, 2], aspect = 1)
        ax23 = Axis(fig[2, 3], aspect = 1)
        ax24 = Axis(fig[2, 4], aspect = 1)
        heatmap!(ax11, P_tabulation_true[:, :, 1], colormap = colormap, colorrange = (0, 1))
        heatmap!(ax12, P_tabulation_true[:, :, 2], colormap = colormap, colorrange = (0, 1))
        heatmap!(ax13, P_tabulation_true[:, :, 3], colormap = colormap, colorrange = (0, 1))
        heatmap!(ax14, P_tabulation_true[:, :, 4], colormap = colormap, colorrange = (0, 1))
        heatmap!(ax21, P_tabulation[sorted_labels, sorted_labels, 1],
            colormap = colormap, colorrange = (0, 1))
        heatmap!(ax22, P_tabulation[sorted_labels, sorted_labels, 2],
            colormap = colormap, colorrange = (0, 1))
        heatmap!(ax23, P_tabulation[sorted_labels, sorted_labels, 3],
            colormap = colormap, colorrange = (0, 1))
        heatmap!(ax24, P_tabulation[sorted_labels, sorted_labels, 4],
            colormap = colormap, colorrange = (0, 1))
        hidedecorations!.([ax12, ax13, ax14, ax22, ax23, ax24])
        hidedecorations!.([ax11, ax21], label = false)
        rowgap!(fig.layout, Relative(0.01))
        colgap!(fig.layout, Relative(0.01))

        gd = fig[1:2, 5] = GridLayout(1, 1)

        Colorbar(gd[1, 1], colorrange = (0, 1),
            colormap = colormap, vertical = true, flipaxis = true, height = Relative(0.8))
        display(fig)
            save("experiments/presentation_approx.png",
                fig, px_per_unit = 2)
    end
end
