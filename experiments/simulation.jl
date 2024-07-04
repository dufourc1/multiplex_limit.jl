

using NetworkHistogram
using Makie, CairoMakie
using LinearAlgebra
using Random
using MVBernoulli
using Statistics
using LaTeXStrings

Random.seed!(12345)
include("utils.jl")


SAVE_FIG = false

function display_approx_and_data(P, A, sorting; label = "", colormap = :lipari)
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1], aspect = 1, title = "X_1", ylabel = "Histogram")
    ax2 = Axis(fig[1, 2], aspect = 1, title = "X_2")
    ax3 = Axis(fig[1, 3], aspect = 1, title = "Correlation")
    ax4 = Axis(fig[2, 1], aspect = 1, ylabel = "Adjacency matrix")
    ax5 = Axis(fig[2, 2], aspect = 1)
    ax6 = Axis(fig[2, 3], aspect = 1)
    heatmap!(ax, P[sorting, sorting, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax2, P[sorting, sorting, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax3, P[sorting, sorting, 3], colormap = :balance, colorrange = (-1, 1))
    heatmap!(ax4, A[sorting, sorting, 1], colormap = :binary)
    heatmap!(ax5, A[sorting, sorting, 2], colormap = :binary)
    heatmap!(
        ax6, A[sorting, sorting, 1] .* A[sorting, sorting, 2],
        colormap = :binary)
    Colorbar(fig[1, end + 1], colorrange = (0, 1),
        colormap = colormap, vertical = true, height = Relative(0.8))
    Colorbar(fig[2, end ], colorrange = (-1, 1), label = "Correlation",
        colormap = :balance, vertical = true, height = Relative(0.8))
    hidedecorations!.([ax2, ax3, ax5, ax6])
    hidedecorations!.([ax, ax4], label = false)
    if label != ""
        supertitle = Label(fig[1, :, Top()], label, font = :bold,
            justification = :center,
            padding = (0, 0, 30, 0), fontsize=20)
    end
    return fig
end


function get_tabulation(x, y)
    tabulation = zeros(4)
    tabulation[1] =  3*x*y
    tabulation[2] =  3*sin(2 * π * x) * sin(2 * π * y)
    tabulation[3] =  exp(-3 * ((x - 0.5)^2 + (y - 0.5)^2))
    tabulation[4] =  2-3 * (x + y)
    #tabulation[1] *= 1.5
    softmax!(tabulation)
    return tabulation
end


function get_independent(x,y)
    tabulation = get_tabulation(x,y)
    mvbern = MVBernoulli.from_tabulation(tabulation)
    marginals = MVBernoulli.marginals(mvbern)
    tabulation = zeros(4)
    tabulation[1] = (1-marginals[1])*(1-marginals[2])
    tabulation[2] = marginals[1]*(1-marginals[2])
    tabulation[3] = (1-marginals[1])*marginals[2]
    tabulation[4] = marginals[1]*marginals[2]
    return tabulation
end




function get_tabulation_extreme_correlation(x,y)
    tab_corr_1 = [0.6,0.0,0.0,0.4]
    tab_corr_neg = [0.0,0.6,0.4,0.0]
    coeff_to_corr_1 = abs.(sin(π * (x/2)) * sin(π * (y/2)))
    coeff_to_corr_neg = 1 - coeff_to_corr_1
    tabulation = tab_corr_1 * coeff_to_corr_1 + tab_corr_neg * coeff_to_corr_neg
    if all(tabulation .== 0)
        tabulation = [25,0.25,0.25,0.25]
    end
    tabulation[1] *= 1.5
    tabulation = tabulation ./ sum(tabulation)
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

n = 100
w = zeros((n, n, 4))
for i in 1:n
    for j in 1:n
        w[i, j, :] = get_independent(i / n, j / n)
    end
end

f = Figure()
ax1 = Axis(f[1, 1], aspect = 1, title = "[0, 0]")
ax2 = Axis(f[1, 2], aspect = 1, title = "[1, 0]")
ax3 = Axis(f[2, 1], aspect = 1, title = "[0, 1]")
ax4 = Axis(f[2, 2], aspect = 1, title = "[1, 1]")
heatmap!(ax1, w[:, :, 1], colormap = :lipari, colorrange = (0, 1))
heatmap!(ax2, w[:, :, 2], colormap = :lipari, colorrange = (0, 1))
heatmap!(ax3, w[:, :, 3], colormap = :lipari, colorrange = (0, 1))
heatmap!(ax4, w[:, :, 4], colormap = :lipari, colorrange = (0, 1))
hidedecorations!.([ax1,ax2, ax3, ax4])
cb = Colorbar(f[1:2, end+1], colorrange = (0, 1), colormap = :lipari, vertical = true, height = Relative(0.8))
display(f)


_, P, A = get_ground_truth_and_adj(100, get_independent)
display(display_approx_and_data(P, A, 1:100, label = "Ground truth, n = 100"))



##

n_small = 300
n = n_small
_, P_true, A = get_ground_truth_and_adj(n, get_independent)
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
sorted_labels = sortperm(estimated.node_labels, rev = true)
sorted_labels_block = sortperm(estimated_block.node_labels, rev = true)


mvberns_block = MVBernoulli.from_tabulation.(estimated_block.θ)

##

P_tabulation_true = zeros(n, n, 4)
for i in 1:n
    for j in 1:n
        P_tabulation_true[i, j, :] = get_independent(i / n, j / n)
    end
end



P_tabulation = zeros(n, n, 4)
P_tabulation_block = zeros(n, n, 4)
for i in 1:4
    P_tabulation[:, :, i] = get_p_matrix([m.tabulation.p[i] for m in mvberns_hat], estimated.node_labels)
    P_tabulation_block[:, :, i] = get_p_matrix(
        [m.tabulation.p[i] for m in mvberns_block], estimated_block.node_labels)[sorted_labels_block, sorted_labels_block]
end

##

with_theme(theme_latexfonts()) do
    colormap = :lipari
    fig = Figure(size = (600, 300), fontsize = 16)
    ax11 = Axis(fig[1, 1], aspect = 1, title = L"w^{(1)}", ylabel = L"W^*")
    ax12 = Axis(fig[1, 2], aspect = 1, title = L"w^{(2)}")
    ax13 = Axis(fig[1, 3], aspect = 1, title = L"w^{(3)}")
    ax14 = Axis(fig[1, 4], aspect = 1, title = L"w^{(4)}")
    ax21 = Axis(fig[2, 1], aspect = 1, ylabel = "SSM")
    ax22 = Axis(fig[2, 2], aspect = 1)
    ax23 = Axis(fig[2, 3], aspect = 1)
    ax24 = Axis(fig[2, 4], aspect = 1)
    heatmap!(ax11, P_tabulation_true[:, :, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax12, P_tabulation_true[:, :, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax13, P_tabulation_true[:, :, 3], colormap = colormap, colorrange = (0,1))
    heatmap!(ax14, P_tabulation_true[:, :, 4], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax21, P_tabulation[sorted_labels, sorted_labels, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax22, P_tabulation[sorted_labels, sorted_labels, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax23, P_tabulation[sorted_labels, sorted_labels, 3], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax24, P_tabulation[sorted_labels, sorted_labels, 4], colormap = colormap, colorrange = (0, 1))
    hidedecorations!.([ax12, ax13, ax14, ax22, ax23, ax24])
    hidedecorations!.([ax11, ax21], label = false)
    rowgap!(fig.layout, Relative(0.01))
    colgap!(fig.layout, Relative(0.01))

    gd = fig[1:2, 5] = GridLayout(1, 1)

    Colorbar(gd[1, 1], colorrange = (0, 1),
        colormap = colormap, vertical = true, flipaxis = true, height = Relative(0.8))
    display(fig)
    if SAVE_FIG
        save("experiments/decorated_graphon_and_approx_independent.png", fig, px_per_unit = 2)
    end
end


##
with_theme(theme_latexfonts()) do
    colormap = :lipari
    fig = Figure(size = (600, 300), fontsize = 16)
    ax11 = Axis(fig[1, 1], aspect = 1, title = L"w^{(1)}", ylabel = L"SBM")
    ax12 = Axis(fig[1, 2], aspect = 1, title = L"w^{(2)}")
    ax13 = Axis(fig[1, 3], aspect = 1, title = L"w^{(3)}")
    ax14 = Axis(fig[1, 4], aspect = 1, title = L"w^{(4)}")
    ax21 = Axis(fig[2, 1], aspect = 1, ylabel = L"SSM")
    ax22 = Axis(fig[2, 2], aspect = 1)
    ax23 = Axis(fig[2, 3], aspect = 1)
    ax24 = Axis(fig[2, 4], aspect = 1)
    heatmap!(ax11, P_tabulation_block[:, :, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax12, P_tabulation_block[:, :, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax13, P_tabulation_block[:, :, 3], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax14, P_tabulation_block[:, :, 4], colormap = colormap, colorrange = (0, 1))
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
    if SAVE_FIG
        save("experiments/SBM_SSM_independent.png", fig, px_per_unit = 2)
    end
end
##




n_big = 500
n = n_big

_, P_true, A = get_ground_truth_and_adj(n, get_independent)
estimator_big, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e7),
    stop_rule = PreviousBestValue(10000))

estimated_big, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator_big, A)

mvberns_hat_big = MVBernoulli.from_tabulation.(estimated_big.θ)
marginals_hat_big = MVBernoulli.marginals.(mvberns_hat_big)
correlations_hat_big = MVBernoulli.correlation_matrix.(mvberns_hat_big)

P_big = zeros(n, n, 3)
P_big[:, :, 1] = get_p_matrix([m[1] for m in marginals_hat_big], estimated_big.node_labels)
P_big[:, :, 2] = get_p_matrix([m[2] for m in marginals_hat_big], estimated_big.node_labels)
P_big[:, :, 3] = get_p_matrix([m[3] for m in correlations_hat_big], estimated_big.node_labels)


##

# adjust the order of the estimated labels to match the ground truth wkth rev
sorted_labels = sortperm(estimated.node_labels, rev = true)
sorted_labels_big = sortperm(estimated_big.node_labels, rev=false)


function make_fig(size = (570,500))
    colormap = :lipari
    fig = Figure(size =size, fontsize =16)

    ax11 = Axis(fig[1, 1], aspect = 1, ylabel = L"\mathrm{pr}(X_1=1)", title = L"W")
    ax12 = Axis(fig[1, 2], aspect = 1, title = L"\hat{W},\, n = %$n_small")
    ax13 = Axis(fig[1, 3], aspect = 1, title = L"\hat{W},\, n = %$n_big" )
    ax21 = Axis(fig[2, 1], aspect = 1, ylabel = L"\mathrm{pr}(X_2=1)")
    ax22 = Axis(fig[2, 2], aspect = 1)
    ax23 = Axis(fig[2, 3], aspect = 1)
    ax31 = Axis(fig[3, 1], aspect = 1, ylabel = L"\mathrm{cor}(X_1, X_2)")
    ax32 = Axis(fig[3, 2], aspect = 1)
    ax33 = Axis(fig[3, 3], aspect = 1)
    heatmap!(ax11, P_true[:, :, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax21, P_true[:, :, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax31, P_true[:, :, 3], colormap = :balance, colorrange = (-1, 1))
    heatmap!(ax12, P[sorted_labels, sorted_labels, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax22, P[sorted_labels, sorted_labels, 2], colormap = colormap,colorrange = (0, 1))
    heatmap!(
        ax32, P[sorted_labels, sorted_labels, 3],
        colormap = :balance, colorrange=(-1,1))

    heatmap!(ax13, P_big[sorted_labels_big, sorted_labels_big, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax23, P_big[sorted_labels_big, sorted_labels_big, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax33, P_big[sorted_labels_big, sorted_labels_big, 3], colormap = :balance, colorrange = (-1, 1))
    hidedecorations!.([ax12, ax12, ax22, ax23, ax32, ax33,ax13])
    hidedecorations!.([ax11, ax21, ax31], label = false)
    rowgap!(fig.layout, Relative(0.01))
    colgap!(fig.layout, Relative(0.01))

    gd = fig[1:3, 4] = GridLayout(3, 1)

    Colorbar(gd[1:2,1], colorrange = (0, 1),
        colormap = colormap, vertical = true, flipaxis = true, height = Relative(0.8))
    Colorbar(gd[3,1], colorrange = (-1.0, 1.0),
        colormap = :balance, vertical = true, flipaxis = true, height = Relative(0.8),
        ticks = [-1.0, 0.0, 1.0])
return fig
end

with_theme(theme_latexfonts()) do
    fig = make_fig()
    display(fig)
    if SAVE_FIG
        #save("experiments/ground_truth_and_estimated.pdf", fig)
        save("experiments/ground_truth_and_estimated_independent.png", fig, px_per_unit = 2)
    end
end

##

ns = [100, 300, 500]

_, P_true, A = get_ground_truth_and_adj(500, get_tabulation_extreme_correlation)
true_correlation = P_true[:, :, 3]
estimators = []

for (index, n) in enumerate(ns)
    _, P_true, A = get_ground_truth_and_adj(n, get_tabulation_extreme_correlation)
    estimator, history = graphhist(A;
        starting_assignment_rule = EigenStart(),
        maxitr = Int(1e7),
        stop_rule = PreviousBestValue(10000))

    estimated, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)
    push!(estimators, estimated)
end

##

order_sorting = [false,false,false]

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 250), fontsize = 16)

    ax = Axis(fig[1, 0], aspect = 1, title = L"\text{Ground truth}")
    heatmap!(ax, true_correlation,
        colormap = :balance, colorrange = (-1, 1))
    hidedecorations!(ax)
    for (index,estimator) in enumerate(estimators)
        correlation = get_p_matrix([m[3] for m in MVBernoulli.correlation_matrix.(MVBernoulli.from_tabulation.(estimator.θ))], estimator.node_labels)
        sorted_labels = sortperm(estimator.node_labels, rev = order_sorting[index])
        ax = Axis(fig[1, index], aspect = 1, title = L"n = %$(ns[index])")
        hidedecorations!(ax)
        heatmap!(ax, correlation[sorted_labels,sorted_labels], colormap = :balance, colorrange = (-1, 1))
    end
    Colorbar(fig[1,end+1], colorrange = (-1.0, 1.0),
        colormap = :balance, vertical = true, flipaxis = true, height = Relative(0.8),
        ticks = [-1.0, 0.0, 1.0])
    colgap!(fig.layout, Relative(0.01))
    save("experiments/extreme_corr_simulation.png", fig, px_per_unit = 2)
display(fig)

end
