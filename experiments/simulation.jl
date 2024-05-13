

using NetworkHistogram
using Makie, CairoMakie
using LinearAlgebra
using Random
using MVBernoulli
using Statistics
using LaTeXStrings
Random.seed!(123354472456192348634612304864326748)
include("utils.jl")


SAVE_FIG = false

softmax(x::AbstractArray{T}; dims = 1) where {T} = softmax!(similar(x, float(T)), x; dims)

softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out .= exp.(x .- max_)
    else
        _zero, _one, _inf = T(0), T(1), T(Inf)
        @fastmath @. out = ifelse(
            isequal(max_, _inf), ifelse(isequal(x, _inf), _one, _zero), exp(x - max_))
    end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    out ./= tmp
end

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



_, P, A = get_ground_truth_and_adj(100, get_tabulation_extreme_correlation)
display(display_approx_and_data(P, A, 1:100, label = "Ground truth, n = 100"))


##
function mse(dist1::MVBernoulli.MultivariateBernoulli, dist2::MVBernoulli.MultivariateBernoulli)
    return mean((dist1.tabulation.p - dist2.tabulation.p).^2)
end
function mse(a::Array{MultivariateBernoulli{T}}, b::Array{MultivariateBernoulli{T}}) where {T}
    return mean(mse.(a, b))
end

ns = [200]
mse_n = zeros(length(ns))

for (index,n) in enumerate(ns)
    mvberns, P_true,A = get_ground_truth_and_adj(n)

    fig_ground_truth = display_approx_and_data(P_true, A, 1:n, label = "Ground truth, n = $n")
    display(fig_ground_truth)


    estimator, history = graphhist(A;
        starting_assignment_rule = EigenStart(),
        maxitr = Int(1e7),
        stop_rule = PreviousBestValue(10000))



    estimated, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)

    mvberns_hat = MVBernoulli.from_tabulation.(estimated.θ)
    marginals_hat = MVBernoulli.marginals.(mvberns_hat)
    correlations_hat = MVBernoulli.correlation_matrix.(mvberns_hat)
    p_berns_hat = similar(mvberns)
    for i in 1:n
        for j in 1:n
            p_berns_hat[i, j] = mvberns_hat[estimated.node_labels[i], estimated.node_labels[j]]
        end
    end

    mse_n[index] = mse(mvberns, p_berns_hat)

    P = zeros(n, n, 3)
    P[:, :, 1] = get_p_matrix([m[1] for m in marginals_hat], estimated.node_labels)
    P[:, :, 2] = get_p_matrix([m[2] for m in marginals_hat], estimated.node_labels)
    P[:, :, 3] = get_p_matrix([m[3] for m in correlations_hat], estimated.node_labels)


    sorted_labels = sortperm(estimated.node_labels, rev = true)

    fig_sorted_estimated_latent = display_approx_and_data(P, A, sorted_labels, label = "Sorted by estimated latents")
    fig_sorted_true_latent = display_approx_and_data(P, A, 1:n, label = "Sorted by true latents")
    display(display_approx_and_data(P,P,sorted_labels, label = "Sorted by estimated latents"))
    display(fig_sorted_estimated_latent)
    display(fig_sorted_true_latent)
end


fig = Figure()
axis = Axis(fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE")
lines!(axis, ns, mse_n, linewidth = 2)
display(fig)


##

n = 200
_, P_true, A = get_ground_truth_and_adj(n)
estimator, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e7),
    stop_rule = PreviousBestValue(10000))

estimated, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)

mvberns_hat = MVBernoulli.from_tabulation.(estimated.θ)
marginals_hat = MVBernoulli.marginals.(mvberns_hat)
correlations_hat = MVBernoulli.correlation_matrix.(mvberns_hat)

P = zeros(n, n, 3)
P[:, :, 1] = get_p_matrix([m[1] for m in marginals_hat], estimated.node_labels)
P[:, :, 2] = get_p_matrix([m[2] for m in marginals_hat], estimated.node_labels)
P[:, :, 3] = get_p_matrix([m[3] for m in correlations_hat], estimated.node_labels)
sorted_labels = sortperm(estimated.node_labels, rev = true)


n = 400
_, P_true, A = get_ground_truth_and_adj(n)
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
sorted_labels = sortperm(estimated.node_labels, rev = false)#, by = x -> (correlations_hat[x, x][3],x))
sorted_labels_big = sortperm(estimated_big.node_labels, rev=false)#, by = x -> (correlations_hat_big[x, x][3],x))


function make_fig()
    colormap = :lipari
    fig = Figure(size = (730, 800), fontsize =14)

    ax = Axis(fig[1, 1], aspect = 1, title = L"\mathrm{pr}(X_1=1)", ylabel = L"W")
    ax2 = Axis(fig[1, 2], aspect = 1, title = L"\mathrm{pr}(X_2=1)")
    ax3 = Axis(fig[1, 3], aspect = 1, title = L"\mathrm{cor}(X_1, X_2)")
    ax4 = Axis(fig[2, 1], aspect = 1, ylabel = L"\hat{W},n = 200")
    ax5 = Axis(fig[2, 2], aspect = 1)
    ax6 = Axis(fig[2, 3], aspect = 1)
    ax7 = Axis(fig[3, 1], aspect = 1, ylabel = L"\hat{W},n = 400")
    ax8 = Axis(fig[3, 2], aspect = 1)
    ax9 = Axis(fig[3, 3], aspect = 1)
    heatmap!(ax, P_true[:, :, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax2, P_true[:, :, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax3, P_true[:, :, 3], colormap = :balance, colorrange = (-1, 1))
    heatmap!(ax4, P[sorted_labels, sorted_labels, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax5, P[sorted_labels, sorted_labels, 2], colormap = colormap,colorrange = (0, 1))
    heatmap!(
        ax6, P[sorted_labels, sorted_labels, 3],
        colormap = :balance, colorrange=(-1,1))

    heatmap!(ax7, P_big[sorted_labels_big, sorted_labels_big, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax8, P_big[sorted_labels_big, sorted_labels_big, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax9, P_big[sorted_labels_big, sorted_labels_big, 3], colormap = :balance, colorrange = (-1, 1))
    hidedecorations!.([ax2, ax3, ax5, ax6, ax8, ax9])
    hidedecorations!.([ax, ax4,ax7], label = false)
    rowgap!(fig.layout, Relative(0.01))

    gd = fig[4, 1:3] = GridLayout(1, 3)

    Colorbar(gd[1,1:2], colorrange = (0, 1), label = "Probability",
        colormap = colormap, vertical = false, flipaxis = false, width = Relative(0.8))
    Colorbar(gd[1,3], colorrange = (-1, 1), label = "Correlation",
        colormap = :balance, vertical = false, flipaxis = false, width = Relative(0.8),
        ticks = [-1, 0, 1])
return fig
end


fig = with_theme(make_fig, theme_minimal())
colgap!(fig.layout, Relative(0.01))
fig

##

if SAVE_FIG
    save("experiments/ground_truth_and_estimated.pdf", fig)
end

##
