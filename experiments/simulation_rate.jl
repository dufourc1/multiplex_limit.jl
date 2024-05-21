
using NetworkHistogram
using Makie, CairoMakie
using LinearAlgebra
using Random
using MVBernoulli
using Statistics
using LaTeXStrings
using multiplex_limit
using ProgressMeter
using StatsBase

Random.seed!(1234)
include("utils.jl")



function get_tabulation(x,y)
    tabulation = zeros(4)
    if x == y
        tabulation[1] = 1
        return tabulation
    end
    p_1 = abs(x-y)
    p_2 = min(x,y)
    tabulation[1] = (1-p_1) * (1-p_2)
    tabulation[2] = p_1 * (1-p_2)
    tabulation[3] = (1-p_1) * p_2
    tabulation[4] = p_1 * p_2
    return tabulation
end


function get_ground_truth_and_adj(n, latents = collect(1:n)./n)
    w = zeros((n, n, 4))
    for i in 1:n
        for j in 1:n
            w[i, j, :] = get_tabulation(latents[i], latents[j])
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

##


n = 100
w = zeros((n, n, 4))
for i in 1:n
    for j in 1:n
        w[i, j, :] = get_tabulation(i / n, j / n)
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
hidedecorations!.([ax1, ax2, ax3, ax4])
cb = Colorbar(f[1:2, end + 1], colorrange = (0, 1), colormap = :lipari,
    vertical = true, height = Relative(0.8))
display(f)

_, P, A = get_ground_truth_and_adj(100)
display(display_approx_and_data(P, A, 1:100, label = "Ground truth, n = 100"))


##
function mse(
        dist1::MVBernoulli.MultivariateBernoulli, dist2::MVBernoulli.MultivariateBernoulli)
    return mean((dist1.tabulation.p - dist2.tabulation.p) .^ 2)
end

function mse(
        a::Array{MultivariateBernoulli{T}}, b::Array{MultivariateBernoulli{T}}) where {T}
    return mean(mse.(a, b))
end

function estimate_and_mse(n, rep = 10)
    mse_inter = -1 .* ones(rep)
    h = 1/sqrt(n)
    for i in 1:rep
        theta_star, _, A = get_ground_truth_and_adj(n)
        estimator, history = graphhist(A; h=h,
            starting_assignment_rule = EigenStart(),
            maxitr = Int(1e7),
            stop_rule = PreviousBestValue(10000))
        estimated = NetworkHistogram.GraphShapeHist(estimator)
        mvberns_hat = MVBernoulli.from_tabulation.(estimated.θ)
        p_berns_hat = similar(theta_star)
        for i in 1:n
            for j in 1:n
                p_berns_hat[i, j] = mvberns_hat[
                    estimated.node_labels[i], estimated.node_labels[j]]
            end
        end
        mse_value = mse(theta_star, p_berns_hat)
        mse_inter[i] = mse_value
    end
    return mean(mse_inter), std(mse_inter)
end

ns = 100:100:1100
rep = 2
std_mse = zeros(length(ns))
upper_bound = log.(ns) ./ ns
mse_n = ones(length(ns))

fig = Figure()
axis = Axis(
    fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE", yscale = log, xscale = log)

@showprogress for (index, n) in enumerate(ns)
    mse_n[index], std_mse[index] = estimate_and_mse(n, rep)
    lines!(
        axis, ns[1:index], mse_n[1:index], linewidth = 2, color = :black, label = "average")
    errorbars!(axis, ns[1:index], mse_n[1:index],std_mse[1:index], color = :red, label = "std", whiskerwidth = 10)
    if index == 1
        fig[1, 2] = Legend(fig, axis, framevisible = false)
    end
    display(fig)
end





with_theme(theme_latexfonts()) do
    fig = Figure(size = (800,400), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE",
        yscale = log, xscale=log,
        title = "independent layers: min and abs")
    lines!(
        axis, ns, mse_n, linewidth = 2, color = :black, label = "average over 10 repetitions")
    errorbars!(axis, ns, mse_n, std_mse,
        color = :red, label = "std", whiskerwidth = 10)
    fig[1, 2] = Legend(fig, axis, framevisible = false)
    display(fig)
end

save("experiments/multiplex_limit_rate_.pdf", fig)
