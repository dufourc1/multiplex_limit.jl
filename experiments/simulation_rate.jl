
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



function get_tabulation_abs_min(x,y)
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


function get_tabulation_weird(x,y)
    tabulation = zeros(4)
    if x == y
        tabulation[1] = 1
        return tabulation
    end
    p_0 = abs(x-y)^0.5
    p_1 = exp(-0.5*abs(x-y))
    p_2 = min(x,y)
    p_3 = exp(-min(x,y)^(3/4))
    tabulation[1] = p_0
    tabulation[2] = p_1
    tabulation[3] = p_2
    tabulation[4] = p_3
    tabulation ./= sum(tabulation)
    return tabulation
end


function get_ground_truth_and_adj(
        n, latents = collect(1:n) ./ n, get_tabulation = get_tabulation_weird)
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
w_min_max= zeros((n, n, 4))
for i in 1:n
    for j in 1:n
        w_min_max[i, j, :] = get_tabulation_abs_min(i / n, j / n)
    end
end

w_complex = zeros((n, n, 4))
for i in 1:n
    for j in 1:n
        w_complex[i, j, :] = get_tabulation_weird(i / n, j / n)
    end
end


for w in [w_min_max,w_complex]

    f = Figure()
    ax1 = Axis(f[1, 1], aspect = 1, title = "[0, 0]")
    ax2 = Axis(f[1, 2], aspect = 1, title = "[1, 0]")
    ax3 = Axis(f[2, 1], aspect = 1, title = "[0, 1]")
    ax4 = Axis(f[2, 2], aspect = 1, title = "[1, 1]")
    mini = 0
    maxi = 1
    heatmap!(ax1, w[:, :, 1], colormap = :lipari, colorrange = (mini, maxi))
    heatmap!(ax2, w[:, :, 2], colormap = :lipari, colorrange = (mini, maxi))
    heatmap!(ax3, w[:, :, 3], colormap = :lipari, colorrange = (mini, maxi))
    heatmap!(ax4, w[:, :, 4], colormap = :lipari, colorrange = (mini, maxi))
    hidedecorations!.([ax1, ax2, ax3, ax4])
    cb = Colorbar(f[1:2, end + 1], colorrange = (mini, maxi), colormap = :lipari,
        vertical = true, height = Relative(0.8))
    display(f)

end


##
function mse(
        dist1::MVBernoulli.MultivariateBernoulli, dist2::MVBernoulli.MultivariateBernoulli)
    return mean((dist1.tabulation.p - dist2.tabulation.p) .^ 2)
end

function mse(
        a::Array{MultivariateBernoulli{T}}, b::Array{MultivariateBernoulli{T}}) where {T}
    return mean(mse.(a, b))
end

function estimate_and_mse(n, rep = 10, get_tabulation = get_tabulation_abs_min)
    mse_inter = -1 .* ones(rep)
    h = 1/sqrt(n)
    for i in 1:rep
        latents = rand(n)
        theta_star, _, A = get_ground_truth_and_adj(n, latents,get_tabulation)
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



##

ns = 100:200:2000
rep = 10

ns_smooth = 300:ns[end]

##


std_mse = zeros(length(ns))
mse_n = ones(length(ns))

fig = Figure()
axis = Axis(
    fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE", yscale = log, xscale = log)

@showprogress for (index, n) in enumerate(ns)
    mse_n[index], std_mse[index] = estimate_and_mse(n, rep, get_tabulation_abs_min)
    lines!(
        axis, ns[1:index], mse_n[1:index], linewidth = 2, color = (:black,0.8), label = "average")
    errorbars!(axis, ns[1:index], mse_n[1:index],std_mse[1:index], color = :gray, label = "std", whiskerwidth = 10)
    if index == 1
        axislegend(axis)
    end
    display(fig)
end

##

alpha = 1
constant = 0.06
L =4


power_holder_smooth = -2*alpha/(alpha+1)
upper_bound = (log.(ns) ./ ns .+ L*ns .^ power_holder_smooth)*constant

upper_bound_smooth = (log.(ns_smooth) ./ ns_smooth .+ L*ns_smooth .^ power_holder_smooth)*constant


with_theme(theme_latexfonts()) do
    fig = Figure(size = (800,400), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = L"n", ylabel = L"\text{MSE}",
        yscale = identity, xscale=identity,
        #xtickformat = values -> ["$(log(value))" for value in values],
        #ytickformat = values -> ["$(log(value))" for value in values],
        title = L"\text{Independent layers: } \mathrm{pr}(X_1)=\min(x,y) \text{ and } \mathrm{pr}(X_2)=|x-y|")
    lines!(
        axis, ns, mse_n, linewidth = 2, color = (:black, 0.2))
    lines!(
        axis, ns, upper_bound, linewidth = 2, color = :red, linestyle = :dash, label = L"%$constant \left(\log(n)/n + %$L n^{-1}\right)")
    scatter!(
        axis, ns, mse_n, color = :black)
    errorbars!(axis, ns, mse_n, std_mse,
        color = :black, whiskerwidth = 10)
    axislegend(axis)
    display(fig)
    save("experiments/multiplex_limit_rate.png", fig, px_per_unit = 2)
end

##


std_mse_complex = zeros(length(ns))
mse_n_complex = ones(length(ns))

fig = Figure()
axis = Axis(
    fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE", yscale = log, xscale = log)

@showprogress for (index, n) in enumerate(ns)
    mse_n_complex[index], std_mse_complex[index] = estimate_and_mse(n, rep, get_tabulation_weird)
    lines!(
        axis, ns[1:index], mse_n_complex[1:index], linewidth = 2,
        color = (:black, 0.8), label = "average")
    errorbars!(axis, ns[1:index], mse_n_complex[1:index], std_mse_complex[1:index],
        color = :gray, label = "std", whiskerwidth = 10)
    if index == 1
        axislegend(axis)
    end
    display(fig)
end

##

alpha_complex = 0.5
L = 4
constant_complex = 0.027


power_holder_smooth_complex = -2 * alpha_complex / (alpha_complex + 1)
upper_bound_complex = (log.(ns) ./ ns .+ L*ns .^ power_holder_smooth_complex) * constant_complex

upper_bound_complex_smooth = (log.(ns_smooth) ./ ns_smooth .+ L*ns_smooth .^ power_holder_smooth_complex) * constant_complex

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 400), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = L"n", ylabel = L"\text{MSE}",
        yscale = identity, xscale = identity,
        #xtickformat = values -> ["$(log(value))" for value in values],
        #ytickformat = values -> ["$(log(value))" for value in values],
        title = "Dependent layers")
    lines!(
        axis, ns[2:end], mse_n_complex[2:end], linewidth = 2, color = (:black, 0.2))
    lines!(
        axis, ns[2:end], upper_bound_complex[2:end], linewidth = 2, color = :red,
        linestyle = :dash, label = L"%$constant_complex \left(\log(n)/n + %$L n^{-2\times %$alpha/(%$alpha_complex+1)}\right)")
    scatter!(
        axis, ns[2:end], mse_n_complex[2:end], color = :black)
    errorbars!(axis, ns[2:end], mse_n_complex[2:end], std_mse_complex[2:end],
        color = :black, whiskerwidth = 10)
    axislegend(axis)
    display(fig)
    save("experiments/multiplex_limit_rate_higher_alpha.png", fig, px_per_unit = 2)
end



##


function get_tabulation_sin(x, y)
    tabulation = zeros(4)
    tabulation[1] = 3 * x * y
    tabulation[2] = 3 * sin(2 * π * x) * sin(2 * π * y)
    tabulation[3] = exp(-3 * ((x - 0.5)^2 + (y - 0.5)^2))
    tabulation[4] = 2 - 3 * (x + y)
    #tabulation[1] *= 1.5
    softmax!(tabulation)
    return tabulation
end




std_mse_sin = zeros(length(ns))
mse_n_sin = ones(length(ns))

fig = Figure()
axis = Axis(
    fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE", yscale = log, xscale = log)

@showprogress for (index, n) in enumerate(ns)
    mse_n_sin[index], std_mse_sin[index] = estimate_and_mse(
        n, rep, get_tabulation_sin)
    lines!(
        axis, ns[1:index], mse_n_sin[1:index], linewidth = 2,
        color = (:black, 0.8), label = "average")
    errorbars!(axis, ns[1:index], mse_n_sin[1:index], std_mse_sin[1:index],
        color = :gray, label = "std", whiskerwidth = 10)
    if index == 1
        axislegend(axis)
    end
    display(fig)
end

##

alpha_sin = 1
L = 4
constant_sin = 0.06

power_holder_smooth_sin = -2 * alpha_sin / (alpha_sin + 1)
upper_bound_sin = (log.(ns) ./ ns .+ L * ns .^ power_holder_smooth_sin) *
                      constant_sin

upper_bound_sin_smooth = (log.(ns_smooth) ./ ns_smooth .+ L * ns_smooth .^ power_holder_smooth_sin) * constant_sin

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 400), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = L"n", ylabel = L"\text{MSE}",
        yscale = identity, xscale = identity,
        #xtickformat = values -> ["$(log(value))" for value in values],
        #ytickformat = values -> ["$(log(value))" for value in values],
        title = "Dependent layers")
    lines!(
        axis, ns, mse_n_sin, linewidth = 2, color = (:black, 0.2))
    lines!(
        axis, ns, upper_bound_sin, linewidth = 2, color = :red,
        linestyle = :dash, label = L"%$constant_sin \left(\log(n)/n + %$L n^{-2\times %$alpha/(%$alpha_sin+1)}\right)")
    scatter!(
        axis, ns, mse_n_sin, color = :black)
    errorbars!(axis, ns, mse_n_sin, std_mse_sin,
        color = :black, whiskerwidth = 10)
    axislegend(axis)
    display(fig)
    save("experiments/multiplex_limit_rate_sin.png", fig, px_per_unit = 2)
end


##

function show_mse_upper_bound!(axis,mse_to_plot, std_mse_to_plot, ns, upper_bound_to_plot, ns_ub,
                                color_mse = :black, color_upper_bound = :red, alpha = 1,
                                transparency = 0.2, label_ub = true,
                                style_main = :dash,
                                style_ub = :dash,
                                markershape = :circle;
                                marker_label = "")
    if alpha == 1
        label_upper_bound = L"C_1\left(\log(n)/n + 4 n^{-1}\right)"
    elseif alpha == 0.5
        label_upper_bound = L"C_{0.5}\left(\log(n)/n + 4 n^{-2/3}\right)"
    else
        label_upper_bound = L"C\left(\log(n)/n + 4 n^{-2\times %$alpha/(%$alpha+1)}\right)"
    end
    #lines!(
    #    axis, ns, mse_to_plot, linewidth = 1, color = (color_mse, transparency),
    #    linestyle = style_main)
    if label_ub
        ub = lines!(
            axis, ns_ub, upper_bound_to_plot, linewidth = 1, color = color_upper_bound,
            linestyle = style_ub, label = label_upper_bound)
    else
        ub = lines!(
            axis, ns_ub, upper_bound_to_plot, linewidth = 1, color = color_upper_bound,
            linestyle = style_ub)
    end
    sc = scatter!(
        axis, ns, mse_to_plot, color = color_mse, marker = markershape, label = marker_label)
    #errorbars!(axis, ns, mse_to_plot, std_mse_to_plot,
    #    color = color_mse)
    return sc, ub
end


with_theme(theme_latexfonts()) do

    fig = Figure(size = (800, 300), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = L"n", ylabel = L"$||\hat{\theta}-\theta||^2/n^2$")



    sc, ub = show_mse_upper_bound!(
        axis, mse_n[2:end], std_mse[2:end], ns[2:end], upper_bound_smooth, ns_smooth, :black, :black, 1, 0.5, true, :dash,:dash, marker_label = L"W_1")
    sc_sin, ub_sin = show_mse_upper_bound!(
        axis, mse_n_sin[2:end], std_mse_sin[2:end],
        ns[2:end], upper_bound_sin_smooth, ns_smooth, :black, :black, 1, 0.5, false, :dash, :dash, :cross, marker_label = L"W_3")
    sc_complex, ub_complex = show_mse_upper_bound!(axis, mse_n_complex[2:end], std_mse_complex[2:end],
        ns[2:end], upper_bound_complex_smooth, ns_smooth, :blue, :blue, 0.5, 0.5,
        true, :solid, :solid, :rect, marker_label = L"W_2")

    axislegend(axis, [ub, sc, sc_sin, ub_complex, sc_complex],
        [L"C_1\left(\log(n)/n + 4 n^{-1}\right)", L"W_1", L"W_3",
            L"C_{0.5}\left(\log(n)/n + 4 n^{-2/3}\right)", L"W_2"], nbanks = 3, position = :rt,
            margin = (5,10,5,10))
    display(fig)
    save("experiments/multiplex_limit_rate_all.png", fig, px_per_unit = 2)
end
