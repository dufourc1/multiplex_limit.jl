using NetworkHistogram
using Makie, CairoMakie
using LinearAlgebra
using Random
using MVBernoulli
using Statistics
using LaTeXStrings
using ProgressMeter
using StatsBase
using Base.Threads
using ParallelKMeans
using JLD

Random.seed!(1234)

if VERSION != v"1.10.1"
    @warn "experiments were run with Julia 1.10.1, you are currently running $VERSION"
end



ns = 300:200:2000
rep = 10


path_to_current_folder = @__DIR__
path_to_figure_folder = joinpath(path_to_current_folder, "../figures")
path_to_data_folder = joinpath(path_to_current_folder, "../data")
path_to_scratch = joinpath(path_to_current_folder, "../scratch")

if !isdir(path_to_scratch)
    mkdir(path_to_scratch)
end

include(joinpath(path_to_current_folder, "utils.jl"))


display_fig = false
save_to_scratch = true
load_from_scratch = false
load_ref = false


if load_ref
    postfix = "_ref"
else
    postfix = ""
end

function w1(x,y)
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


function w2(x,y)
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
        n, latents, get_tabulation)
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
visual_check = false

if visual_check
    n = 100
    w_min_max= zeros((n, n, 4))
    for i in 1:n
        for j in 1:n
            w_min_max[i, j, :] = w1(i / n, j / n)
        end
    end

    w_complex = zeros((n, n, 4))
    for i in 1:n
        for j in 1:n
            w_complex[i, j, :] = w2(i / n, j / n)
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

function estimate_and_mse(n, rep, get_tabulation; alpha = 0, beta = 1)
    mse_inter = -1 .* ones(rep)
    n_shapes = zeros(rep)
    resolution_shapes = zeros(rep)
    for i in 1:rep
        latents = rand(n)
        theta_star, _, A = get_ground_truth_and_adj(n, latents,get_tabulation)
        if alpha == 0
            h = NetworkHistogram.select_bandwidth(A)
        else
            h = n^(-1/(alpha+1))
        end

        estimator, history = graphhist(A; h=h,
            starting_assignment_rule = EigenStart(),
            maxitr = Int(1e7),
            stop_rule = PreviousBestValue(1000))
        #min_shapes = NetworkHistogram.get_num_blocks(estimator)
        #min_shapes = binomial(min_shapes, 2)
        if beta == 0 || alpha == 0
            estimated, bic = NetworkHistogram.get_best_smoothed_estimator(
                estimator, A; show_progress = true, max_iterations_stalled = 10, algo = Yinyang(), n_min = 20)
            if display_fig
                display(lines(bic[1], bic[2], color = :black))
            end
        elseif beta == 1
            estimated = NetworkHistogram.GraphShapeHist(estimator)
        else
            num_shapes = round(Int64, n^(2/(beta*(min(alpha,1) +1))))
            estimated = NetworkHistogram.GraphShapeHist(num_shapes, estimator, Hamerly())
        end

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
        n_shapes[i] = estimated.num_shapes
        resolution_shapes[i] = length(unique(estimated.node_labels))
    end
    return mean(mse_inter), std(mse_inter), mean(n_shapes), std(n_shapes), mean(resolution_shapes), std(resolution_shapes)
end



##


ns_smooth = ns[1]:ns[end]

##


std_mse = zeros(length(ns))
mse_n = ones(length(ns))
n_shapes = ones(length(ns))
std_n_shapes = ones(length(ns))
resolution_shapes = ones(length(ns))
std_resolution_shapes = ones(length(ns))
resolution_shapes = ones(length(ns))

beta = 1.6


if !load_from_scratch

    @showprogress for (index, n) in enumerate(ns)
        mse_n[index], std_mse[index], n_shapes[index], std_n_shapes[index], resolution_shapes[index], std_resolution_shapes[index] = estimate_and_mse(n, rep, w1, alpha = 1, beta=beta)
    end

    if save_to_scratch
        save(joinpath(path_to_scratch,"rates_w1.jld"), "mse", mse_n, "std_mse", std_mse,
            "n_shapes", n_shapes, "std_n_shapes", std_n_shapes, "resolution_shapes",
            resolution_shapes, "std_resolution_shapes", std_resolution_shapes)
    end
else
    mse_n, std_mse, n_shapes, std_n_shapes, resolution_shapes, std_resolution_shapes = load(joinpath(path_to_scratch, "rates_w1$postfix.jld"), "mse", "std_mse", "n_shapes", "std_n_shapes", "resolution_shapes", "std_resolution_shapes")
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
    if display_fig
        display(fig)
    end
    #save(joinpath(path_to_figure_folder,"rate_w1.png"), fig, px_per_unit = 2)
end

##
beta_complex = 2.2


std_mse_complex = zeros(length(ns))
mse_n_complex = ones(length(ns))
n_shapes_complex = ones(length(ns))
std_n_shapes_complex = ones(length(ns))
resolution_shapes_complex = ones(length(ns))
std_resolution_shapes_complex = ones(length(ns))
resolution_shapes_complex = ones(length(ns))


if !load_from_scratch

    @showprogress for (index, n) in enumerate(ns)
        mse_n_complex[index], std_mse_complex[index], n_shapes_complex[index], std_n_shapes_complex[index], resolution_shapes_complex[index], std_resolution_shapes_complex[index] = estimate_and_mse(
            n, rep, w2, alpha = 0.5, beta = beta_complex)
    end

    if save_to_scratch
        save(joinpath(path_to_scratch, "rates_w2.jld"), "mse", mse_n_complex,
            "std_mse", std_mse_complex, "n_shapes", n_shapes_complex, "std_n_shapes",
            std_n_shapes_complex, "resolution_shapes", resolution_shapes_complex,
            "std_resolution_shapes", std_resolution_shapes_complex)
    end
else
    mse_n_complex, std_mse_complex, n_shapes_complex, std_n_shapes_complex, resolution_shapes_complex, std_resolution_shapes_complex = load(
        joinpath(path_to_scratch, "rates_w2$postfix.jld"), "mse", "std_mse",
        "n_shapes", "std_n_shapes", "resolution_shapes", "std_resolution_shapes")
end

##

alpha_complex = 0.5
L = 4
constant_complex = 0.09


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
        axis, ns, mse_n_complex, linewidth = 2, color = (:black, 0.2))
    lines!(
        axis, ns, upper_bound_complex, linewidth = 2, color = :red,
        linestyle = :dash, label = L"%$constant_complex \left(\log(n)/n + %$L n^{-2\times %$alpha/(%$alpha_complex+1)}\right)")
    scatter!(
        axis, ns, mse_n_complex, color = :black)
    errorbars!(axis, ns, mse_n_complex, std_mse_complex,
        color = :black, whiskerwidth = 10)
    axislegend(axis)
    if display_fig
        display(fig)
    end
    #save(joinpath(path_to_scratch,"rate_w2.png"), fig, px_per_unit = 2)
end



##
beta_sin = 1.7


function w3(x, y)
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
n_shapes_sin = ones(length(ns))
std_n_shapes_sin = ones(length(ns))
resolution_shapes_sin = ones(length(ns))
std_resolution_shapes_sin = ones(length(ns))
resolution_shapes_sin = ones(length(ns))


if !load_from_scratch
    @showprogress for (index, n) in enumerate(ns)
        mse_n_sin[index], std_mse_sin[index],n_shapes_sin[index], std_n_shapes_sin[index], resolution_shapes_sin[index], std_resolution_shapes_sin[index] = estimate_and_mse(
            n, rep, w3, alpha = 1, beta = beta_sin)
    end

    if save_to_scratch
        save(joinpath(path_to_scratch, "rates_w3.jld"),
            "mse", mse_n_sin, "std_mse", std_mse_sin,
            "n_shapes", n_shapes_sin, "std_n_shapes", std_n_shapes_sin, "resolution_shapes",
            resolution_shapes_sin, "std_resolution_shapes", std_resolution_shapes_sin)
    end
else
    mse_n_sin, std_mse_sin, n_shapes_sin, std_n_shapes_sin, resolution_shapes_sin, std_resolution_shapes_sin = load(
        joinpath(path_to_scratch, "rates_w3$postfix.jld"), "mse", "std_mse",
        "n_shapes", "std_n_shapes", "resolution_shapes", "std_resolution_shapes")
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
    if display_fig
        display(fig)
    end
    #save(joinpath(path_to_figure_folder,"rates_w3.png"), fig, px_per_unit = 2)
end

##





function show_mse_upper_bound!(axis,mse_to_plot, std_mse_to_plot, ns, upper_bound_to_plot, ns_ub,
                                color_mse = :black, color_upper_bound = :red, alpha = 1,
                                transparency = 0.2, label_ub = true,
                                style_main = :dash,
                                style_ub = :dash,
                                markershape = :circle;
                                marker_label = "",
                                show_std = false)
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
    if show_std
        errorbars!(axis, ns, mse_to_plot, std_mse_to_plot,
            color = color_mse)
    end
    return sc, ub
end

for show_std in [true,false]

    with_theme(theme_latexfonts()) do

        fig = Figure(size = (800, 300), fontsize = 16)
        axis = Axis(
            fig[1, 1], xlabel = L"n", ylabel = L"$||\hat{\theta}-\theta||^2/n^2$")
        sc, ub = show_mse_upper_bound!(
            axis, mse_n, std_mse, ns, upper_bound_smooth, ns_smooth, :black, :black, 1, 0.5, true, :dash,:dash, marker_label = L"W_1", show_std = show_std)
        sc_sin, ub_sin = show_mse_upper_bound!(
            axis, mse_n_sin, std_mse_sin,
            ns, upper_bound_sin_smooth, ns_smooth, :black, :black, 1, 0.5, false, :dash, :dash, :cross, marker_label = L"W_3", show_std = show_std)
        sc_complex, ub_complex = show_mse_upper_bound!(axis, mse_n_complex, std_mse_complex,
            ns, upper_bound_complex_smooth, ns_smooth, :blue, :blue, 0.5, 0.5,
            true, :solid, :solid, :rect, marker_label = L"W_2", show_std = show_std)

        axislegend(axis, [ub, sc, sc_sin, ub_complex, sc_complex],
            [L"C_1\left(\log(n)/n + 4 n^{-1}\right)", L"W_1", L"W_3",
                L"C_{0.5}\left(\log(n)/n + 4 n^{-2/3}\right)", L"W_2"], nbanks = 3, position = :rt,
                margin = (5,10,5,10))
        if display_fig
            display(fig)
        end
        file_name = show_std ? "Fig2_rates_std.png" : "Fig2_rates.png"
        save(joinpath(path_to_figure_folder,file_name), fig, px_per_unit = 2)
    end

end

##

ub_resolution = ns_smooth .^(0.5)
ub_resolution_sin = ns_smooth .^(0.5)
ub_resolution_complex = ns_smooth .^(2/3)


with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 300), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = L"n", ylabel = L"k")

    sc, ub = show_mse_upper_bound!(
        axis, resolution_shapes, std_resolution_shapes, ns, ub_resolution, ns_smooth, :black,
        :black, 1, 0.5, true, :dash, :dash, marker_label = L"W_1", show_std = true)
    sc_sin, ub_sin = show_mse_upper_bound!(
        axis, resolution_shapes_sin, std_resolution_shapes_sin, ns,
        ub_resolution_sin, ns_smooth, :black, :black, 1,
        0.5, false, :dash, :dash, :cross, marker_label = L"W_3", show_std = true)
    sc_complex, ub_complex = show_mse_upper_bound!(
        axis, resolution_shapes_complex, std_resolution_shapes_complex, ns,
        ub_resolution_complex, ns_smooth, :blue, :blue, 0.5, 0.5,
        true, :solid, :solid, :rect, marker_label = L"W_2", show_std = true)

    axislegend(axis, [ub, sc, sc_sin, ub_complex, sc_complex],
        [L"\sqrt{n}", L"W_1", L"W_3",
            L"n^{2/3}", L"W_2"], nbanks = 3, position = :lt,
        margin = (5, 10, 5, 10))
    if display_fig
        display(fig)
    end
    #save(joinpath(path_to_figure_folder,"rate_ws_k_num_blocks.png"), fig, px_per_unit = 2)
end


##

ub_shapes = binomial.(ub_resolution .+1, 2)
ub_shapes_sin = binomial.(ub_resolution_sin .+1, 2)
ub_shapes_complex = binomial.(ub_resolution_complex .+1, 2)

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 300), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = L"n", ylabel = L"s")

    sc, ub = show_mse_upper_bound!(
        axis, n_shapes, std_n_shapes, ns, ub_shapes, ns_smooth, :black,
        :black, 1, 0.5, true, :dash, :dash, marker_label = L"W_1")
    sc_sin, ub_sin = show_mse_upper_bound!(
        axis, n_shapes_sin, std_n_shapes_sin, ns,
        ub_shapes_sin, ns_smooth, :black, :black, 1,
        0.5, false, :dash, :dash, :cross, marker_label = L"W_3")
    sc_complex, ub_complex = show_mse_upper_bound!(
        axis, n_shapes_complex, std_n_shapes_complex, ns,
        ub_shapes_complex, ns_smooth, :blue, :blue, 0.5, 0.5,
        true, :solid, :solid, :rect, marker_label = L"W_2")

    axislegend(axis, [ub, sc, sc_sin, ub_complex, sc_complex],
        [L"\sqrt{n}", L"W_1", L"W_3",
            L"n^{2/3}", L"W_2"], nbanks = 3, position = :lt,
        margin = (5, 10, 5, 10))
    if display_fig
        display(fig)
    end
    #save(joinpath(path_to_figure_folder, "rate_ws_s_num_shapes.png"), fig, px_per_unit = 2)
end


##

# compute betas
beta = log.(resolution_shapes) .* 2 ./ log.(n_shapes)
beta_sin = log.(resolution_shapes_sin) .* 2 ./ log.(n_shapes_sin)
beta_complex = log.(resolution_shapes_complex) .* 2 ./ log.(n_shapes_complex)

println("β_w1: $(mean(beta)) ± $(std(beta))")
println("β_w2: $(mean(beta_complex)) ± $(std(beta_complex))")
println("β_w3: $(mean(beta_sin)) ± $(std(beta_sin))")
