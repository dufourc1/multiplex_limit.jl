
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

Random.seed!(123354472456192348634612304864326748)
include("utils.jl")


number_groups = 2
max_shapes = number_groups * (number_groups + 1) ÷ 2
multiplexon = multiplex_limit.random_multiplexon(number_groups, 2)
num_params = 2^2

function get_ground_truth(multiplexon, n)
    #latents = inverse_rle(1:number_groups, repeat([n÷number_groups], number_groups))
    A, latents =  rand(multiplexon, n)
    P = [multiplexon.θ[i,j] for i in latents, j in latents]
    return P,A
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

function estimate_and_mse(n, rep = 10)
    mse_inter = -1 .* ones(rep)
    for i in 1:rep
        mvberns, A = get_ground_truth(multiplexon, n)
        estimator, history = graphhist(A; h = n÷number_groups,
            starting_assignment_rule = EigenStart(),
            maxitr = Int(1e7),
            stop_rule = PreviousBestValue(10000))
        @assert NetworkHistogram.get_num_blocks(estimator) == number_groups
        estimated = NetworkHistogram.GraphShapeHist(max_shapes,estimator)
        mvberns_hat = MVBernoulli.from_tabulation.(estimated.θ)
        p_berns_hat = similar(mvberns)
        for i in 1:n
            for j in 1:n
                p_berns_hat[i, j] = mvberns_hat[
                    estimated.node_labels[i], estimated.node_labels[j]]
            end
        end
        mse_value = mse(mvberns, p_berns_hat)
        mse_inter[i] = mse_value
    end
    return mean(mse_inter), std(mse_inter)
end

ns = 100:100:2300
std_mse = zeros(length(ns))
upper_bound = max_shapes*num_params ./ (ns.^2) + log(max_shapes) ./ ns
mse_n = ones(length(ns))

fig = Figure()
axis = Axis(
    fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE", yscale = log)
xlims!(axis, 0,  maximum(ns))
lines!(
    axis, [-100], mse_n[1:1], linewidth = 2, color = :black, label = "average")
errorbars!(axis,[-100],[1.0], [0.0],
    color = :red, label = "std", whiskerwidth = 10)
lines!(axis, ns, upper_bound, linewidth = 2,
    color = :blue, linestyle = :dash, label = L" sL/n^2 + \log(s)/n")
fig[1, 2] = Legend(fig, axis, framevisible = false)
display(fig)

@showprogress for (index, n) in enumerate(ns)
    mse_n[index], std_mse[index] = estimate_and_mse(n)
    println("\n n = $n, mse = $(mse_n[index]), std = $(std_mse[index])")
    lines!(
        axis, ns[1:index], mse_n[1:index], linewidth = 2, color = :black, label = "average")
    #errorbars!(axis, ns[1:index], mse_n[1:index],std_mse[1:index], color = :red, label = "std", whiskerwidth = 10)
    display(fig)
end



with_theme(theme_latexfonts()) do
    fig = Figure(size = (800,400), fontsize = 16)
    axis = Axis(
        fig[1, 1], xlabel = "Number of nodes", ylabel = "MSE", yscale = log, title = "2 block SBM for 2 layers")
    xlims!(axis, 0, maximum(ns)+minimum(ns))
    lines!(
        axis, ns, mse_n, linewidth = 2, color = :black, label = "average over 10 repetitions")
    errorbars!(axis, ns, mse_n, std_mse,
        color = :red, label = "std", whiskerwidth = 10)
    lines!(axis, ns, upper_bound, linewidth = 2,
        color = :blue, linestyle = :dash, label = L" sL/n^2 + \log(s)/n")
    fig[1, 2] = Legend(fig, axis, framevisible = false)
    display(fig)
end

save("experiments/multiplex_limit_rate_.pdf", fig)
