function getci!(M,k,cmins,vmins) # inplace
    ci = CartesianIndices(size(M))
    for i in 1:k
        cmins[i] = ci[i]
        vmins[i] = M[i]
    end
    imin = findmin(vmins)[2]
    for i in firstindex(M)+k:lastindex(M)
        if M[i] > vmins[imin]
            cmins[imin] = ci[i]
            vmins[imin] = M[i]
            imin = findmin(vmins)[2]
        end
    end
    return cmins, vmins
end

function getci(M,k) # allocating
    cmins = Vector{CartesianIndex{2}}(undef,k)
    vmins = Vector{eltype(M)}(undef,k)
    return getci!(M,k,cmins,vmins)
end

function rectangle_from_coords(xb, yb, xt, yt)
    [xb yb
     xt yb
     xt yt
     xb yt
     xb yb
     NaN NaN]
end

function rectangle_around_cell(i, j)
    return rectangle_from_coords(i - 0.5, j - 0.5, i + 0.5, j + 0.5)
end

function edge_list_to_adjs(edge_lists = [
        "data/malaria_data/HVR_1.txt",
        "data/malaria_data/HVR_5.txt",
        "data/malaria_data/HVR_6.txt"])
    M = length(edge_lists)

    edges = [readdlm(e, ',', Int, '\n') for e in edge_lists]
    n = maximum([maximum(edges[i]) for i in eachindex(edges)])

    A = Array{Int, 3}(undef, n, n, M)
    A .= 0
    for i in 1:M
        for e in eachrow(edges[i])
            A[e[1], e[2], i] = 1
            A[e[2], e[1], i] = 1
        end
    end
    return A
end



function get_p_matrix(θ, node_labels)
    n = length(node_labels)
    pij = zeros(n, n)
    for i in 1:n
        for j in 1:n
            pij[i, j] = θ[node_labels[i], node_labels[j]]
        end
    end
    return pij
end



function orderered_latents(n,π)
    k = length(π)
    n_per_group = round.(Int, n .* π)
    latents = vcat([ones(Int, n_per_group[i]) .* i for i in 1:k]...)
    if length(latents) > n
        latents = latents[1:n]
    end
    return latents
end


function count_matches(x, y)
    sort!(x)
    sort!(y)
    i = j = 1
    matches = 0
    while i <= length(x) && j <= length(y)
        if x[i] == y[j]
            i += 1
            j += 1
            matches += 1
        elseif x[i] < y[j]
            i += 1
        else
            j += 1
        end
    end
    matches
end



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
    Colorbar(fig[2, end], colorrange = (-1, 1), label = "Correlation",
        colormap = :balance, vertical = true, height = Relative(0.8))
    hidedecorations!.([ax2, ax3, ax5, ax6])
    hidedecorations!.([ax, ax4], label = false)
    if label != ""
        supertitle = Label(fig[1, :, Top()], label, font = :bold,
            justification = :center,
            padding = (0, 0, 30, 0), fontsize = 20)
    end
    return fig
end
