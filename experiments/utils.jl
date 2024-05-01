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
