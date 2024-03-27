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
