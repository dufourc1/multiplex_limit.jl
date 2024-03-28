module multiplex_limit

using MVBernoulli
using Distributions

struct Multiplexon{T}
    θ::Matrix{MultivariateBernoulli{T}}
    π::Vector{T}
    num_layers::Int
end

function Multiplexon(θ::Matrix{MultivariateBernoulli{T}}, π::Vector{T},
        num_layers::Int = size(θ[1, 1], 1)) where {T}
    if size(θ, 1) != size(θ, 2)
        throw(ArgumentError("θ must be square"))
    end
    if length(π) != size(θ, 1)
        throw(ArgumentError("π must have the same length as the side of θ"))
    end
    if any(π .< 0)
        throw(ArgumentError("π must be a proportional to a probability distribution"))
    end
    if abs(sum(π) - 1) > 1e-6
        π = π ./ sum(π)
    end
    return Multiplexon{T}(θ, π, num_layers)
end



function random_multiplexon(num_groups::Int, num_layers::Int)
    params = [rand(2^num_layers) for _ in 1:(num_groups * (num_groups + 1) ÷ 2)]
    inter = MVBernoulli.from_tabulation(params[1] ./ sum(params[1]))
    θ = Matrix{typeof(inter)}(undef, num_groups, num_groups)
    index = 1
    for i in 1:num_groups
        for j in i:num_groups
            θ[i, j] = MVBernoulli.from_tabulation(params[index] ./ sum(params[index]))
            θ[j, i] = θ[i, j]
            index += 1
        end
    end
    π = rand(num_groups)
    return Multiplexon(θ, π, num_layers)
end


function Base.rand(s::Multiplexon, n::Int = 1, latents = rand(Categorical(s.π), n),
        x = Array{Int}(undef, n, n, s.num_layers))
    for i in 1:n
        for j in 1:n
            if i == j
               x[i, j, :] .= 0
            else
                x[i, j, :] .= rand(s.θ[latents[i], latents[j]])
                x[j, i, :] .= x[i, j, :]
            end
        end
    end
    return x, latents
end


# function Base.rand(s::Multiplexon, n::Int = 1, latents = rand(Categorical(s.π), n),
#         x = Array{Vector{Int}}(undef, n, n))
#     for j in 1:n
#         for i in 1:n
#             if i == j
#                 x[i, j] = zeros(Int, s.num_layers)
#             else
#                 x[i, j] = rand(s.θ[latents[i], latents[j]])
#                 x[j, i] = x[i, j]
#             end
#         end
#     end
#     return x, latents
# end



function get_matrix_moment(model::multiplex_limit.Multiplexon, moment_index)
    k = length(model.π)
    moment = Matrix{Float64}(undef, k, k)
    for i in 1:k
        for j in 1:k
            moment[i, j] = model.θ[i, j].ordinary_moments[moment_index + 1]
        end
    end
    return moment
end

end
