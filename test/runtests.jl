using multiplex_limit
using Test
using Aqua

@testset "multiplex_limit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(multiplex_limit)
    end
    # Write your tests here.
end
