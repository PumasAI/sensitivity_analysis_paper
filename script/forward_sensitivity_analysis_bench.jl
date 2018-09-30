# =============================================================== #
# Small regime (2x3 Jacobian matrix)

include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays#, Profile, ProfileView

function make_lotkavolterra
f = function (u, p, t)
    a,b,c = p
    x, y = u
    dx = a*x - b*x*y
    dy = -c*y + x*y
    @SVector [dx, dy]
end

df = function (du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

com_df = function (du, u, p, t)
    a,b,c = p
    x, y, s1, s2, s3, s4, s5, s6 = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    #####################
    #     [a-by -bx]
    # J = [        ]
    #     [y    x-c]
    #####################
    J  = @SMatrix [a-b*y -b*x
                   y    x-c]
    JS = J*@SMatrix[s1 s3 s5
                    s2 s4 s6]
    G  = @SMatrix [x -x*y 0
                   0  0  -y]
    du[3:end] .= vec(JS+G)
    nothing
end

com_u0 = [u0...;zeros(6)]
comprob = ODEProblem(com_df, com_u0, tspan, p)
u0s = @SVector [1.,1.]; u0 = [1.,1.]; tspan = (0., 10.); sp = @SVector [1.5,1.0,3.0]; p = [1.5,1.0,3.0]

@time auto_sen(f, u0s, tspan, sp, Vern9(), abstol=1e-5,reltol=1e-7)
@time auto_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
@time diffeq_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
@time solve(comprob, Vern9(),abstol=1e-5,reltol=1e-7)
@btime auto_sen($f, $u0s, $tspan, $sp, $(Vern9()), abstol=1e-5,reltol=1e-7)
@btime auto_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
@btime diffeq_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
@btime solve($comprob, $(Vern9()),abstol=1e-5,reltol=1e-7)

#======================================
julia> @time auto_sen(f, u0s, tspan, sp)
 21.132961 seconds (35.99 M allocations: 1.818 GiB, 4.47% gc time)
2×3 SArray{Tuple{2,3},Float64,2,6}:
  2.16057   0.18857    0.563188
 -6.25674  -0.697974  -1.70902

julia> @time auto_sen(df, u0, tspan, p)
  8.660524 seconds (18.07 M allocations: 904.867 MiB, 6.13% gc time)
2×3 Array{Float64,2}:
  2.16057   0.188568   0.563195
 -6.25674  -0.697975  -1.70902

julia> @time diffeq_sen(df, u0, tspan, p)
  5.388473 seconds (12.41 M allocations: 637.880 MiB, 6.57% gc time)
3-element Array{Array{Float64,1},1}:
 [2.16056, -6.25677]
 [0.188568, -0.697976]
 [0.563185, -1.70902]

julia> @btime auto_sen($f, $u0s, $tspan, $sp)
  219.863 μs (1201 allocations: 89.85 KiB)
2×3 SArray{Tuple{2,3},Float64,2,6}:
  2.16057   0.18857    0.563188
 -6.25674  -0.697974  -1.70902

julia> @btime auto_sen($df, $u0, $tspan, $p)
  239.715 μs (836 allocations: 57.65 KiB)
2×3 Array{Float64,2}:
  2.16057   0.188568   0.563195
 -6.25674  -0.697975  -1.70902

julia> @btime diffeq_sen($df, $u0, $tspan, $p)
  500.736 μs (9635 allocations: 456.77 KiB)
3-element Array{Array{Float64,1},1}:
 [2.16056, -6.25677]
 [0.188568, -0.697976]
 [0.563185, -1.70902]

1. autodiff is much faster than the diffeq approach.

2. Using `SVector` doesn't give any significant run-time benefits, but
increases compile time drastically
======================================#

# Profile
#=
auto_sen(df, u0, tspan, p)
Profile.clear()
@profile for i in 1:500
    auto_sen(df, u0, tspan, p)
end
ProfileView.svgwrite("autodiff_profile.svg")

diffeq_sen(df, u0, tspan, p)
Profile.clear()
@profile for i in 1:500
    diffeq_sen(df, u0, tspan, p)
end
ProfileView.svgwrite("diffeq_profile.svg")
=#

# =============================================================== #
# Large regime (200x3 Jacobian matrix)
using LinearAlgebra, Test
function makebrusselator(N=10)
    xyd_brusselator = range(0,stop=1,length=N)
    function limit(a, N)
      if a == N+1
        return 1
      elseif a == 0
        return N
      else
        return a
      end
    end
    brusselator_f(x, y, t) = ifelse((((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) &&
                                    (t >= 1.1), 5., 0.)
    function brusselator_2d_loop(du, u, p, t)
        @inbounds begin
            A, B, α  = p
            xyd = xyd_brusselator
            dx = step(xyd)
            N = length(xyd)
            α = α/dx^2
            II = LinearIndices((N, N, 2))
            for I in CartesianIndices((N, N))
                x = xyd[I[1]]
                y = xyd[I[2]]
                i = I[1]
                j = I[2]
                ip1 = limit(i+1, N); im1 = limit(i-1, N)
                jp1 = limit(j+1, N); jm1 = limit(j-1, N)
                du[II[i,j,1]] = α*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]]) +
                    B + u[II[i,j,1]]^2*u[II[i,j,2]] - (A + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
            end
            for I in CartesianIndices((N, N))
              i = I[1]
              j = I[2]
              ip1 = limit(i+1, N)
              im1 = limit(i-1, N)
              jp1 = limit(j+1, N)
              jm1 = limit(j-1, N)
              du[II[i,j,2]] = α*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]]) +
                  A*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
            end
            return nothing
        end
    end
    function init_brusselator_2d(xyd)
        N = length(xyd)
        u = zeros(N, N, 2)
        for I in CartesianIndices((N, N))
            x = xyd[I[1]]
            y = xyd[I[2]]
            u[I,1] = 22*(y*(1-y))^(3/2)
            u[I,2] = 27*(x*(1-x))^(3/2)
        end
        vec(u)
    end
    brusselator_2d_loop, init_brusselator_2d(xyd_brusselator)
end
# Run low tolerance to test correctness
sol1 = @time auto_sen(makebrusselator(5)..., (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  8.943112 seconds (50.37 M allocations: 2.323 GiB, 9.23% gc time)
sol2 = @time diffeq_sen(makebrusselator(5)..., (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  13.934268 seconds (195.79 M allocations: 10.914 GiB, 16.79% gc time)
difference = copy(sol1)
for i in eachindex(sol2)
    difference[:, i] .-= sol2[i]
end
@test norm(difference) < 0.01

# High tolerance to benchmark
@time auto_sen(makebrusselator()..., (0.,10.), [3.4, 1., 10.])
# 28.316505 seconds (598.77 M allocations: 25.160 GiB, 16.81% gc time)
@time diffeq_sen(makebrusselator()..., (0.,10.), [3.4, 1., 10.])
# Cannot finish in my laptop
