# =============================================================== #
# Small regime (2x3 Jacobian matrix)

using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays, Profile, ProfileView

function f(u, p, t)
    a,b,c = p
    x, y = u
    dx = a*x - b*x*y
    dy = -c*y + x*y
    @SVector [dx, dy]
end

function df(du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

u0s = @SVector [1.,1.]; u0 = [1.,1.]; tspan = (0., 10.); sp = @SVector [1.5,1.0,3.0]; p = [1.5,1.0,3.0]

function diffeq_sen(f, init, tspan, p)
    prob = ODELocalSensitivityProblem(f,init,tspan,p)
    sol = solve(prob,Vern6(),save_everystep=false,abstol=1e-5,reltol=1e-7)
    extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, init, tspan, p)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),eltype(p).(tspan),p)
        solve(prob,Vern6(),save_everystep=false,abstol=1e-5,reltol=1e-7)[end]
    end
    ForwardDiff.jacobian(test_f, p)
end
@time auto_sen(f, u0s, tspan, sp)
@time auto_sen(df, u0, tspan, p)
@time diffeq_sen(df, u0, tspan, p)
@btime auto_sen($f, $u0s, $tspan, $sp)
@btime auto_sen($df, $u0, $tspan, $p)
@btime diffeq_sen($df, $u0, $tspan, $p)

function com_df(du, u, p, t)
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

    #du[1+1] = -(b*s2*x) + s1*(a - b*y) + x
    #du[1+2] = s2*(-c + x) + s1*y       + 0
    #du[1+3] = -(b*s4*x) + s3*(a - b*y) - x*y
    #du[1+4] = s4*(-c + x) + s3*y       + 0
    #du[1+5] = -(b*s6*x) + s5*(a - b*y) + 0
    #du[1+6] = s6*(-c + x) + s5*y       - y
    nothing
end
com_u0 = [u0...;zeros(6)]
comprob = ODEProblem(com_df, com_u0, tspan, p)
@time solve(comprob, Vern6(),save_everystep=false,abstol=1e-5,reltol=1e-7)
@btime solve($comprob, $(Vern6()),save_everystep=false,abstol=1e-5,reltol=1e-7)
#=====================================
julia> @time solve(comprob, Vern6(),save_everystep=false,abstol=1e-5,reltol=1e-7)
  2.678311 seconds (5.41 M allocations: 271.934 MiB, 8.16% gc time)
retcode: Success
Interpolation: 1st order linear
t: 2-element Array{Float64,1}:
  0.0
 10.0
u: 2-element Array{Array{Float64,1},1}:
 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [1.02635, 0.909691, 2.16056, -6.25677, 0.188568, -0.697976, 0.563185, -1.70902]

julia> @btime solve($comprob, $(Vern6()),save_everystep=false,abstol=1e-5,reltol=1e-7)
  52.278 μs (103 allocations: 9.13 KiB)
retcode: Success
Interpolation: 1st order linear
t: 2-element Array{Float64,1}:
  0.0
 10.0
u: 2-element Array{Array{Float64,1},1}:
 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [1.02635, 0.909691, 2.16056, -6.25677, 0.188568, -0.697976, 0.563185, -1.70902]
======================================#

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
# Large regime (100x100 Jacobian matrix)

using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays, Profile, ProfileView
using LinearAlgebra

const D = Tridiagonal(rand(99), rand(100), rand(99))
function df(du, u, p, t)
    mul!(du, D, u)
    @. du += p
end
u0 = rand(100); tspan = (0., 0.4); p = rand(100);

function diffeq_sen(f, init, tspan, p, alg=Vern6())
    prob = ODELocalSensitivityProblem(f,init,tspan,p)
    sol = solve(prob,alg,save_everystep=false,abstol=1e-5,reltol=1e-7)
    extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, init, tspan, p, alg=Vern6())
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),eltype(p).(tspan),p)
        solve(prob,alg,save_everystep=false,abstol=1e-5,reltol=1e-7)[end]
    end
    ForwardDiff.jacobian(test_f, p)
end
DiffEqBase.has_syms(::DiffEqSensitivity.ODELocalSensitvityFunction) = false
DiffEqBase.has_tgrad(::DiffEqSensitivity.ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::DiffEqSensitivity.ODELocalSensitvityFunction) = false
@time auto_sen(df, u0, tspan, p, Rodas5(autodiff=false))[1:3]
@time diffeq_sen(df, u0, tspan, p, Rodas5(autodiff=false))[1][1:3]
@time auto_sen(df, u0, tspan, p, Rodas5(autodiff=false));
@time diffeq_sen(df, u0, tspan, p, Rodas5(autodiff=false));

@time auto_sen(df, u0, tspan, p)[1:3]
@time diffeq_sen(df, u0, tspan, p)[1][1:3]
@btime auto_sen($df, $u0, $tspan, $p);
@btime diffeq_sen($df, $u0, $tspan, $p);
#==========================================
julia> @time auto_sen(df, u0, tspan, p, Rodas5(autodiff=false))[1:3]
 12.322556 seconds (38.20 M allocations: 2.334 GiB, 7.86% gc time)
3-element Array{Float64,1}:
 0.4937800270096975
 0.0652016945329739
 0.006998118349731402

julia> @time diffeq_sen(df, u0, tspan, p, Rodas5(autodiff=false))[1][1:3]
 98.789351 seconds (41.17 M allocations: 3.418 GiB, 0.81% gc time)
3-element Array{Float64,1}:
 0.49378002795271414
 0.06520169515500866
 0.006998119064051696

julia> @time auto_sen(df, u0, tspan, p, Rodas5(autodiff=false));
  0.191787 seconds (7.57 k allocations: 20.058 MiB, 3.20% gc time)

julia> @time diffeq_sen(df, u0, tspan, p, Rodas5(autodiff=false));
106.350115 seconds (24.42 M allocations: 2.614 GiB, 0.49% gc time)

# TODO: investigate allocations
==========================================#
