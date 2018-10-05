# =============================================================== #
# Small regime (2x3 Jacobian matrix)

include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays#, Profile, ProfileView

include("lotka-volterra.jl")

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
lvcom_u0 = [u0...;zeros(6)]
comprob = ODEProblem(lvcom_df, lvcom_u0, tspan, p)
#=
f = function (u, p, t)
    a,b,c = p
    x, y = u
    dx = a*x - b*x*y
    dy = -c*y + x*y
    @SVector [dx, dy]
end

# Using `SVector` doesn't give any significant run-time benefits, but increases
# compile time drastically

u0s = @SVector [1.,1.]; sp = @SVector [1.5,1.0,3.0];
@time auto_sen(f, u0s, tspan, sp, Vern9(), abstol=1e-5,reltol=1e-7)
# 139.719343 seconds (22.66 M allocations: 1.558 GiB, 0.58% gc time)
@btime auto_sen($f, $u0s, $tspan, $sp, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 135.706 μs (499 allocations: 84.55 KiB)
=#

@time numerical_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 3.401622 seconds (7.42 M allocations: 387.500 MiB, 6.50% gc time) 
@time auto_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 13.564837 seconds (43.31 M allocations: 2.326 GiB, 9.15% gc time)
@time diffeq_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 5.712511 seconds (16.38 M allocations: 931.242 MiB, 9.13% gc time)
@time diffeq_sen(df_with_jacobian, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 2.679172 seconds (6.21 M allocations: 320.881 MiB, 5.36% gc time)
@time solve(comprob, Vern9(),abstol=1e-5,reltol=1e-7,save_everystep=false)
# 3.484515 seconds (8.10 M allocations: 417.261 MiB, 7.50% gc time)

@btime numerical_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 534.718 μs (2614 allocations: 222.88 KiB) 
@btime auto_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 99.404 μs (485 allocations: 57.11 KiB)
@btime diffeq_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 308.289 μs (8137 allocations: 391.78 KiB)
@btime diffeq_sen($df_with_jacobian, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 263.562 μs (8084 allocations: 389.08 KiB)
@btime solve($comprob, $(Vern9()),abstol=1e-5,reltol=1e-7,save_everystep=false)
# 36.517 μs (111 allocations: 14.67 KiB)

@btime numerical_sen($df, $u0, $tspan, $p, $(Vern9()))
# 472.643 μs (2585 allocations: 222.13 KiB)
@btime auto_sen($df, $u0, $tspan, $p, $(Vern9()))
# 90.698 μs (467 allocations: 56.95 KiB)
@btime diffeq_sen($df, $u0, $tspan, $p, $(Vern9()))
# 255.729 μs (5849 allocations: 292.16 KiB)
@btime diffeq_sen($df_with_jacobian, $u0, $tspan, $p, $(Vern9()))
# 192.806 μs (5619 allocations: 273.50 KiB)
@btime solve($comprob, $(Vern9()),save_everystep=false)
# 27.937 μs (112 allocations: 14.66 KiB)

# =============================================================== #
# Large regime (128x3 Jacobian matrix)
using LinearAlgebra, Test
include("brusselator.jl")

bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(5)
# Run low tolerance to test correctness
sol1 = @time numerical_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
# 9.300622 seconds (157.90 M allocations: 3.140 GiB, 8.82% gc time)
sol2 = @time auto_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  8.943112 seconds (50.37 M allocations: 2.323 GiB, 9.23% gc time)
sol3 = @time diffeq_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  13.934268 seconds (195.79 M allocations: 10.914 GiB, 16.79% gc time)
sol4 = @time diffeq_sen(ODEFunction(bfun, jac=brusselator_jac), b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  9.747963 seconds (175.60 M allocations: 10.206 GiB, 20.70% gc time)
sol5 = @time solve(brusselator_comp, Tsit5(), abstol=1e-5,reltol=1e-7,save_everystep=false)
#  3.850392 seconds (36.08 M allocations: 941.787 MiB, 7.35% gc time)

@btime numerical_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), abstol=1e-5,reltol=1e-7);
# 6.883 s (153423891 allocations: 2.92 GiB)
@btime auto_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), abstol=1e-5,reltol=1e-7);
#   1.159 s (25611160 allocations: 1.07 GiB)
@btime diffeq_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), abstol=1e-5,reltol=1e-7);
#   10.091 s (178932172 allocations: 10.27 GiB)
@btime diffeq_sen($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $((0.,10.)), $([3.4, 1., 10.]), abstol=1e-5,reltol=1e-7);
#   1.975 s (51638028 allocations: 1.20 GiB)
@btime solve($brusselator_comp, $(Tsit5()), abstol=1e-5,reltol=1e-7,save_everystep=false);
#   1.171 s (29331219 allocations: 593.95 MiB)

difference1 = copy(sol2)
difference2 = copy(sol2)
difference3 = vec(sol2) .- vec(sol5[2][5*5*2+1:end])
for i in eachindex(sol3)
    difference1[:, i] .-= sol3[i]
    difference2[:, i] .-= sol4[i]
end
@test norm(difference1) < 0.01 && norm(difference2) < 0.01 && norm(difference3) < 0.01

# # High tolerance to benchmark
bfun_n, b_u0_n, brusselator_jacn, b_comp = makebrusselator(8)
@time numerical_sen(bfun_n, b_u0_n, (0.,10.), [3.4, 1., 10.])
# 57.474043 seconds (1.32 G allocations: 25.101 GiB, 9.46% gc time)
@time auto_sen(bfun_n, b_u0_n, (0.,10.), [3.4, 1., 10.])
#  13.632362 seconds (238.33 M allocations: 10.063 GiB, 15.94% gc time)
@time diffeq_sen(bfun_n, b_u0_n, (0.,10.), [3.4, 1., 10.])
# 302.428220 seconds (3.42 G allocations: 216.285 GiB, 12.05% gc time)
@time diffeq_sen(ODEFunction(bfun_n, jac=brusselator_jacn), b_u0_n, (0.,10.), [3.4, 1., 10.])
#  36.712953 seconds (442.08 M allocations: 10.215 GiB, 6.01% gc time)
@time solve(b_comp, Tsit5(), abstol=1e-5,reltol=1e-7,save_everystep=false)
#  12.807239 seconds (249.44 M allocations: 4.843 GiB, 10.15% gc time)

# 20×25 Jacobian
include("pollution.jl")
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

pprob, pprob_jac = make_pollution()
@btime auto_sen($(pprob.f), $(pprob.u0), $(pprob.tspan), $(pprob.p), $(Rodas5()),abstol=1e-5,reltol=1e-7)
#   7.269 ms (5802 allocations: 750.31 KiB)
@btime diffeq_sen($(pprob.f.f), $(pprob.u0), $(pprob.tspan), $(pprob.p), $(Rodas5(autodiff=false)),abstol=1e-5,reltol=1e-7)
#   906.562 ms (3724524 allocations: 174.96 MiB)
@btime diffeq_sen($(pprob_jac.f), $(pprob_jac.u0), $(pprob_jac.tspan), $(pprob_jac.p), $(Rodas5(autodiff=false)),abstol=1e-5,reltol=1e-7)
#  830.873 ms (3724295 allocations: 174.94 MiB)
# TODO: complile time sensitivity analysis
