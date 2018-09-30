include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools, Profile, ProfileView, ParameterizedFunctions

df = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0];

t = 0:0.5:10

# TODO: Need to re-run
@time auto_sen_l2(df, u0, tspan, p, t, Vern9(), abstol=1e-5,reltol=1e-7)
# 34.347232 seconds (57.73 M allocations: 3.018 GiB, 6.38% gc time)
@time auto_sen_l2(df, u0, tspan, p, t, Vern9(); diffalg=ForwardDiff.gradient, abstol=1e-5,reltol=1e-7)
# 11.576240 seconds (25.06 M allocations: 1.357 GiB, 9.43% gc time)
@time diffeq_sen_l2(df, u0, tspan, p, t, Vern9(), abstol=1e-5,reltol=1e-7)
# 20.523481 seconds (48.88 M allocations: 3.765 GiB, 8.79% gc time)

@btime diffeq_sen_l2($df, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 7.146 ms (87366 allocations: 2.12 MiB)
@btime auto_sen_l2($df, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 14.232 ms (271707 allocations: 10.07 MiB)
@btime auto_sen_l2($df, $u0, $tspan, $p, $t, $(Vern9()); diffalg=$(ForwardDiff.gradient), abstol=1e-5,reltol=1e-7)
# 116.052 μs (513 allocations: 72.96 KiB)
