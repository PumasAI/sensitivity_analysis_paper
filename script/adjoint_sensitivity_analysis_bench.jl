include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools, Profile, ProfileView, ParameterizedFunctions

df = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0];

t = 0:0.5:10

@time auto_sen_l2(df, u0, tspan, p, t, Vern9(), abstol=1e-5,reltol=1e-7)
#  28.793466 seconds (57.92 M allocations: 3.029 GiB, 7.07% gc time)
@time auto_sen_l2(df, u0, tspan, p, t, Vern9(); diffalg=ForwardDiff.gradient, abstol=1e-5,reltol=1e-7)
#  9.185815 seconds (25.21 M allocations: 1.364 GiB, 8.01% gc time)
@time diffeq_sen_l2(df, u0, tspan, p, t, Vern9(), abstol=1e-5,reltol=1e-7)
# 17.668040 seconds (50.34 M allocations: 3.835 GiB, 9.81% gc time)

# With Jacobian & paramjac
@btime diffeq_sen_l2($df, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7)
#   5.477 ms (84547 allocations: 2.05 MiB)
@btime auto_sen_l2($df, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7)
#   13.174 ms (271708 allocations: 10.07 MiB)
@btime auto_sen_l2($df, $u0, $tspan, $p, $t, $(Vern9()); diffalg=$(ForwardDiff.gradient), abstol=1e-5,reltol=1e-7)
#   89.042 μs (513 allocations: 72.96 KiB)

df_nojac = ODEFunction(df.f)
# Without analytical Jacobian
@btime diffeq_sen_l2($df_nojac, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7)
#   5.810 ms (84827 allocations: 2.08 MiB)
