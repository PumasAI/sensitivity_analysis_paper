include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools#, Profile, ProfileView
using ParameterizedFunctions

df = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

u0 = [1.,1.]; tspan = (-0.1, 10.1); p = [1.5,1.0,3.0];

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

include("brusselator.jl")
bp = [3.4, 1., 10.]
bt = 0:0.5:10
tspan = (0., 10.)
bfun, b_u0, brusselator_jac, _ = makebrusselator(5)

@btime auto_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Tsit5()), abstol=1e-5,reltol=1e-7, save_everystep=false) # reverse mode
#   29.925 s (162941580 allocations: 5.44 GiB)
#=
3253.997828166661
-10120.945026662224
-0.07716107201482299
=#
@btime auto_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Tsit5()), abstol=1e-5,reltol=1e-7, diffalg=$(ForwardDiff.gradient), save_everystep=false)
#   1.291 s (25611301 allocations: 1.07 GiB)
#=
   3253.9979588435704
 -10120.945328043885
     -0.06790648237049532
=#
@btime diffeq_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Tsit5()), abstol=1e-5,reltol=1e-7, save_everystep=false)
# 9.239 s (154560416 allocations: 9.64 GiB)
#=
   3313.467995615832
 -10250.292894948318
    -10.312088281648458
=#
@btime diffeq_sen_l2($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $tspan, $bp, $bt, $(Tsit5()), abstol=1e-5,reltol=1e-7, save_everystep=false)
#  964.180 ms (27009683 allocations: 563.93 MiB)
#=
   3313.467995615854
 -10250.292894948521
    -10.312088281649485
=#

include("pollution.jl")
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')
using BenchmarkTools, LinearAlgebra
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

pcomp, pu0, pp, pcompu0 = make_pollution(pollution)
pts = 0:0.5:2
@btime diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $(-0.1,2.1), $pp, $pts, $(Rodas5(autodiff=false)), sensalg=$(SensitivityAlg()))
# 2.593 s (19837898 allocations: 725.16 MiB)
@btime diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $(-0.1,2.1), $pp, $pts, $(Rodas5(autodiff=false)), sensalg=$(SensitivityAlg(autojacvec=false)))
# 2.267 s (19837898 allocations: 725.16 MiB)
@btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $(-0.1,2.1), $pp, $pts, $(Rodas5(autodiff=false)))
# 351.652 ms (3063903 allocations: 111.37 MiB)
@btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $(-0.1,2.1), $pp, $pts, $(Rodas5(autodiff=false)), diffalg=ForwardDiff.gradient)
# 3.605 ms (1917 allocations: 453.20 KiB)
