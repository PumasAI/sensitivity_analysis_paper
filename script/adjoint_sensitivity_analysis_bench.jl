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
pts = 0:0.5:60
@time diffeq_sen_l2(ODEFunction(pollution.f), pu0, (-0.1,60.1), pp, pts, Rodas5(autodiff=false),
                     sensalg=SensitivityAlg())
# 75.165828 seconds (548.84 M allocations: 19.589 GiB, 10.73% gc time)
# -1.21909  0.0163272  4.88067e-10  -167.566  77.4104  4.4199e-6  -45.7476  …  -9.66059e-7  0.000415794  -0.000841743  0.156532  2.24244e-6  -0.00126731
@time diffeq_sen_l2(ODEFunction(pollution.f), pu0, (-0.1,60.1), pp, pts, Rodas5(autodiff=false),
                     sensalg=SensitivityAlg(autojacvec=false)) # TODO: find out the reason which it has the exact same number of allocations, maybe I messed up?
# 3.842465 seconds (27.84 M allocations: 762.102 MiB, 5.54% gc time)
# 1×25 Adjoint{Float64,Array{Float64,1}}:
#  -1.21908  0.0163272  4.88082e-10  -167.566  77.4104  4.4199e-6  -45.7476  …  -9.66059e-7  0.000415794  -0.000841743  0.156532  2.24244e-6  -0.00126731
@btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $(-0.1,60.1), $pp, $pts, $(Rodas5(autodiff=false)))
# 255.377 ms (4735747 allocations: 167.20 MiB)
#    -1.2204908070947482
#     0.01663065506456956
#    -1.5796224637461386e-9
#  -168.85949034944915
#    77.68507737751857
#     4.443640629135168e-6
#   -45.950810915251004
#    -7.5723830201792884e-6
#    -2.3389664100300814e-6
#     4.287677550981215e-6
#    -0.614306543296696
#    -1.6286046860046427e-10
#    -0.0005404691561648167
#     7.132429401840044e-6
#     7.385110397256353e-14
#    -0.0007954694689778228
#    -9.432291836475315e-8
#    -2.7831861639726452e-15
#     6.268437313429756e-19
#    -9.700931271097962e-7
#     0.00041628469961206246
#    -0.0008421086549872315
#     0.1564411116382127
#     2.2430527273057995e-6
#    -0.0012674492857014034

@btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $(-0.1,60.1), $pp, $pts, $(Rodas5(autodiff=false)),
                   diffalg=ForwardDiff.gradient)
# 6.524 ms (4656 allocations: 1.16 MiB)
#   -1.2204891916144742
#    0.01663060212953999
#   -1.5983905312252315e-9
# -168.8587889224989
#   77.6850441897015
#    4.443586172503929e-6
#  -45.95072600694094
#   -7.572344828012601e-6
#   -2.338958091980496e-6
#    4.2876607638262905e-6
#   -0.6143198990081108
#   -1.6284880053274313e-10
#   -0.000540441997276013
#    7.132414913317529e-6
#    7.385088915972147e-14
#   -0.0007955542566227372
#   -9.432503569389826e-8
#   -2.7834828464346823e-15
#    6.269105517990765e-19
#   -9.700906320162897e-7
#    0.00041629349554099423
#   -0.0008421014941126205
#    0.15644023032995322
#    2.2430196399649795e-6
#   -0.0012674293032663478
