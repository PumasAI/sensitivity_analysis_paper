include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools#, Profile, ProfileView
using Test
include("lotka-volterra.jl")

lvu0 = [1.,1.]; lvtspan = (-0.01, 10.01); lvp = [1.5,1.0,3.0];

lvt = 0:0.5:10

lsol1 = @btime auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()); diffalg=$(ForwardDiff.gradient));
lsol2 = @btime auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()); diffalg=$(ReverseDiff.gradient));
lsol3 = @btime diffeq_sen_l2($lvdf_with_jacobian, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()));
lsol4 = @btime diffeq_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()), abstol=1e-5,reltol=1e-7;
                             sensalg=$(SensitivityAlg(autojacvec=false)));
lsol5 = @btime diffeq_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()), abstol=1e-5,reltol=1e-7;
                             sensalg=$(SensitivityAlg(autojacvec=true))); # with seeding
lsol6 = @btime numerical_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()), abstol=1e-5,reltol=1e-7);
@test maximum(abs, lsol1 .- lsol2)/maximum(abs,  lsol1) < 0.2
@test maximum(abs, lsol1 .- lsol3')/maximum(abs, lsol1) < 0.2
@test maximum(abs, lsol1 .- lsol4')/maximum(abs, lsol1) < 0.2
@test maximum(abs, lsol1 .- lsol5')/maximum(abs, lsol1) < 0.2
@test maximum(abs, lsol1 .- lsol6)/maximum(abs, lsol1) < 0.2
#=
julia> lsol1 = @btime auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()); diffalg=$(ForwardDiff.gradient));
  133.855 μs (824 allocations: 97.41 KiB)

julia> lsol2 = @btime auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()); diffalg=$(ReverseDiff.gradient));
  9.488 ms (222039 allocations: 7.95 MiB)

julia> lsol3 = @btime diffeq_sen_l2($lvdf_with_jacobian, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()));
  4.544 ms (92686 allocations: 2.42 MiB)

julia> lsol4 = @btime diffeq_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()), abstol=1e-5,reltol=1e-7;
                                    sensalg=$(SensitivityAlg(autojacvec=false)));
  4.730 ms (91094 allocations: 2.41 MiB)

julia> lsol5 = @btime diffeq_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Vern9()), abstol=1e-5,reltol=1e-7;
                                    sensalg=$(SensitivityAlg(autojacvec=true))); # with seeding
  9.388 ms (179860 allocations: 5.86 MiB)
=#

include("brusselator.jl")
bt = 0:0.5:10
tspan = (-0.01, 10.01)
n = 5
bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')

bsol1 = @btime auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()), diffalg=$(ForwardDiff.gradient), save_everystep=false);
bsol2 = @btime auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)), diffalg=$(ReverseDiff.gradient), save_everystep=false);
bsol3 = @btime diffeq_sen_l2($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)), save_everystep=false);
bsol4 = @btime diffeq_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
bsol5 = @btime diffeq_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true));
bsol6 = @btime numerical_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()), save_everystep=false);
using Test
@test maximum(abs, bsol1 .- bsol2)/maximum(abs,  bsol1) < 1e-2
@test maximum(abs, bsol1 .- bsol3')/maximum(abs, bsol1) < 1e-2
@test maximum(abs, bsol1 .- bsol4')/maximum(abs, bsol1) < 1e-2
@test maximum(abs, bsol1 .- bsol5')/maximum(abs, bsol1) < 1e-2
#=
julia> bsol1 = @btime auto_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5()), diffalg=$(ForwardDiff.gradient), save_everystep=false);
  16.678 ms (2377 allocations: 389.45 KiB)

julia> bsol2 = @btime auto_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), diffalg=$(ReverseDiff.gradient), save_everystep=false);
  11.851 s (85025440 allocations: 2.90 GiB)

julia> bsol3 = @btime diffeq_sen_l2($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), save_everystep=false);
  512.618 ms (3105112 allocations: 118.30 MiB)

julia> bsol4 = @btime diffeq_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
  1.068 s (2617960 allocations: 81.97 MiB)

julia> bsol5 = @btime diffeq_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true));
  13.875 s (108723076 allocations: 4.03 GiB)
=#

include("pollution.jl")
using BenchmarkTools, LinearAlgebra
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

pcomp, pu0, pp, pcompu0 = make_pollution();
ptspan = (-0.01, 60.01)
pts = 0:0.5:60
psol1 = @btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)), diffalg=$(ForwardDiff.gradient));
psol2 = @btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)), diffalg=$(ReverseDiff.gradient));
psol3 = @btime diffeq_sen_l2($(ODEFunction(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)));
psol4 = @btime diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)),
                             sensalg=$(SensitivityAlg(autojacvec=false)));
psol5 = @btime diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)),
                             sensalg=(SensitivityAlg(autojacvec=true)));
psol6 = @btime numerical_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)));
@test maximum(abs, psol1 .- psol2)/maximum(abs,  psol1) < 1e-2
@test maximum(abs, psol1 .- psol3')/maximum(abs, psol1) < 1e-2
@test maximum(abs, psol1 .- psol4')/maximum(abs, psol1) < 1e-2
@test maximum(abs, psol1 .- psol5')/maximum(abs, psol1) < 1e-2
@test maximum(abs, psol1 .- psol6)/maximum(abs, psol1) < 1e-2
#=
julia> psol1 = @btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)), diffalg=$(ForwardDiff.gradient));
  6.652 ms (4658 allocations: 1.16 MiB)

julia> psol2 = @btime auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)), diffalg=$(ReverseDiff.gradient));
  800.144 ms (4735749 allocations: 167.20 MiB)

julia> psol3 = @btime diffeq_sen_l2($(ODEFunction(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)));
  3.132 s (27878798 allocations: 762.70 MiB)

julia> psol4 = @btime diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)),
                                    sensalg=$(SensitivityAlg(autojacvec=false)));
  4.763 s (27872094 allocations: 762.60 MiB)

julia> psol5 = @btime diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)),
                                    sensalg=(SensitivityAlg(autojacvec=true)));
  70.707 s (549603371 allocations: 19.62 GiB)
=#
