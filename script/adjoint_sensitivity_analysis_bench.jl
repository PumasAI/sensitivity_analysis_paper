include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools#, Profile, ProfileView
using ParameterizedFunctions

lvdf = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

u0 = [1.,1.]; tspan = (-0.1, 10.1); p = [1.5,1.0,3.0];

t = 0:0.5:10
lvdf_nojac = ODEFunction(lvdf.f)

@btime auto_sen_l2($lvdf, $u0, $tspan, $p, $t, $(Vern9()); diffalg=$(ForwardDiff.gradient));
@btime auto_sen_l2($lvdf, $u0, $tspan, $p, $t, $(Vern9()); diffalg=$(ReverseDiff.gradient));
@btime diffeq_sen_l2($lvdf, $u0, $tspan, $p, $t, $(Vern9()));
@btime diffeq_sen_l2($lvdf_nojac, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7;
                     sensalg=SensitivityAlg(autojacvec=false));
@btime diffeq_sen_l2($lvdf_nojac, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7;
                     sensalg=SensitivityAlg(autojacvec=true)); # with seeding
#=
julia> @btime auto_sen_l2($lvdf, $u0, $tspan, $p, $t, $(Vern9()); diffalg=$(ForwardDiff.gradient));
  75.715 μs (502 allocations: 74.30 KiB)

julia> @btime auto_sen_l2($lvdf, $u0, $tspan, $p, $t, $(Vern9()); diffalg=$(ReverseDiff.gradient));
  10.208 ms (229816 allocations: 8.20 MiB)

julia> @btime diffeq_sen_l2($lvdf, $u0, $tspan, $p, $t, $(Vern9()));
  3.776 ms (75493 allocations: 1.78 MiB)

julia> @btime diffeq_sen_l2($lvdf_nojac, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7;
                            sensalg=SensitivityAlg(autojacvec=false));
  4.872 ms (93105 allocations: 2.43 MiB)

julia> @btime diffeq_sen_l2($lvdf_nojac, $u0, $tspan, $p, $t, $(Vern9()), abstol=1e-5,reltol=1e-7;
                            sensalg=SensitivityAlg(autojacvec=true));
  9.917 ms (192929 allocations: 6.35 MiB)
=#

include("brusselator.jl")
bp = [3.4, 1., 10.]
bt = 0:0.5:10
tspan = (-0.01, 10.01)
bfun, b_u0, brusselator_jac, _ = makebrusselator(5)
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')

bsol1 = @btime auto_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5()), diffalg=$(ForwardDiff.gradient), save_everystep=false);
bsol2 = @btime auto_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), diffalg=$(ReverseDiff.gradient), save_everystep=false);
bsol3 = @btime diffeq_sen_l2($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), save_everystep=false);
bsol4 = @btime diffeq_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
bsol5 = @btime diffeq_sen_l2($bfun, $b_u0, $tspan, $bp, $bt, $(Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true));
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
@test maximum(abs, psol1 .- psol2)/maximum(abs,  psol1) < 1e-2
@test maximum(abs, psol1 .- psol3')/maximum(abs, psol1) < 1e-2
@test maximum(abs, psol1 .- psol4')/maximum(abs, psol1) < 1e-2
@test maximum(abs, psol1 .- psol5')/maximum(abs, psol1) < 1e-2
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
