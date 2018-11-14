# =============================================================== #
# Small regime (2x3 Jacobian matrix)

include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays#, Profile, ProfileView

include("lotka-volterra.jl")

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
lvcom_u0 = [u0...;zeros(6)]
comprob = ODEProblem(lvcom_df, lvcom_u0, tspan, p)

@btime solve($comprob, $(Vern9()),save_everystep=false);
@btime auto_sen($lvdf, $u0, $tspan, $p, $(Vern9()));
@btime diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Vern9()));
@btime diffeq_sen($lvdf, $u0, $tspan, $p, $(Vern9()), sensalg=SensitivityAlg(autojacvec=false));
@btime diffeq_sen($lvdf, $u0, $tspan, $p, $(Vern9()));
@btime numerical_sen($lvdf, $u0, $tspan, $p, $(Vern9()));
#=
julia> @btime solve($comprob, $(Vern9()),save_everystep=false);
  29.779 μs (106 allocations: 14.53 KiB)

julia> @btime auto_sen($lvdf, $u0, $tspan, $p, $(Vern9()));
  85.941 μs (461 allocations: 56.17 KiB)

julia> @btime diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Vern9()));
  188.735 μs (5615 allocations: 273.42 KiB)

julia> @btime diffeq_sen($lvdf, $u0, $tspan, $p, $(Vern9()), sensalg=SensitivityAlg(autojacvec=false));
  260.237 μs (5843 allocations: 291.42 KiB)

julia> @btime diffeq_sen($lvdf, $u0, $tspan, $p, $(Vern9()));
  212.115 μs (5839 allocations: 291.19 KiB)

julia> @btime numerical_sen($lvdf, $u0, $tspan, $p, $(Vern9()));
  419.034 μs (2579 allocations: 218.19 KiB)
=#
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

#=
@time numerical_sen(lvdf, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 3.401622 seconds (7.42 M allocations: 387.500 MiB, 6.50% gc time)
@time auto_sen(lvdf, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 13.564837 seconds (43.31 M allocations: 2.326 GiB, 9.15% gc time)
@time diffeq_sen(lvdf, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 5.712511 seconds (16.38 M allocations: 931.242 MiB, 9.13% gc time)
# with seeding 10.179159 seconds (32.43 M allocations: 1.730 GiB, 9.75% gc time)
@time diffeq_sen(lvdf_with_jacobian, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 2.679172 seconds (6.21 M allocations: 320.881 MiB, 5.36% gc time)
@time solve(comprob, Vern9(),abstol=1e-5,reltol=1e-7,save_everystep=false)
# 3.484515 seconds (8.10 M allocations: 417.261 MiB, 7.50% gc time)

@btime numerical_sen($lvdf, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 534.718 μs (2614 allocations: 222.88 KiB)
@btime auto_sen($lvdf, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 99.404 μs (485 allocations: 57.11 KiB)
@btime diffeq_sen($lvdf, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 308.289 μs (8137 allocations: 391.78 KiB)
# with seeding 268.012 μs (8310 allocations: 406.86 KiB)
@btime diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 263.562 μs (8084 allocations: 389.08 KiB)
@btime solve($comprob, $(Vern9()),abstol=1e-5,reltol=1e-7,save_everystep=false)
# 36.517 μs (111 allocations: 14.67 KiB)
=#

# =============================================================== #
# Large regime (128x3 Jacobian matrix)
using LinearAlgebra, Test
include("brusselator.jl")
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(5)
# Run low tolerance to test correctness
sol1 = @time numerical_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], Rodas5(), abstol=1e-5,reltol=1e-7);
#  4.086760 seconds (11.94 M allocations: 615.768 MiB, 8.97% gc time)
sol2 = @time auto_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], Rodas5(), abstol=1e-5,reltol=1e-7);
#  6.404542 seconds (18.41 M allocations: 862.547 MiB, 9.09% gc time)
sol3 = @time diffeq_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
#  3.390236 seconds (5.19 M allocations: 253.180 MiB, 4.74% gc time)
sol4 = @time diffeq_sen(ODEFunction(bfun, jac=brusselator_jac), b_u0, (0.,10.), [3.4, 1., 10.], Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
#  3.018417 seconds (6.22 M allocations: 339.178 MiB, 4.57% gc time)
sol5 = @time solve(brusselator_comp, Rodas5(), abstol=1e-5,reltol=1e-7,save_everystep=false);
#  2.699624 seconds (6.05 M allocations: 347.092 MiB, 4.42% gc time)

difference1 = copy(sol2)
difference2 = copy(sol2)
difference3 = vec(sol2) .- vec(sol5[2][5*5*2+1:end])
for i in eachindex(sol3)
    difference1[:, i] .-= sol3[i]
    difference2[:, i] .-= sol4[i]
end
@test norm(difference1) < 0.01 && norm(difference2) < 0.01 && norm(difference3) < 0.01 && norm(sol5[end][51:end] .- vec(sol1)) < 0.01

# High tolerance to benchmark
bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(8)
@btime solve($brusselator_comp, $(Rodas5(autodiff=false)), save_everystep=false);
@btime auto_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5()));
@btime diffeq_sen($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5(autodiff=false)));
@btime diffeq_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5(autodiff=false)), sensalg=SensitivityAlg(autojacvec=false));
@btime diffeq_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5(autodiff=false)));
@btime numerical_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5()));
#=
julia> @btime solve($brusselator_comp, $(Rodas5(autodiff=false)), save_everystep=false);
  3.920 s (2551697 allocations: 276.98 MiB)

julia> @btime auto_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5()));
  182.126 ms (2140 allocations: 1.36 MiB)

julia> @btime diffeq_sen($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5(autodiff=false)));
  4.063 s (2551726 allocations: 228.80 MiB)

julia> @btime diffeq_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5(autodiff=false)), sensalg=SensitivityAlg(autojacvec=false));
  19.600 s (968372 allocations: 48.96 MiB)

julia> @btime diffeq_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5(autodiff=false)));
  1.992 s (968370 allocations: 48.81 MiB)

julia> @btime numerical_sen($bfun, $b_u0, $((0.,10.)), $([3.4, 1., 10.]), $(Rodas5()));
  582.517 ms (4628 allocations: 2.50 MiB)
=#

# 20×25 Jacobian
include("pollution.jl")
pcomp, pu0, pp, pcompu0 = make_pollution(pollution)
using BenchmarkTools
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

ptspan = (0.,60.)
@btime solve($(ODEProblem(pcomp, pcompu0, ptspan, (pp, zeros(20, 20), zeros(20, 25), zeros(20,25), zeros(20,25)))),
             $(Rodas5(autodiff=false)),save_everystep=false);
@btime auto_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()));
@btime diffeq_sen($(ODEFunction(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)));
@btime diffeq_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)),sensalg=SensitivityAlg(autojacvec=false));
@btime diffeq_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)));
@btime numerical_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()));
#=
julia> @btime solve($(ODEProblem(pcomp, pcompu0, ptspan, (pp, zeros(20, 20), zeros(20, 25), zeros(20,25), zeros(20,25)))),
                    $(Rodas5(autodiff=false)),save_everystep=false);
  427.458 ms (81494 allocations: 8.73 MiB)

julia> @btime auto_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()));
  8.913 ms (3116 allocations: 610.72 KiB)

julia> @btime diffeq_sen($(ODEFunction(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)));
  721.244 ms (3075753 allocations: 145.22 MiB)

julia> @btime diffeq_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)),sensalg=SensitivityAlg(autojacvec=false));
  836.049 ms (3075757 allocations: 145.22 MiB)

julia> @btime diffeq_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)));
  718.607 ms (3075755 allocations: 145.21 MiB)

julia> @btime numerical_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()));
  39.125 ms (15708 allocations: 1.64 MiB)
=#

include("pkpd.jl")
using Test
sol1 = solve(pkpdcompprob, Vern9(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=1:2:49,save_everystep=false)[end][6:end]
sol2 = vec(auto_sen(pkpdprob, Vern9(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=1:2:49))
sol3 = vec(hcat(diffeq_sen(pkpdprob, Vern9(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=1:2:49)...))
@test sol1 ≈ sol2 atol=1e-6
@test sol2 ≈ sol3 atol=1e-6
@btime solve($pkpdcompprob, $(Vern9()),callback=$pkpdcb,tstops=1:2:49,save_everystep=false);
@btime auto_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
@btime diffeq_sen($(ODEFunction(pkpdf.f, jac=pkpdf.jac)), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
@btime diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49,
                  sensalg=SensitivityAlg(autojacvec=false));
@btime diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
@btime numerical_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
#=
julia> @btime solve($pkpdcompprob, $(Vern9()),callback=$pkpdcb,tstops=1:2:49,save_everystep=false);
  7.826 ms (26114 allocations: 1.11 MiB)

julia> @btime auto_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
  4.235 ms (10704 allocations: 614.00 KiB)

julia> @btime diffeq_sen($(ODEFunction(pkpdf.f, jac=pkpdf.jac)), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
  9.502 ms (142587 allocations: 6.59 MiB)

julia> @btime diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49,
                         sensalg=SensitivityAlg(autojacvec=false));
  9.935 ms (142815 allocations: 6.61 MiB)

julia> @btime diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
  20.042 ms (142807 allocations: 6.61 MiB)

julia> @btime numerical_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Vern9()),callback=$pkpdcb,tstops=1:2:49);
  15.921 ms (15945 allocations: 1.35 MiB)
=#
