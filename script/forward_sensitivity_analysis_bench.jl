# =============================================================== #
# Small regime (2x3 Jacobian matrix)

include("sensitivity.jl")
DiffEqBase.has_tgrad(::ODELocalSensitivityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitivityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitivityFunction) = false

using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays#, Profile, ProfileView
using LinearAlgebra, Test

forward_lv = let
  include("lotka-volterra.jl")
  @info "Running the Lotka-Volterra model:"
  u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]; lvcom_u0 = [u0...;zeros(6)]
  comprob = ODEProblem(lvcom_df, lvcom_u0, tspan, p)
  @info "  Running compile-time CSA"
  t1 = @belapsed solve($comprob, $(Tsit5()),)
  @info "  Running DSA"
  t2 = @belapsed auto_sen($lvdf, $u0, $tspan, $p, $(Tsit5()))
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Tsit5()))
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5()), sensalg=SensitivityAlg(autojacvec=false))
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5()), sensalg=SensitivityAlg(autojacvec=true))
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($lvdf, $u0, $tspan, $p, $(Tsit5()))
  print('\n')
  [t1, t2, t3, t4, t5, t6]
end

# 18x36 Jacobian matrix
forward_bruss = let
  include("brusselator.jl")
  @info "Running the Brusselator model:"
  n = 5
  # Run low tolerance to test correctness
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  sol1 = @time numerical_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(), abstol=1e-5,reltol=1e-7);
  sol2 = @time auto_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(), abstol=1e-5,reltol=1e-7);
  @test sol1 ≈ sol2 atol=1e-2
  sol3 = @time diffeq_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
  @test sol1 ≈ hcat(sol3...) atol=1e-3
  sol4 = @time diffeq_sen(ODEFunction(bfun, jac=brusselator_jac), b_u0, (0.,10.), b_p, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
  @test sol1 ≈ hcat(sol4...) atol=1e-3
  sol5 = @time solve(brusselator_comp, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7,);
  @test sol1 ≈ reshape(sol5[end][2n*n+1:end], 2n*n, 4n*n) atol=1e-3

  # High tolerance to benchmark
  @info "  Running compile-time CSA"
  t1 = @belapsed solve($brusselator_comp, $(Rodas5(autodiff=false)), );
  @info "  Running DSA"
  t2 = @belapsed auto_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5()));
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($(ODEFunction(bfun, jac=brusselator_jac)), $b_u0, $((0.,10.)), $b_p, $(Rodas5(autodiff=false)));
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5(autodiff=false)), sensalg=SensitivityAlg(autojacvec=false));
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5(autodiff=false)), sensalg=SensitivityAlg(autojacvec=true));
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5()));
  print('\n')
  [t1, t2, t3, t4, t5, t6]
end

# 20×25 Jacobian
forward_pollution = let
  include("pollution.jl")
  @info "Running the pollution model:"
  pcomp, pu0, pp, pcompu0 = make_pollution()
  ptspan = (0.,60.)
  @info "  Running compile-time CSA"
  t1 = @belapsed solve($(ODEProblem(pcomp, pcompu0, ptspan, pp)), $(Rodas5(autodiff=false)),);
  @info "  Running DSA"
  t2 = @belapsed auto_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()));
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($(ODEFunction(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)));
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)), sensalg=SensitivityAlg(autojacvec=false));
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)), sensalg=SensitivityAlg(autojacvec=true));
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()));
  print('\n')
  [t1, t2, t3, t4, t5, t6]
end

forward_pkpd = let
  include("pkpd.jl")
  @info "Running the PKPD model:"
  #sol1 = solve(pkpdcompprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240,)[end][6:end]
  sol2 = vec(auto_sen(pkpdprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240))
  sol3 = vec(hcat(diffeq_sen(pkpdprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240)...))
  #@test sol1 ≈ sol2 atol=1e-3
  @test sol2 ≈ sol3 atol=1e-3
  @info "  Running compile-time CSA"
  #t1 = @belapsed solve($pkpdcompprob, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240,);
  @info "  Running DSA"
  t2 = @belapsed auto_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240);
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($(ODEFunction(pkpdf.f, jac=pkpdf.jac)), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240);
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240,
                    sensalg=SensitivityAlg(autojacvec=false));
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240,
                         sensalg=SensitivityAlg(autojacvec=true));
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240);
  print('\n')
  [0, t2, t3, t4, t5, t6]
end

using CSV, DataFrames
let
  forward_methods = ["Compile-time CSA", "DSA", "CSA user-Jacobian", "AD-Jacobian", "AD-Jv seeding", "Numerical Differentiation"]
  forward_timings = DataFrame(methods=forward_methods, LV=forward_lv, Bruss=forward_bruss, Pollution=forward_pollution, PKPD=forward_pkpd)
  bench_file_path = joinpath(@__DIR__, "..", "forward_timings.csv")
  display(forward_timings)
  @info "Writing the benchmark results to $bench_file_path"
  CSV.write(bench_file_path, forward_timings)
end

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
@time auto_sen(f, u0s, tspan, sp, Tsit5(), abstol=1e-5,reltol=1e-7)
# 139.719343 seconds (22.66 M allocations: 1.558 GiB, 0.58% gc time)
@btime auto_sen($f, $u0s, $tspan, $sp, $(Tsit5()), abstol=1e-5,reltol=1e-7)
# 135.706 μs (499 allocations: 84.55 KiB)
=#

#=
@time numerical_sen(lvdf, u0, tspan, p, Tsit5(), abstol=1e-5,reltol=1e-7)
# 3.401622 seconds (7.42 M allocations: 387.500 MiB, 6.50% gc time)
@time auto_sen(lvdf, u0, tspan, p, Tsit5(), abstol=1e-5,reltol=1e-7)
# 13.564837 seconds (43.31 M allocations: 2.326 GiB, 9.15% gc time)
@time diffeq_sen(lvdf, u0, tspan, p, Tsit5(), abstol=1e-5,reltol=1e-7)
# 5.712511 seconds (16.38 M allocations: 931.242 MiB, 9.13% gc time)
# with seeding 10.179159 seconds (32.43 M allocations: 1.730 GiB, 9.75% gc time)
@time diffeq_sen(lvdf_with_jacobian, u0, tspan, p, Tsit5(), abstol=1e-5,reltol=1e-7)
# 2.679172 seconds (6.21 M allocations: 320.881 MiB, 5.36% gc time)
@time solve(comprob, Tsit5(),abstol=1e-5,reltol=1e-7,)
# 3.484515 seconds (8.10 M allocations: 417.261 MiB, 7.50% gc time)

@btime numerical_sen($lvdf, $u0, $tspan, $p, $(Tsit5()), abstol=1e-5,reltol=1e-7)
# 534.718 μs (2614 allocations: 222.88 KiB)
@btime auto_sen($lvdf, $u0, $tspan, $p, $(Tsit5()), abstol=1e-5,reltol=1e-7)
# 99.404 μs (485 allocations: 57.11 KiB)
@btime diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5()), abstol=1e-5,reltol=1e-7)
# 308.289 μs (8137 allocations: 391.78 KiB)
# with seeding 268.012 μs (8310 allocations: 406.86 KiB)
@btime diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Tsit5()), abstol=1e-5,reltol=1e-7)
# 263.562 μs (8084 allocations: 389.08 KiB)
@btime solve($comprob, $(Tsit5()),abstol=1e-5,reltol=1e-7,)
# 36.517 μs (111 allocations: 14.67 KiB)
=#
