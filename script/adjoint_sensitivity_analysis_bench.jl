include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools#, Profile, ProfileView
using LinearAlgebra
using Test
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')
DiffEqBase.has_tgrad(::ODELocalSensitivityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitivityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitivityFunction) = false

adjoint_lv = let
  include("lotka-volterra.jl")
  @info "Running the Lotka-Volerra model:"
  lvu0 = [1.,1.]; lvtspan = (0.0, 10.0); lvp = [1.5,1.0,3.0];
  lvt = 0:0.5:10
  @time lsol1 = auto_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg=(ForwardDiff.gradient), abstol=1e-5,reltol=1e-7);
  @time lsol2 = auto_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg=(ReverseDiff.gradient), abstol=1e-5,reltol=1e-7);
  @time lsol3 = diffeq_sen_l2(lvdf_with_jacobian, lvu0, lvtspan, lvp, lvt, (Tsit5()), abstol=1e-5,reltol=1e-7);
  @time lsol4 = diffeq_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()), abstol=1e-5,reltol=1e-7;
                               sensalg=(SensitivityAlg(autojacvec=false)));
  @time lsol5 = diffeq_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()), abstol=1e-5,reltol=1e-7;
                               sensalg=(SensitivityAlg(autojacvec=true))); # with seeding
  @time lsol6 = numerical_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()), abstol=1e-5,reltol=1e-7);
  @test maximum(abs, lsol1 .- lsol2)/maximum(abs,  lsol1) < 0.2
  @test maximum(abs, lsol1 .- lsol3')/maximum(abs, lsol1) < 0.2
  @test maximum(abs, lsol1 .- lsol4')/maximum(abs, lsol1) < 0.2
  @test maximum(abs, lsol1 .- lsol5')/maximum(abs, lsol1) < 0.2
  @test maximum(abs, lsol1 .- lsol6)/maximum(abs, lsol1) < 0.2
  t1 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); diffalg=$(ForwardDiff.gradient));
  t2 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); diffalg=$(ReverseDiff.gradient));
  t3 = @belapsed diffeq_sen_l2($lvdf_with_jacobian, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()));
  t4 = @belapsed diffeq_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5());
                               sensalg=$(SensitivityAlg(autojacvec=false)));
  t5 = @belapsed diffeq_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5());
                               sensalg=$(SensitivityAlg(autojacvec=true))); # with seeding
  t6 = @belapsed numerical_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()));
  [t1, t2, t3, t4, t5, t6]
end

forward_bruss = let
  include("brusselator.jl")
  @info "Running the Brusselator model:"
  n = 5
  # Run low tolerance to test correctness
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  sol1 = @time numerical_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(), abstol=1e-5,reltol=1e-7);
  sol2 = @time auto_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(), abstol=1e-5,reltol=1e-7);
  sol3 = @time diffeq_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
  sol4 = @time diffeq_sen(ODEFunction(bfun, jac=brusselator_jac), b_u0, (0.,10.), b_p, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
  sol5 = @time solve(brusselator_comp, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7,);
  @test maximum(abs, sol1 - sol2) < 1e-3
  @test maximum(abs, sol2 - hcat(sol3...)) < 1e-3
  @test maximum(abs, sol2 - hcat(sol4...)) < 1e-3
  @test maximum(sol2 - reshape(sol5[end][2n*n+1:end], 2n*n, 4n*n)) < 1e-3

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

adjoint_pollution = let
  include("pollution.jl")
  @info "Running the Pollution model:"
  pcomp, pu0, pp, pcompu0 = make_pollution();
  ptspan = (0., 60.)
  pts = 0+0.01:0.5:60-0.01
  @time psol1 = auto_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)), diffalg=(ForwardDiff.gradient));
  @time psol2 = auto_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)), diffalg=(ReverseDiff.gradient));
  @time psol3 = diffeq_sen_l2((ODEFunction(pollution.f, jac=pollution.jac)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)));
  @time psol4 = diffeq_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)),
                               sensalg=(SensitivityAlg(autojacvec=false)));
  @time psol5 = diffeq_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)),
                               sensalg=(SensitivityAlg(autojacvec=true)));
  @time psol6 = numerical_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)));
  @test maximum(abs, psol1 .- psol2)/maximum(abs,  psol1) < 1e-2
  @test maximum(abs, psol1 .- psol3')/maximum(abs, psol1) < 1e-2
  @test maximum(abs, psol1 .- psol4')/maximum(abs, psol1) < 1e-2
  @test maximum(abs, psol1 .- psol5')/maximum(abs, psol1) < 1e-2
  @test maximum(abs, psol1 .- psol6)/maximum(abs, psol1) < 1e-2
  t1 = @belapsed auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)), diffalg=$(ForwardDiff.gradient));
  t2 = @belapsed auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)), diffalg=$(ReverseDiff.gradient));
  t3 = @belapsed diffeq_sen_l2($(ODEFunction(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)));
  t4 = @belapsed diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)),
                               sensalg=$(SensitivityAlg(autojacvec=false)));
  t5 = @belapsed diffeq_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)),
                               sensalg=(SensitivityAlg(autojacvec=true)));
  t6 = @belapsed numerical_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)));
  [t1, t2, t3, t4, t5, t6]
end

adjoint_pkpd = let
  include("pkpd.jl")
  @info "Running the PKPD model:"
  pts = 0:0.5:50
  # need to use lower tolerances to avoid running into the complex domain because of exponentiation
  pkpdsol1 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()), callback=pkpdcb, tstops=0:24:240,
                                diffalg=(ForwardDiff.gradient), reltol=1e-5, abstol=1e-7);
  pkpdsol2 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()), callback=pkpdcb, tstops=0:24:240,
                                diffalg=(ReverseDiff.gradient), reltol=1e-5, abstol=1e-7);
  pkpdsol3 = @time diffeq_sen_l2((ODEFunction(pkpdf.f, jac=pkpdf.jac)), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()),
                                  callback=pkpdcb, tstops=0:24:240, reltol=1e-5, abstol=1e-7);
  pkpdsol4 = @time diffeq_sen_l2((ODEFunction(pkpdf.f)), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()),
                                  sensalg=(SensitivityAlg(autojacvec=false)), callback=pkpdcb, tstops=0:24:240, reltol=1e-5, abstol=1e-7);
  pkpdsol5 = @time diffeq_sen_l2((ODEFunction(pkpdf.f)), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()),
                                  sensalg=(SensitivityAlg(autojacvec=true)), callback=pkpdcb, tstops=0:24:240, reltol=1e-5, abstol=1e-7);
  pkpdsol6 = @time numerical_sen_l2((ODEFunction(pkpdf.f)), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()),
                                     callback=pkpdcb, tstops=0:24:240, reltol=1e-5, abstol=1e-7);
  @test maximum(abs, pkpdsol1 .- pkpdsol2)/maximum(abs,  pkpdsol1) < 0.2
  @test maximum(abs, pkpdsol1 .- pkpdsol3')/maximum(abs,  pkpdsol1) < 0.2
  @test maximum(abs, pkpdsol1 .- pkpdsol4')/maximum(abs,  pkpdsol1) < 0.2
  @test maximum(abs, pkpdsol1 .- pkpdsol5')/maximum(abs,  pkpdsol1) < 0.2
  @test maximum(abs, pkpdsol1 .- pkpdsol6)/maximum(abs,  pkpdsol1) < 0.2
  t1 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()), callback=pkpdcb, tstops=0:24:240,
                                diffalg=$(ForwardDiff.gradient), reltol=1e-5, abstol=1e-7);
  t2 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()), callback=pkpdcb, tstops=0:24:240,
                                diffalg=$(ReverseDiff.gradient), reltol=1e-5, abstol=1e-7);
  t3 = @belapsed diffeq_sen_l2($(ODEFunction(pkpdf.f, jac=pkpdf.jac)), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()), tstops=0:24:240,
                                  callback=pkpdcb, reltol=1e-5, abstol=1e-7);
  t4 = @belapsed diffeq_sen_l2($(ODEFunction(pkpdf.f)), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()), tstops=0:24:240,
                                  sensalg=$(SensitivityAlg(autojacvec=false)), callback=pkpdcb, reltol=1e-5, abstol=1e-7);
  t5 = @belapsed diffeq_sen_l2($(ODEFunction(pkpdf.f)), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()), tstops=0:24:240,
                                  sensalg=(SensitivityAlg(autojacvec=true)), callback=pkpdcb, reltol=1e-5, abstol=1e-7);
  t6 = @belapsed numerical_sen_l2($(ODEFunction(pkpdf.f)), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()), tstops=0:24:240,
                                     callback=pkpdcb, reltol=1e-5, abstol=1e-7);
  [t1, t2, t3, t4, t5, t6]
end

using CSV, DataFrames
let
  adjoint_methods = ["Forward-Mode DSAAD", "Reverse-Mode DSAAD", "CASA User-Jacobian",
                     "CASA AD-Jacobian", "CASA AD-Jv seeding", "Numerical Differentiation"]
  adjoint_timings = DataFrame(methods=adjoint_methods, LV=adjoint_lv, Bruss=adjoint_bruss,
                               Pollution=adjoint_pollution, PKPD=adjoint_pkpd)
  bench_file_path = joinpath(@__DIR__, "..", "adjoint_timings.csv")
  display(adjoint_timings)
  @info "Writing the benchmark results to $bench_file_path"
  CSV.write(bench_file_path, adjoint_timings)
end
