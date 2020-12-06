include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools#, Profile, ProfileView
using DiffEqSensitivity: alg_autodiff
using LinearAlgebra
using Test
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')
DiffEqBase.has_tgrad(::ODEForwardSensitivityFunction) = false
DiffEqBase.has_invW(::ODEForwardSensitivityFunction) = false
DiffEqBase.has_jac(::ODEForwardSensitivityFunction) = false

_adjoint_methods = ntuple(3) do ii
  Alg = (InterpolatingAdjoint, QuadratureAdjoint, BacksolveAdjoint)[ii]
  (
    user = Alg(autodiff=false,autojacvec=false), # user Jacobian
    adjc = Alg(autodiff=true,autojacvec=false), # AD Jacobian
    advj = Alg(autodiff=true,autojacvec=true), # AD vJ
  )
end |> NamedTuple{(:interp, :quad, :backsol)}
adjoint_methods = mapreduce(collect, vcat, _adjoint_methods)
tols = (abstol=1e-5, reltol=1e-7)

adjoint_lv = let
  include("lotka-volterra.jl")
  @info "Running the Lotka-Volerra model:"
  lvu0 = [1.,1.]; lvtspan = (0.0, 10.0); lvp = [1.5,1.0,3.0];
  lvt = 0:0.5:10
  @time lsol1 = auto_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg=(ForwardDiff.gradient), tols...);
  @time lsol2 = auto_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg=(ReverseDiff.gradient), tols...);
  @time lsol3 = map(adjoint_methods) do alg
    f = alg_autodiff(alg) ? lvdf : lvdf_with_jacobian
    diffeq_sen_l2(f, lvu0, lvtspan, lvp, lvt, (Tsit5()); sensalg=alg, tols...)
  end
  @time lsol4 = numerical_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, Tsit5(); tols...);
  @test maximum(abs, lsol1 .- lsol2)/maximum(abs,  lsol1) < 0.2
  @test all(i -> maximum(abs, lsol1 .- lsol3[i]')/maximum(abs, lsol1) < 0.2, eachindex(adjoint_methods))
  @test maximum(abs, lsol1 .- lsol4)/maximum(abs, lsol1) < 0.2
  t1 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); diffalg=$(ForwardDiff.gradient), $tols...);
  t2 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); diffalg=$(ReverseDiff.gradient), $tols...);
  t3 = map(adjoint_methods) do alg
    f = alg_autodiff(alg) ? lvdf : lvdf_with_jacobian
    @belapsed diffeq_sen_l2($f, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); sensalg=$alg, $tols...);
  end
  t4 = @belapsed numerical_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); $tols...);
  [t1, t2, t3, t4]
end

adjoint_bruss = let
  include("brusselator.jl")
  @info "Running the Brusselator model:"
  bt = 0:0.1:10
  tspan = (0.0, 10.0)
  n = 5
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @time bsol1 = auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(ForwardDiff.gradient), tols...);
  @time bsol2 = auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...);
  @time bsol3 = map(adjoint_methods) do alg
    @info "Runing $alg"
    f = alg_autodiff(alg) ? bfun : ODEFunction(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver, reltol=1e-7; sensalg=alg, tols...)
  end
  @time bsol4 = numerical_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); tols...);
  @test maximum(abs, bsol1 .- bsol2)/maximum(abs,  bsol1) < 1e-2
  # NOTE: black solve gives unstable results!!!
  @test all(i->maximum(abs, bsol1 .- bsol3[i]')/maximum(abs, bsol1) < 4e-2, eachindex(adjoint_methods)[1:2end÷3])
  @test all(i->maximum(abs, bsol1 .- bsol3[i]')/maximum(abs, bsol1) >= 4e-2, eachindex(adjoint_methods)[2end÷3+1:end])
  @test maximum(abs, bsol1 .- bsol4)/maximum(abs, bsol1) < 2e-2
  t1 = @belapsed auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()); diffalg=$(ForwardDiff.gradient), $tols...);
  t2 = @belapsed auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)); diffalg=$(ReverseDiff.gradient), $tols...);
  t3 = map(adjoint_methods[1:2end÷3]) do alg
    @info "Runing $alg"
    f = alg_autodiff(alg) ? bfun : ODEFunction(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    @elapsed diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...);
  end
  t3 = [t3; fill(NaN, length(adjoint_methods)÷3)]
  t4 = @belapsed numerical_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()); $tols...);
  [t1, t2, t3, t4]
end

adjoint_pollution = let
  include("pollution.jl")
  @info "Running the Pollution model:"
  pcomp, pu0, pp, pcompu0 = make_pollution();
  ptspan = (0.0, 60.0)
  pts = 0:0.5:60
  @time psol1 = auto_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); diffalg=(ForwardDiff.gradient), tols...);
  @time psol2 = auto_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...);
  @time psol3 = map(adjoint_methods) do alg
    @info "Runing $alg"
    f = alg_autodiff(alg) ? pollution.f : ODEFunction(pollution.f, jac=pollution.jac)
    solver = Rodas5(autodiff=false)
    diffeq_sen_l2(f, pu0, ptspan, pp, pts, solver; sensalg=alg, tols...);
  end
  @time psol4 = numerical_sen_l2((ODEFunction(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); tols...);
  @test maximum(abs, psol1 .- psol2)/maximum(abs,  psol1) < 1e-2
  # NOTE: black solve gives unstable results!!!
  @test all(i->maximum(abs, psol1 .- psol3[i]')/maximum(abs, psol1) < 1e-2, eachindex(adjoint_methods)[1:2end÷3])
  @test all(i->maximum(abs, psol1 .- psol3[i]')/maximum(abs, psol1) >= 1e-2, eachindex(adjoint_methods)[2end÷3+1:end])
  @test maximum(abs, psol1 .- psol4)/maximum(abs, psol1) < 1e-2
  t1 = @belapsed auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); diffalg=$(ForwardDiff.gradient), $tols...);
  t2 = @belapsed auto_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); diffalg=$(ReverseDiff.gradient), $tols...);
  t3 = map(adjoint_methods[1:2end÷3]) do alg
    @info "Runing $alg"
    f = alg_autodiff(alg) ? pollution.f : ODEFunction(pollution.f, jac=pollution.jac)
    solver = Rodas5(autodiff=false)
    @elapsed diffeq_sen_l2(f, pu0, ptspan, pp, pts, solver; sensalg=alg, tols...);
  end
  t4 = @belapsed numerical_sen_l2($(ODEFunction(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); $tols...);
  [t1, t2, t3, t4]
end

adjoint_pkpd = let
  include("pkpd.jl")
  @info "Running the PKPD model:"
  pts = 0:0.5:50
  # need to use lower tolerances to avoid running into the complex domain because of exponentiation
  pkpdsol1 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=(ForwardDiff.gradient), tols...);
  pkpdsol2 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=(ReverseDiff.gradient), tols...);
  pkpdsol3 = @time map(adjoint_methods) do alg
    f = alg_autodiff(alg) ? pkpdf.f : ODEFunction(pkpdf.f, jac=pkpdf.jac)
    diffeq_sen_l2(f, pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); sensalg=alg,
                                  callback=pkpdcb, tstops=0:24:240, tols...);
  end
  pkpdsol6 = @time numerical_sen_l2((ODEFunction(pkpdf.f)), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5());
                                     callback=pkpdcb, tstops=0:24:240, tols...);
  @test maximum(abs, pkpdsol1 .- pkpdsol2)/maximum(abs,  pkpdsol1) < 0.2
  @test all(i->maximum(abs, pkpdsol1 .- pkpdsol3[i]')/maximum(abs,  pkpdsol1) < 0.2, eachindex(adjoint_methods))
  @test maximum(abs, pkpdsol1 .- pkpdsol4)/maximum(abs,  pkpdsol1) < 0.2
  t1 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=$(ForwardDiff.gradient), $tols...);
  t2 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=$(ReverseDiff.gradient), $tols...);
  t3 = map(adjoint_methods) do alg
    f = alg_autodiff(alg) ? pkpdf.f : ODEFunction(pkpdf.f, jac=pkpdf.jac)
    @belapsed diffeq_sen_l2($f, $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); tstops=0:24:240,
                                  callback=pkpdcb, sensalg=$alg, tols...);
  end
  t4 = @belapsed numerical_sen_l2($(ODEFunction(pkpdf.f)), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); tstops=0:24:240,
                                     callback=$pkpdcb, $tols...);
  [t1, t2, t3, t4]
end

open("../adjoint_bench.txt", "w") do f
  write(f, "adjoint_lv = $adjoint_lv \n")
  write(f, "adjoint_bruss = $adjoint_bruss \n")
  write(f, "adjoint_pollution = $adjoint_pollution \n")
  write(f, "adjoint_pkpd = $adjoint_pkpd \n")
end

#=
using CSV, DataFrames
let
  adjoint_methods = ["Forward-Mode DSAAD", "Reverse-Mode DSAAD", "CASA User-Jacobian",
                     "CASA AD-Jacobian", "CASA AD-vJ seeding", "Numerical Differentiation"]
  adjoint_timings = DataFrame(methods=adjoint_methods, LV=adjoint_lv, Bruss=adjoint_bruss,
                               Pollution=adjoint_pollution, PKPD=adjoint_pkpd)
  bench_file_path = joinpath(@__DIR__, "..", "adjoint_timings.csv")
  display(adjoint_timings)
  @info "Writing the benchmark results to $bench_file_path"
  CSV.write(bench_file_path, adjoint_timings)
end
=#
