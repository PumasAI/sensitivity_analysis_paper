include("parameter_estimation.jl")
@elapsed 1+1
using LinearAlgebra
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

forward_param_lv, adjoint_param_lv = let
  include("lotka-volterra.jl")
  @info "Running Lotka-Volterra"
  u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
  param_benchmark(lvdf, lvcom_df, lvdf_with_jacobian.jac,
                    u0, [u0; zeros(6)], tspan, p, range(0, stop=10, length=100), 0.8.*p,
                    iter=10, dropfirst=true, verbose=true)
end

forward_param_bruss, adjoint_param_bruss = let
  include("brusselator.jl")
  @info "Running Brusselator"
  n = 3
  tspan = (0., 5.)
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  param_benchmark(bfun, brusselator_comp.f, brusselator_jac,
                    b_u0, brusselator_comp.u0, tspan, b_p, range(tspan[1], stop=tspan[end], length=20), 0.9.*b_p,
                    alg=Rodas5(autodiff=false), verbose=true)
end

forward_param_pollution, adjoint_param_pollution = let
  include("pollution.jl")
  @info "Running pollution"
  pcomp, pu0, pp, pcompu0 = make_pollution()
  ptspan = (0., 5.)
  param_benchmark(pollution.f, pcomp, pollution.jac,
                    pu0, pcompu0, ptspan, pp, range(ptspan[1], stop=ptspan[end], length=10), 0.9.*pp,
                    alg=Rodas5(autodiff=false))
end

forward_param_pkpd, adjoint_param_pkpd = let
  include("pkpd.jl")
  @info "Running the PKPD"
  t = 1.: 49
  param_benchmark(pkpdf.f, pkpdcompprob.f, pkpdf.jac,
                    pkpdu0, pkpdcompprob.u0, pkpdtspan, pkpdp, t,
                    0.9.*pkpdp, tstops=t, callback=pkpdcb, reltol=1e-7, abstol=1e-7, iter=1, verbose=true)
end


using CSV, DataFrames
let
  forward_methods = ["Compile-time CSA", "DSA", "CSA user-Jacobian", "CSA AD-Jacobian",
                           "CSA AD-Jv seeding", "Numerical Differentiation"]
  adjoint_methods = ["Forward-Mode DSAAD", "Reverse-Mode DSAAD", "CASA User-Jacobian",
                     "CASA AD-Jacobian", "CASA AD-Jv seeding", "Numerical Differentiation"]
  forward_param_timings = DataFrame(methods=forward_methods, LV=forward_param_lv,
                                    Bruss=forward_param_bruss, Pollution=forward_param_pollution,
                                    PKPD=forward_param_pkpd)
  adjoint_param_timings = DataFrame(methods=adjoint_methods, LV=adjoint_param_lv,
                                    Bruss=adjoint_param_bruss, Pollution=adjoint_param_pollution,
                                    PKPD=adjoint_param_pkpd)
  f_bench_file_path = joinpath(@__DIR__, "..", "forward_param_timings.csv")
  a_bench_file_path = joinpath(@__DIR__, "..", "adjoint_param_timings.csv")
  display(forward_param_timings)
  display(adjoint_param_timings)
  @info "Writing the benchmark results to $f_bench_file_path"
  CSV.write(f_bench_file_path, forward_param_timings)
  @info "Writing the benchmark results to $a_bench_file_path"
  CSV.write(a_bench_file_path, adjoint_param_timings)
end
