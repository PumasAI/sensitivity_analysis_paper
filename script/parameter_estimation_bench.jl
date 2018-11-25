include("parameter_estimation.jl")
@elapsed 1+1

DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

forward_param_lv = let
  include("lotka-volterra.jl")
  @info "Running Lotka-Volterra"
  u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
  forward_benchmark(lvdf, lvcom_df, lvdf_with_jacobian.jac,
                    u0, [u0; zeros(6)], tspan, p, range(0, stop=10, length=20), 0.9.*p)
end

forward_param_bruss = let
  include("brusselator.jl")
  @info "Running Brusselator"
  n = 3
  tspan = (0., 1.)
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  forward_benchmark(bfun, brusselator_comp.f, brusselator_jac,
                    b_u0, brusselator_comp.u0, tspan, b_p, range(tspan[1], stop=tspan[end], length=10), 0.9.*b_p,
                    alg=Rodas5(autodiff=false))
end

forward_param_pollution = let
  include("pollution.jl")
  @info "Running pollution"
  pcomp, pu0, pp, pcompu0 = make_pollution()
  ptspan = (0., 5.)
  forward_benchmark(pollution.f, pcomp, pollution.jac,
                    pu0, pcompu0, ptspan, pp, range(ptspan[1], stop=ptspan[end], length=10), 0.9.*pp,
                    alg=Rodas5(autodiff=false))
end

forward_param_pkpd = let
  include("pkpd.jl")
  @info "Running the PKPD"
  forward_benchmark(pkpdf.f, pkpdcompprob.f, pkpdf.jac,
                    pkpdu0, pkpdcompprob.u0, pkpdtspan, pkpdp, range(pkpdtspan[1], stop=pkpdtspan[end], length=30),
                    0.9.*pkpdp, callbak=pkpdcb, reltol=1e-7, abstol=1e-5)
end


using CSV, DataFrames
let
  forward_param_methods = ["Compile-time CSA", "DSA", "CSA user-Jacobian", "CSA AD-Jacobian",
                           "CSA AD-Jv seeding", "Numerical Differentiation"]
  forward_param_timeings = DataFrame(methods=forward_param_methods, LV=forward_param_lv)
  bench_file_path = joinpath(@__DIR__, "..", "forward_param_timings.csv")
  display(forward_param_timeings)
  @info "Writing the benchmark results to $bench_file_path"
  CSV.write(bench_file_path, forward_param_timeings)
end
