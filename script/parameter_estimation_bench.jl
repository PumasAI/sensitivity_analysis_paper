include("parameter_estimation.jl")
using LinearAlgebra

forward_param_lv, adjoint_param_lv = let
  include("lotka-volterra.jl")
  @info "Running Lotka-Volterra"
  u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
  arr = trues(10)
  arr[1] = false
  param_benchmark(lvdf, lvcom_df, lvdf_with_jacobian.jac,
                    u0, [u0; zeros(6)], tspan, p, range(0, stop=10, length=100), 0.8.*p,
                    dropfirst=true, adjoint_methods=ADJOINT_METHODS, run=arr)
end

forward_param_bruss, adjoint_param_bruss = let
  include("brusselator.jl")
  @info "Running Brusselator"
  n = 3
  tspan = (0., 5.)
  arr = trues(10)
  arr[1] = false
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  param_benchmark(bfun, brusselator_comp.f, brusselator_jac,
                    b_u0, brusselator_comp.u0, tspan, b_p, range(tspan[1]+0.01, stop=tspan[end]-0.01, length=20), 0.9.*b_p,
                    alg=Rodas5(autodiff=false), adjoint_methods=ADJOINT_METHODS[1:2end÷3], run=arr)
end

forward_param_pollution, adjoint_param_pollution = let
  include("pollution.jl")
  @info "Running pollution"
  pcomp, pu0, pp, pcompu0 = make_pollution()
  ptspan = (0., 5.)
  arr = trues(10)
  arr[1] = false
  param_benchmark(pollution.f, pcomp, pollution.jac,
                    pu0, pcompu0, ptspan, pp, range(ptspan[1]+0.01, stop=ptspan[end]-0.01, length=10), 0.9.*pp,
                    alg=Rodas5(autodiff=false), adjoint_methods=ADJOINT_METHODS[1:2end÷3], run=arr)
end

forward_param_pkpd, adjoint_param_pkpd = let
  include("pkpd.jl")
  @info "Running the PKPD"
  t = 0.:6:240
  arr = trues(10)
  arr[1] = false
  param_benchmark(pkpdf.f, pkpdcompprob.f, pkpdf.jac,
                  pkpdu0, pkpdcompprob.u0, pkpdtspan, pkpdp, t, callback=pkpdcb, tstops=0:24.:240,
                  0.95.*pkpdp.+0.001, reltol=1e-7, abstol=1e-7, iter=2, iabstol=1e-12, ireltol=1e-12,
                  lower=0.5.*pkpdp.-0.02, upper=1.5.*pkpdp.+0.02, run=arr, adjoint_methods=ADJOINT_METHODS[1:2end÷3])
end

open("../forward_param_timings.txt", "w") do f
  write(f, "lv = $forward_param_lv \n")
  write(f, "bruss = $forward_param_bruss \n")
  write(f, "pollution = $forward_param_pollution \n")
  write(f, "pkpd = $forward_param_pkpd \n")
end

open("../adjoint_param_timings.txt", "w") do f
  write(f, "lv = $adjoint_param_lv \n")
  write(f, "bruss = $adjoint_param_bruss \n")
  write(f, "pollution = $adjoint_param_pollution \n")
  write(f, "pkpd = $adjoint_param_pkpd \n")
end
