include("sensitivity.jl")
include("brusselator.jl")

using Plots
gr()

using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, DiffEqDiffTools
using LinearAlgebra
using Test
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')
DiffEqBase.has_tgrad(::ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitvityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitvityFunction) = false

bt = 0:0.1:1
tspan = (-0.01, 1.01)
forwarddiffn = 2:10
reversediffn = 2:6
numdiffn = 2:10
csan = 2:10
csaseedn = vcat(2:10, 15:5:25)


forwarddiff = map(forwarddiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(ForwardDiff.gradient), save_everystep=false)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(ForwardDiff.gradient), save_everystep=false)
  @show t
end
reversediff = map(reversediffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), diffalg=(ReverseDiff.gradient), save_everystep=false)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), diffalg=(ReverseDiff.gradient), save_everystep=false)
  @show t
end
numdiff = map(numdiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(DiffEqDiffTools.finite_difference_gradient), save_everystep=false)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(DiffEqDiffTools.finite_difference_gradient), save_everystep=false)
  @show t
end
csa = map(csan) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false))
  t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false))
  @show t
end
csaseed = map(csaseedn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true))
  t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true))
  @show t
end

plt = plot(title="Brusselator Scaling Plot")
plot!(plt, forwarddiffn, forwarddiff)
plot!(plt, reversediffn, reversediff)
plot!(plt, numdiffn, numdiff)
plot!(plt, csan, csa)
plot!(plt, csaseedn, csaseed)
xaxis!("Dimension")
yaxis!("Runtime (s)")
savefig("figure2.pdf")
