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
forwarddiffn = vcat(2:10,12,15)
reversediffn = 2:10
numdiffn = vcat(2:10,12)
csan = vcat(2:10,12,15:5:20)
csaseedn = vcat(2:10,12,15:5:20)

println("Forward Diff")
forwarddiff = map(forwarddiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(ForwardDiff.gradient), save_everystep=false)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(ForwardDiff.gradient), save_everystep=false)
  @show n,t
end
println("Reverse Diff")
reversediff = map(reversediffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), diffalg=(ReverseDiff.gradient), save_everystep=false)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), diffalg=(ReverseDiff.gradient), save_everystep=false)
  @show n,t
end
println("Num Diff")
numdiff = map(numdiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(DiffEqDiffTools.finite_difference_gradient), save_everystep=false)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()), diffalg=(DiffEqDiffTools.finite_difference_gradient), save_everystep=false)
  @show n,t
end
println("CSA")
csa = map(csan) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false))
  t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=false))
  @show n,t
end
println("CSA Seed")
csaseed = map(csaseedn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true))
  t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)), save_everystep=false, sensalg=SensitivityAlg(autojacvec=true))
  @show n,t
end

open("../bruss_scaling_data.txt", "w") do f
  write(f, "forwarddiffn = $forwarddiffn")
  write(f, "forwarddiff = $forwarddiff")
  write(f, "reversediffn = $reversediffn")
  write(f, "reversediff = $reversediff")
  write(f, "numdiffn = $numdiffn")
  write(f, "numdiff = $numdiff")
  write(f, "csan = $csan")
  write(f, "csa = $csa")
  write(f, "csaseedn = $csaseedn")
  write(f, "csaseed = $csaseed")
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
