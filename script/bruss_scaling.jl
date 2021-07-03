include("sensitivity.jl")
include("brusselator.jl")

using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, DiffEqDiffTools
using DiffEqSensitivity: alg_autodiff
using LinearAlgebra
using Test
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')
DiffEqBase.has_tgrad(::ODEForwardSensitivityFunction) = false
DiffEqBase.has_invW(::ODEForwardSensitivityFunction) = false
DiffEqBase.has_jac(::ODEForwardSensitivityFunction) = false

bt = 0:0.1:1
tspan = (0.0, 1.0)
forwarddiffn = vcat(2:10,12,15)
reversediffn = 2:10
numdiffn = vcat(2:10,12)
csan = vcat(2:10,12,15,17)
#csaseedn = 2:10

_adjoint_methods = ntuple(2) do ii
  Alg = (InterpolatingAdjoint, QuadratureAdjoint)[ii]
  (
    user = Alg(autodiff=false,autojacvec=false), # user Jacobian
    adjc = Alg(autodiff=true,autojacvec=false), # AD Jacobian
    advj = Alg(autodiff=true,autojacvec=true), # AD vJ
    advj_compiled = Alg(autodiff=true,autojacvec=ReverseDiffVJP(true)), # AD vJ compiled
  )
end |> NamedTuple{(:interp, :quad)}
adjoint_methods = mapreduce(collect, vcat, _adjoint_methods)
tols = (abstol=1e-5, reltol=1e-7)

println("Forward Diff")
forwarddiff = map(forwarddiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(ForwardDiff.gradient), tols...)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(ForwardDiff.gradient), tols...)
  @show n,t
  t
end
open("../bruss_scaling_data.txt", "w") do f
  write(f, "forwarddiffn = $forwarddiffn \n")
  write(f, "forwarddiff = $forwarddiff \n")
end

println("Reverse Diff")
reversediff = map(reversediffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...)
  @show n,t
  t
end
open("../bruss_scaling_data.txt", "w") do f
  write(f, "reversediffn = $reversediffn \n")
  write(f, "reversediff = $reversediff \n")
end

println("Num Diff")
numdiff = map(numdiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(DiffEqDiffTools.finite_difference_gradient), tols...)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(DiffEqDiffTools.finite_difference_gradient), tols...)
  @show n,t
  t
end
open("../bruss_scaling_data.txt", "w") do f
  write(f, "numdiffn = $numdiffn \n")
  write(f, "numdiff = $numdiff \n")
end


println("CSA")
csa = map(csan) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @time ts = map(adjoint_methods) do alg
    @info "Runing $alg"
    f = alg_autodiff(alg) ? bfun : ODEFunction(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    @time diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...)
    t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...)
    return t
  end
  @show n,ts
  ts
end

open("../bruss_scaling_data.txt", "w") do f
  write(f, "csan = $csan \n")
  write(f, "csa = $csa \n")
end
