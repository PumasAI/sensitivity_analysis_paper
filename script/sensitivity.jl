using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, DiffEqDiffTools, Calculus, DiffResults, DiffEqBase

function diffeq_sen(prob::DiffEqBase.DEProblem, args...; kwargs...)
  diffeq_sen(prob.f, prob.u0, prob.tspan, prob.p, args...; kwargs...)
end
function auto_sen(prob::DiffEqBase.DEProblem, args...; kwargs...)
  auto_sen(prob.f, prob.u0, prob.tspan, prob.p, args...; kwargs...)
end

function diffeq_sen(f, u0, tspan, p, alg=Tsit5(); save_everystep=false, sensalg=SensitivityAlg(), kwargs...)
  prob = ODELocalSensitivityProblem(f,u0,tspan,p,sensalg)
  sol = solve(prob,alg; save_everystep=save_everystep, kwargs...)
  extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, u0, tspan, p, alg=Tsit5(); save_everystep=false, kwargs...)
  test_f(p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    solve(prob,alg; save_everystep=save_everystep, kwargs...)[end]
  end
  ForwardDiff.jacobian(test_f, p)
end

function diffeq_sen_l2(df, u0, tspan, p, t, alg=Tsit5();
                       abstol=1e-5, reltol=1e-7, iabstol=abstol, ireltol=reltol,
                       sensalg=SensitivityAlg(), kwargs...)
  prob = ODEProblem(df,u0,tspan,p)
  saveat = tspan[1] != t[1] && tspan[end] != t[end] ? vcat(tspan[1],t,tspan[end]) : t
  sol = solve(prob, alg, abstol=abstol, reltol=reltol, saveat=saveat; kwargs...)
  dg(out,u,p,t,i) = (out.=1.0.-u)
  adjoint_sensitivities(sol,alg,dg,t,abstol=abstol,
                        reltol=reltol,iabstol=abstol,ireltol=reltol,sensealg=sensalg)
end

function auto_sen_l2(f, u0, tspan, p, t, alg=Tsit5(); diffalg=ReverseDiff.gradient, kwargs...)
  test_f(p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    sol = solve(prob,alg,saveat=t; kwargs...)
    sum(sol.u) do x
      sum(z->(1-z)^2/2, x)
    end
  end
  diffalg(test_f, p)
end

function numerical_sen_l2(f, u0, tspan, p, t, alg=Tsit5(); save_everystep=false, kwargs...)
  test_f(p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    sol = solve(prob,alg,saveat=t; kwargs...)
    sum(sol.u) do x
      sum(z->(1-z)^2/2, x)
    end
  end
  DiffEqDiffTools.finite_difference_gradient(test_f, p, Val{:central})
end

function diffeq_sen_full(f, u0, tspan, p, t; alg=Tsit5(), save_everystep=false, sensalg=SensitivityAlg(), kwargs...)
  prob = ODELocalSensitivityProblem(f,u0,tspan,p)
  sol = solve(prob,alg;saveat=t,save_everystep=false,sensealg=sensalg,kwargs...)
  extract_local_sensitivities(sol)
end

function auto_sen_full(f, u0, tspan, p, t; alg=Tsit5(), save_everystep=false, kwargs...)
  test_f(p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    vec(solve(prob,alg;saveat=t,save_everystep=save_everystep,kwargs...))
  end
  result = DiffResults.DiffResult(similar(p, length(t)*length(u0)), similar(p, length(t)*length(u0), length(p)))
  result = ForwardDiff.jacobian!(result, test_f, p)
  sol, sens = DiffResults.value(result), DiffResults.jacobian(result)
  sens
  reshape(sol',length(u0),length(t)), [Array(reshape(@view(sens[:,i])',length(u0),length(t))) for i in 1:length(p)]
end

function numerical_sen_full(f, u0, tspan, p, t; alg=Tsit5(), save_everystep=false, kwargs...)
  n = length(u0)
  test_f(out,p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    sol = solve(prob,alg; saveat=t,save_everystep=save_everystep, kwargs...)
    out .= vec(sol)
  end
  fxcache = similar(p, length(t)*n)
  cache = DiffEqDiffTools.JacobianCache(p, fxcache)
  sens = similar(p, length(t)*length(u0), length(p))
  DiffEqDiffTools.finite_difference_jacobian!(sens, test_f, p, cache)
  test_f(fxcache, p)
  sol = fxcache
  reshape(sol',length(u0),length(t)), [Array(reshape(@view(sens[:,i])',length(u0),length(t))) for i in 1:length(p)]
end

function numerical_sen(f,u0, tspan, p, alg=Tsit5(); save_everystep=false, kwargs...)
  test_f(out,p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    copyto!(out, solve(prob,alg; save_everystep=save_everystep, kwargs...)[end])
  end
  DiffEqDiffTools.finite_difference_jacobian!(Matrix{Float64}(undef,length(u0),length(p)),test_f, p, DiffEqDiffTools.JacobianCache(p,Array{Float64}(undef,length(u0))))
end
