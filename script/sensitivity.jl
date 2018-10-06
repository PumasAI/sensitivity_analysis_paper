using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, DiffEqDiffTools, Calculus, DiffResults

function diffeq_sen(f, u0, tspan, p, alg=Tsit5(); save_everystep=false, kwargs...)
    prob = ODELocalSensitivityProblem(f,u0,tspan,p)
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
                       abstol=1e-5, reltol=1e-7, iabstol=abstol, ireltol=reltol, kwargs...)
    prob = ODEProblem(df,u0,tspan,p)
    sol = solve(prob, alg, abstol=abstol, reltol=reltol, saveat=t; kwargs...)
    dg(out,u,p,t,i) = (out.=1.0.-u)
    adjoint_sensitivities(sol,alg,dg,t,abstol=abstol,
                          reltol=reltol,iabstol=abstol,ireltol=reltol)
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

function diffeq_sen_full(f, u0, tspan, p, t)
    prob = ODELocalSensitivityProblem(f,u0,tspan,p)
    sol = solve(prob,Vern6(),abstol=1e-5,reltol=1e-7,saveat=t,save_everystep=false)
    extract_local_sensitivities(sol)
end

function auto_sen_full(f, u0, tspan, p, t)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(u0),tspan,p)
        u = vec(solve(prob,Vern6(),saveat=t,abstol=1e-5,reltol=1e-7,save_everystep=false))
    end
    sol, sens = test_f(p), ForwardDiff.jacobian(test_f, p)
    [reshape(sol',length(u0),length(t)), [reshape(sens[:,i]',length(u0),length(t)) for i in 1:length(p)]]
end

function numerical_sen(f,u0, tspan, p, alg=Tsit5(); save_everystep=false, kwargs...)
    test_f(out,p) = begin
        prob = ODEProblem(f,eltype(p).(u0),tspan,p)
        out .= solve(prob,alg; save_everystep=save_everystep, kwargs...)[end]
    end
    DiffEqDiffTools.finite_difference_jacobian!(Matrix{Float64}(undef,length(u0),length(p)),test_f, p, DiffEqDiffTools.JacobianCache(p,Array{Float64}(undef,length(u0))))
end
