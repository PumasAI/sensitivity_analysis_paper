using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff

function diffeq_sen(f, init, tspan, p, alg=Tsit5(); save_everystep=false, kwargs...)
    prob = ODELocalSensitivityProblem(f,init,tspan,p)
    sol = solve(prob,alg; save_everystep=save_everystep, kwargs...)
    extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, init, tspan, p, alg=Tsit5(); save_everystep=false, kwargs...)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),tspan,p)
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

function auto_sen_l2(f, init, tspan, p, t, alg=Tsit5(); diffalg=ReverseDiff.gradient, kwargs...)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),tspan,p)
        sol = solve(prob,alg,saveat=t; kwargs...)
        sum(sol.u) do x
            sum(z->(1-z)^2/2, x)
        end
    end
    diffalg(test_f, p)
end
