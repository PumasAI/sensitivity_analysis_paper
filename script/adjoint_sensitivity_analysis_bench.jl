using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools, Profile, ProfileView, ParameterizedFunctions

@eval begin
    df = @ode_def $(gensym()) begin
      dx = a*x - b*x*y
      dy = -c*y + x*y
    end a b c
end

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0];

function diffeq_sen_l2(df, u0, tspan, p, t, alg=Vern6())
    prob = ODEProblem(df,u0,tspan,p)
    sol = solve(prob, alg, abstol=1e-5, reltol=1e-7)
    dg(out,u,p,t,i) = (out.=1.0.-u)
    adjoint_sensitivities(sol,alg,dg,t,abstol=1e-5,
                          reltol=1e-7,iabstol=1e-5,ireltol=1e-7)
end

function auto_sen_l2(f, init, tspan, p, t, alg=Vern6(); diffalg=ForwardDiff.gradient)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),eltype(p).(tspan),p)
        sol = solve(prob,alg,abstol=1e-5,reltol=1e-7,saveat=t)
        sum(sol.u) do x
            sum(z->(1-z)^2/2, x)
        end
    end
    diffalg(test_f, p)
end

#@time auto_sen_l2(df, u0, tspan, p, t; diffalg=ReverseDiff.gradient)
#@btime auto_sen_l2($df, $u0, $tspan, $p, $t; diffalg=$(ReverseDiff.gradient))

t = 0:0.5:10
@time auto_sen_l2(df, u0, tspan, p, t)
@time diffeq_sen_l2(df, u0, tspan, p, t)
@btime auto_sen_l2($df, $u0, $tspan, $p, $t)
@btime diffeq_sen_l2($df, $u0, $tspan, $p, $t)

#=============================
ReverseDiff is currently broken
julia> @time auto_sen_l2(df, u0, tspan, p, t)
 14.558341 seconds (37.78 M allocations: 1.896 GiB, 8.39% gc time)
3-element Array{Float64,1}:
  25.50046010429557
 -77.2548746896757
  93.53153569559123

julia> @time diffeq_sen_l2(df, u0, tspan, p, t)
 13.633570 seconds (40.21 M allocations: 3.180 GiB, 8.28% gc time)
1×3 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 25.5013  -77.2551  93.5321

julia> @btime auto_sen_l2($df, $u0, $tspan, $p, $t)
  117.032 μs (652 allocations: 52.70 KiB)
3-element Array{Float64,1}:
  25.50046010429557
 -77.2548746896757
  93.53153569559123

julia> @btime diffeq_sen_l2($df, $u0, $tspan, $p, $t)
  4.709 ms (78461 allocations: 1.89 MiB)
1×3 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 25.5013  -77.2551  93.5321
=============================#
