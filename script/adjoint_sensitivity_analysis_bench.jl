using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, ReverseDiff, BenchmarkTools, Profile, ProfileView

function df(du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0];

function diffeq_sen_l2(f, init, tspan, p, alg=Vern6())
    prob = ODEProblem(f,init,tspan,p)
    dg(out,u,i) = (out.=1.0.-u)
    _sol = solve(prob, alg, abstol=1e-5, reltol=1e-7)
    sol = adjoint_sensitivities(_sol,alg,dg,tspan[end],abstol=1e-5,
                            reltol=1e-7,iabstol=1e-5,ireltol=1e-7)
    #extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen_l2(f, init, tspan, p, alg=Vern6(); diffalg=ForwardDiff.gradient)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),eltype(p).(tspan),p)
        sol = solve(prob,alg,save_everystep=false,abstol=1e-5,reltol=1e-7)[end]
        sum(x->(1-x)^2/2, sol)
    end
    diffalg(test_f, p)
end

@time auto_sen_l2(df, u0, tspan, p)
#@time auto_sen_l2(df, u0, tspan, p; diffalg=ReverseDiff.gradient)
#@time diffeq_sen_l2(df, u0, tspan, p)
@btime auto_sen_l2($df, $u0, $tspan, $p)
#@btime auto_sen_l2($df, $u0, $tspan, $p; diffalg=$(ReverseDiff.gradient))
#@btime diffeq_sen_l2($df, $u0, $tspan, $p)

#=============================
ReverseDiff and adjoint_sensitivities are currently broken
julia> @btime auto_sen_l2($df, $u0, $tspan, $p)
  178.416 μs (757 allocations: 52.80 KiB)
3-element Array{Float64,1}:
 0.6219783212353721
 0.06800309430057831
 0.16918198800982667

julia> @btime auto_sen($df, $u0, $tspan, $p)
  184.872 μs (754 allocations: 52.71 KiB)
2×3 Array{Float64,2}:
  2.16057   0.188568   0.563195
 -6.25674  -0.697975  -1.70902
=============================#
