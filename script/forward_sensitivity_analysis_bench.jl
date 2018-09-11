# ===============================================================
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays, Profile, ProfileView

function f(u, p, t)
    a,b,c = p
    x, y = u
    dx = a*x - b*x*y
    dy = -c*y + x*y
    @SVector [dx, dy]
end

function df(du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

u0s = @SVector [1.,1.]; u0 = [1.,1.]; tspan = (0., 10.); sp = @SVector [1.5,1.0,3.0]; p = [1.5,1.0,3.0]

function diffeq_sen(f, init, tspan, p)
    prob = ODELocalSensitivityProblem(f,init,tspan,p)
    sol = solve(prob,Vern6(),save_everystep=false,abstol=1e-5,reltol=1e-7)
    extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, init, tspan, p)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),eltype(p).(tspan),p)
        solve(prob,Vern6(),save_everystep=false,abstol=1e-5,reltol=1e-7)[end]
    end
    ForwardDiff.jacobian(test_f, p)
end
@time auto_sen(f, u0s, tspan, sp)
@time auto_sen(df, u0, tspan, p)
@time diffeq_sen(df, u0, tspan, p)
@btime auto_sen($f, $u0s, $tspan, $sp)
@btime auto_sen($df, $u0, $tspan, $p)
@btime diffeq_sen($df, $u0, $tspan, $p)

# Profile
#=
auto_sen(df, u0, tspan, p)
Profile.clear()
@profile for i in 1:500
    auto_sen(df, u0, tspan, p)
end
ProfileView.svgwrite("autodiff_profile.svg")

diffeq_sen(df, u0, tspan, p)
Profile.clear()
@profile for i in 1:500
    diffeq_sen(df, u0, tspan, p)
end
ProfileView.svgwrite("diffeq_profile.svg")
=#

# ===============================================================
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays, Profile, ProfileView
using LinearAlgebra

const D = Tridiagonal(rand(99), rand(100), rand(99))
function df(du, u, p, t)
    mul!(du, D, u)
    @. du += p
end
u0 = rand(100); tspan = (0., 0.4); p = rand(100);

function diffeq_sen(f, init, tspan, p, alg=Vern6())
    prob = ODELocalSensitivityProblem(f,init,tspan,p)
    sol = solve(prob,alg,save_everystep=false,abstol=1e-5,reltol=1e-7)
    extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, init, tspan, p, alg=Vern6())
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),eltype(p).(tspan),p)
        solve(prob,alg,save_everystep=false,abstol=1e-5,reltol=1e-7)[end]
    end
    ForwardDiff.jacobian(test_f, p)
end
DiffEqBase.has_syms(::DiffEqSensitivity.ODELocalSensitvityFunction) = false
DiffEqBase.has_tgrad(::DiffEqSensitivity.ODELocalSensitvityFunction) = false
DiffEqBase.has_invW(::DiffEqSensitivity.ODELocalSensitvityFunction) = false
@time auto_sen(df, u0, tspan, p, Rodas5(autodiff=false))[1:3]
@time diffeq_sen(df, u0, tspan, p, Rodas5(autodiff=false))[1][1:3]
@time auto_sen(df, u0, tspan, p, Rodas5(autodiff=false));
@time diffeq_sen(df, u0, tspan, p, Rodas5(autodiff=false));

@time auto_sen(df, u0, tspan, p)[1:3]
@time diffeq_sen(df, u0, tspan, p)[1][1:3]
@btime auto_sen($df, $u0, $tspan, $p);
@btime diffeq_sen($df, $u0, $tspan, $p);
