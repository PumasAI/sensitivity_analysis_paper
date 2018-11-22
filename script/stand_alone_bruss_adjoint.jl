using OrdinaryDiffEq, ReverseDiff, ForwardDiff, Flux, DiffEqSensitivity
using LinearAlgebra
function auto_sen_l2(f, u0, tspan, p, t, alg=Tsit5(); diffalg=ReverseDiff.gradient, kwargs...)
  test_f(p) = begin
    prob = ODEProblem(f,convert.(eltype(p),u0),tspan,p)
    sol = solve(prob,alg,saveat=t; kwargs...)
    sum(sol.u) do x
      sum(z->(1-z)^2/2, x)
    end
  end
  diffalg(test_f, p)
end
@inline function diffeq_sen_l2(df, u0, tspan, p, t, alg=Tsit5();
                       abstol=1e-5, reltol=1e-7, iabstol=abstol, ireltol=reltol,
                       sensalg=SensitivityAlg(), kwargs...)
    prob = ODEProblem(df,u0,tspan,p)
    saveat = tspan[1] != t[1] && tspan[end] != t[end] ? vcat(tspan[1],t,tspan[end]) : t
    sol = solve(prob, alg, abstol=abstol, reltol=reltol, saveat=saveat; kwargs...)
    dg(out,u,p,t,i) = (out.=1.0.-u)
    adjoint_sensitivities(sol,alg,dg,t,abstol=abstol,
                          reltol=reltol,iabstol=abstol,ireltol=reltol,sensealg=sensalg)
end
function makebrusselator(N=8)
  xyd_brusselator = range(0,stop=1,length=N)
  function limit(a, N)
    if a == N+1
      return 1
    elseif a == 0
      return N
    else
      return a
    end
  end
  brusselator_f(x, y, t) = ifelse((((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) &&
                                  (t >= 1.1), 5., 0.)
  brusselator_2d_loop = let N = N, xyd=xyd_brusselator
    function brusselator_2d_loop(du, u, p, t)
      @inbounds begin
        ii1 = N^2*2
        ii2 = ii1+N^2
        ii3 = ii2+N^2
        α = @view p[1:ii1]
        A = @view p[ii1+1:ii2]
        B = @view p[ii2+1:ii3]
        dx = step(xyd)
        II = LinearIndices((N, N, 2))
        for I in CartesianIndices((N, N))
          i = I[1]
          j = I[2]
          x = xyd[i]
          y = xyd[j]
          ip1 = limit(i+1, N); im1 = limit(i-1, N)
          jp1 = limit(j+1, N); jm1 = limit(j-1, N)
          du[II[i,j,1]] = α[II[i,j,1]]*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2 +
          B[II[i,j,1]] + u[II[i,j,1]]^2*u[II[i,j,2]] - (A[II[i,j,1]] + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
        end
        for I in CartesianIndices((N, N))
          i = I[1]
          j = I[2]
          ip1 = limit(i+1, N)
          im1 = limit(i-1, N)
          jp1 = limit(j+1, N)
          jm1 = limit(j-1, N)
          du[II[i,j,2]] = α[II[i,j,2]]*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2 +
          A[II[i,j,1]]*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
        end
        return nothing
      end
    end
  end
  function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
      x = xyd[I[1]]
      y = xyd[I[2]]
      u[I,1] = 22*(y*(1-y))^(3/2)
      u[I,2] = 27*(x*(1-x))^(3/2)
    end
    vec(u)
  end
  u0 = init_brusselator_2d(xyd_brusselator)
  brusselator_2d_loop, u0, fill!(similar(u0, 4*N^2), 0.1)
end

Base.eps(::Type{Flux.Tracker.TrackedReal{T}}) where T = eps(T)
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v') # bad bad hack
function bench(ns=2:5, pre=true)
  alg = Rodas5(autodiff=false)
  println("Algorithm: $(nameof(typeof(alg)))")
  bt = 0:0.5:10
  tspan = (-0.1, 10.1)
  println("ts = $bt")
  bt = collect(bt)
  for n in ns
    bfun, b_u0, bp = makebrusselator(n)
    println("$(n)x$(n)x2:")
    pre && auto_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ForwardDiff.gradient);
    print("ForwardDiff:                        ")
    @time auto_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ForwardDiff.gradient);
    pre && auto_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ReverseDiff.gradient);
    print("ReverseDiff:                        ")
    @time auto_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ReverseDiff.gradient);
    #pre && auto_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=Flux.Tracker.gradient_);
    #print("Flux:                               ")
    #@time auto_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=Flux.Tracker.gradient_);
    pre && diffeq_sen_l2(bfun, b_u0, tspan, bp, bt, alg);
    print("Continuous SA with vec'Jac seeding: ")
    @time diffeq_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=true));
    pre && diffeq_sen_l2(bfun, b_u0, (-.01,10.01), bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
    print("Continuous SA:                      ")
    @time diffeq_sen_l2(bfun, b_u0, tspan, bp, bt, alg, abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
  end
end
#=
Algorithm: Rodas5
ts = 0.0:0.5:10.0
2x2x2:
ForwardDiff:                          0.002463 seconds (2.34 k allocations: 198.031 KiB)
ReverseDiff:                          0.143720 seconds (660.05 k allocations: 23.851 MiB, 32.02% gc time)
Continuous SA with vec'Jac seeding:   0.190598 seconds (487.94 k allocations: 23.980 MiB, 7.98% gc time)
Continuous SA:                        0.037339 seconds (69.21 k allocations: 1.560 MiB, 55.05% gc time)
3x3x2:
ForwardDiff:                          0.030961 seconds (6.81 k allocations: 816.359 KiB)
ReverseDiff:                          2.082692 seconds (6.43 M allocations: 230.579 MiB, 52.18% gc time)
Continuous SA with vec'Jac seeding:   0.388259 seconds (166.64 k allocations: 3.680 MiB)
Continuous SA:                        0.186917 seconds (158.41 k allocations: 3.508 MiB)
4x4x2:
ForwardDiff:                          0.108875 seconds (11.91 k allocations: 2.723 MiB)
ReverseDiff:                          4.454240 seconds (26.82 M allocations: 958.981 MiB, 15.27% gc time)
Continuous SA with vec'Jac seeding:   3.135371 seconds (435.71 k allocations: 10.082 MiB)
Continuous SA:                        2.281915 seconds (412.34 k allocations: 9.577 MiB)
5x5x2:
ForwardDiff:                          0.974421 seconds (17.59 k allocations: 7.217 MiB, 34.48% gc time)
ReverseDiff:                         15.934102 seconds (61.90 M allocations: 2.146 GiB, 48.37% gc time)
Continuous SA with vec'Jac seeding:   4.758021 seconds (358.61 k allocations: 8.022 MiB)
Continuous SA:                        3.983128 seconds (338.24 k allocations: 7.589 MiB)
=#
