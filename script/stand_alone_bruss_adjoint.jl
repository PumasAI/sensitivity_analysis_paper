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
  for n in ns
    bt = 0:0.5:1
    tspan = (-0.001, 1.001)
    bfun, b_u0, bp = makebrusselator(n)
    println("$(n)x$(n)x2:")
    pre && auto_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ForwardDiff.gradient);
    print("ForwardDiff:                        ")
    @time auto_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ForwardDiff.gradient);
    pre && auto_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ReverseDiff.gradient);
    print("ReverseDiff:                        ")
    @time auto_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=ReverseDiff.gradient);
    pre && auto_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=Flux.Tracker.gradient_);
    print("Flux:                               ")
    @time auto_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, diffalg=Flux.Tracker.gradient_);
    pre && diffeq_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=true));
    print("Continuous SA with vec'Jac seeding: ")
    @time diffeq_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=true));
    pre && diffeq_sen_l2(bfun, b_u0, (-.01,10.01), bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
    print("Continuous SA:                      ")
    @time diffeq_sen_l2(bfun, b_u0, tspan, bp, bt, Rosenbrock23(autodiff=false), abstol=1e-5,reltol=1e-7, save_everystep=false, sensalg=SensitivityAlg(autojacvec=false));
  end
end
#=
2x2x2:
ForwardDiff:                          0.001018 seconds (1.40 k allocations: 127.969 KiB)
ReverseDiff:                          0.028679 seconds (336.38 k allocations: 12.194 MiB)
Flux:                                 0.050367 seconds (476.76 k allocations: 15.004 MiB, 44.69% gc time)
Continuous SA with vec'Jac seeding:   0.042142 seconds (460.32 k allocations: 16.641 MiB, 20.04% gc time)
Continuous SA:                        0.005899 seconds (31.81 k allocations: 930.484 KiB)
3x3x2:
ForwardDiff:                          0.015531 seconds (5.13 k allocations: 608.781 KiB)
ReverseDiff:                          0.939547 seconds (9.50 M allocations: 333.775 MiB, 34.88% gc time)
Flux:                                 0.899117 seconds (9.06 M allocations: 306.411 MiB, 42.57% gc time)
Continuous SA with vec'Jac seeding:   2.175091 seconds (20.16 M allocations: 727.885 MiB, 10.08% gc time)
Continuous SA:                        0.088874 seconds (398.24 k allocations: 7.954 MiB)
4x4x2:
ForwardDiff:                          0.108623 seconds (9.35 k allocations: 2.184 MiB)
ReverseDiff:                          3.102426 seconds (32.67 M allocations: 1.130 GiB, 30.52% gc time)
Flux:                                 9.245650 seconds (33.83 M allocations: 1.217 GiB, 77.44% gc time)
Continuous SA with vec'Jac seeding:  10.001870 seconds (66.19 M allocations: 2.362 GiB, 22.93% gc time)
Continuous SA:                        0.392478 seconds (734.77 k allocations: 14.349 MiB, 5.59% gc time)
5x5x2:
ForwardDiff:                          0.474690 seconds (13.80 k allocations: 6.123 MiB, 2.57% gc time)
ReverseDiff:                         17.901871 seconds (104.61 M allocations: 3.624 GiB, 59.06% gc time)
Flux:                                26.580947 seconds (100.33 M allocations: 3.818 GiB, 74.91% gc time)
=#
