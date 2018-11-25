using Optim, BenchmarkTools, Random
using Test
include("sensitivity.jl")

function forward_benchmark(fun, compfun, jac, u0, compu0, tspan, p, t, p0;
                           alg=Tsit5(), lower=0.5.*p, upper=1.5.*p, save_everystep=false,
                           verbose=false, iter=2, dropfirst=true, kwargs...)
  prob_original = ODEProblem(fun, u0, tspan, p)
  data = solve(prob_original, alg; saveat=t, save_everystep=save_everystep, kwargs...)
  function l2loss(sol, data)
    l2loss_ = zero(eltype(data))
    for j in 1:size(data, 2), i in 1:size(data, 1)
      l2loss_ += (sol[i,j] - data[i,j])^2
    end
    l2loss_
  end
  function costfunc(p,data,df,t,u0)
    tmp_prob = ODEProblem(df, u0, tspan, p)
    sol = solve(tmp_prob, alg; saveat=t, save_everystep=save_everystep, kwargs...)
    loss = l2loss(sol,data)
    verbose && @info "L2 Loss: $loss"
    return loss
  end
  function l2lossgradient!(grad,sol,data,sensitivities,num_p)
    fill!(grad,0.0)
    data_x_size = size(data,1)
    my_grad = @. -2 * (data - sol)
    u0len = length(data[1])
    K = size(my_grad,2)
    for k in 1:K, i in 1:num_p, j in 1:data_x_size
      grad[i] += my_grad[j,k]*sensitivities[i][j,k]
    end
  end
  function costfunc_gradient_diffeq(p,grad,df,u0,tspan,data,t;kwargs...)
    sol,sensitivities = diffeq_sen_full(df,u0,tspan,p,t; kwargs...)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
    costfunc(p,data,df,t,u0)
  end
  function costfunc_gradient_autosen(p,grad,df,u0,tspan,data,t;kwargs...)
    sol, sensitivities = auto_sen_full(df, u0, tspan, p, t; kwargs...)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
  end
  function costfunc_gradient_num(p,grad,df,u0,tspan,data,t;kwargs...)
    sol, sensitivities = numerical_sen_full(df, u0, tspan, p, t; kwargs...)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
  end
  function costfunc_gradient_comp(p,grad,comdf,u0,tspan,data,t; kwargs...)
    comprob = ODEProblem(comdf, u0, tspan, p)
    sol = reshape(vec(solve(comprob, alg; saveat=t, save_everystep=save_everystep, kwargs...)), length(u0), length(t))
    nvar = length(data[1])
    l2lossgradient!(grad,sol[1:nvar,:],data,[sol[i*nvar+1:i*nvar+nvar,:] for i in 1:length(p)], length(p))
  end
  inner_optimizer = BFGS()
  opt = Optim.Options(x_tol=1e-4, f_tol=1e-4, g_tol=3e-3)
  cost = let p=p, data=data, fun=fun, t=t, u0=u0
    p->costfunc(p,data,fun,t,u0)
  end
  forward_param_timings = let p=p, data=data, fun=fun, t=t, u0=u0, tspan=tspan, compu0=compu0, compfun=compfun, fun=fun
    t1, t2, t3, t4, t5, t6 = zeros(6)
    for i in 1:iter
      @info " Iteration $i"
      @info "  Running compile-time"
      t1 += @elapsed (s=optimize(
        cost,
        (grad,p)->costfunc_gradient_comp(
           p,grad,compfun,compu0,tspan,data,t; saveat=t, alg=alg, save_everystep=save_everystep,kwargs...),
        lower, upper, p0, (Fminbox(inner_optimizer)), opt); @test Optim.converged(s));
      @info "  Running DSA"
      t2 += @elapsed (s=optimize(
        cost,
        (grad,p)->costfunc_gradient_autosen(
          p,grad,fun,u0,tspan,data,t; saveat=t, alg=alg, save_everystep=save_everystep,kwargs...),
        lower, upper, p0, (Fminbox(inner_optimizer)), opt); @test Optim.converged(s));
      @info "  Running CSA with user-Jacobian"
      t3 += @elapsed (s=optimize(
        cost,
        (grad,p)->costfunc_gradient_diffeq(
          p,grad,ODEFunction(fun, jac=jac),u0,tspan,data,t; saveat=t, alg=alg, save_everystep=save_everystep,kwargs...),
        lower, upper, p0, (Fminbox(inner_optimizer)), opt); @test Optim.converged(s));
      @info "  Running CSA AD-Jacobian"
      t4 += @elapsed (s=optimize(
        cost,
        (grad,p)->costfunc_gradient_diffeq(
          p,grad,fun,u0,tspan,data,t; sensalg=SensitivityAlg(autojacvec=false),
          saveat=t, alg=alg, save_everystep=save_everystep,kwargs...),
        lower, upper, p0, (Fminbox(inner_optimizer)), opt); @test Optim.converged(s));
      @info "  Running CSA AD-Jv seeding"
      t5 += @elapsed (s=optimize(
        cost,
        (grad,p)->costfunc_gradient_diffeq(
          p,grad,fun,u0,tspan,data,t; sensalg=SensitivityAlg(autojacvec=true),
          saveat=t, alg=alg, save_everystep=save_everystep,kwargs...),
        lower, upper, p0, (Fminbox(inner_optimizer)), opt); @test Optim.converged(s));
      @info "  Running numerical differentiation"
      t6 += @elapsed (s=optimize(
        cost,
        (grad,p)->costfunc_gradient_num(
          p,grad,fun,u0,tspan,data,t; saveat=t, alg=alg, save_everystep=save_everystep,kwargs...),
        lower, upper, p0, (Fminbox(inner_optimizer)), opt); @test Optim.converged(s))
      if dropfirst && i == 1
        t1, t2, t3, t4, t5, t6 = zeros(6)
      end
    end
    num = dropfirst ? iter-1 : iter
    [t1, t2, t3, t4, t5, t6] ./ num
  end
end
