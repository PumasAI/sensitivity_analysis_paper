using Optim, BenchmarkTools, Random

param_lv = let
  include("sensitivity.jl")
  include("lotka-volterra.jl")
  u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
  t = collect(range(0, stop=10, length=200));
  prob_original = ODEProblem(lvdf,u0,tspan,p)
  data = solve(prob_original,Vern9(),abstol=1e-5,reltol=1e-7,saveat=t)

  function l2loss(oursol_type, data)
    l2loss_ = zero(eltype(data))
    for j in 1:size(data)[2]
      for i in 1:size(data)[1]
        l2loss_ += (oursol_type[i,j] - data[i,j])^2
      end
    end
    l2loss_
  end

  function costfunc(p,data,df,t,u0)
    tmp_prob = ODEProblem(df,u0,tspan,p)
    sol = solve(tmp_prob,Vern9(),abstol=1e-5,reltol=1e-7,saveat=t)
    l2loss(sol,data)
  end

  function l2lossgradient!(grad,sol,data,sensitivities,num_p)
    fill!(grad,0.0)
    data_x_size = size(data,1)
    my_grad = -1 .*2 .* (data .- sol)
    u0len = length(data[1])
    K = size(my_grad,2)
    @inbounds for k in 1:K
      for i in 1:num_p
        for j in 1:data_x_size
          grad[i] += my_grad[j,k]*sensitivities[i][j,k]
        end
      end
    end
  end

  function costfunc_gradient_diffeq(p,grad,df,u0,tspan,data,t)
    sol,sensitivities = diffeq_sen_full(df,u0,tspan,p,t)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
    costfunc(p,data,df,t,u0)
  end

  function costfunc_gradient_autosen(p,grad,df,u0,tspan,data,t)
    sol, sensitivities = auto_sen_full(df, u0, tspan, p, t)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
  end

  function costfunc_gradient_comp(p,grad,comdf,u0,tspan,data,t)
    comprob = ODEProblem(comdf, u0, tspan, p)
    sol = solve(comprob, Vern6(),abstol=1e-5,reltol=1e-7,saveat=t)
    nvar = length(data[1])
    l2lossgradient!(grad,sol[1:nvar,:],data,[sol[i*nvar+1:i*nvar+nvar,:] for i in 1:length(p)],length(p))
  end

  lower = [0.1,0.1,0.1]
  upper = [3.0,2.0,5.0]
  x0  = [0.5,0.5,0.5]
  inner_optimizer = LBFGS()
  res = optimize(p->costfunc(p,data,lvdf,t,u0),
                 (grad,p)->costfunc_gradient_comp(p,grad,lvcom_df,[u0;zeros(6)],tspan,data,t),
                 lower, upper, x0, Fminbox(inner_optimizer));
  @show Optim.minimizer(res), Optim.f_calls(res)
  res = optimize(p->costfunc(p,data,lvdf,t,u0),
                 (grad,p)->costfunc_gradient_autosen(p,grad,lvdf,u0,tspan,data,t),
                 lower, upper, x0, Fminbox(inner_optimizer));
  @show Optim.minimizer(res), Optim.f_calls(res)
  res = optimize(p->costfunc(p,data,lvdf,t,u0),
                 (grad,p)->costfunc_gradient_diffeq(p,grad,lvdf,u0,tspan,data,t),
                 lower, upper, x0, Fminbox(inner_optimizer));
  @show Optim.minimizer(res), Optim.f_calls(res)

  Random.seed!(1)
  @info "Running Lor"
  t1 = @belapsed optimize($(p->costfunc(p,data,lvdf,t,u0)), $((grad,p)->costfunc_gradient_comp(p,grad,lvcom_df,[u0;zeros(6)],tspan,data,t)),
                          $lower, $upper, $x0, $(Fminbox(inner_optimizer)));
  t2 = @belapsed optimize($(p->costfunc(p,data,lvdf,t,u0)), $((grad,p)->costfunc_gradient_autosen(p,grad,lvdf,u0,tspan,data,t)),
                          $lower, $upper, $x0, $(Fminbox(inner_optimizer)));
  t3 = @belapsed optimize($(p->costfunc(p,data,lvdf,t,u0)), $((grad,p)->costfunc_gradient_diffeq(p,grad,lvdf,u0,tspan,data,t)),
                          $lower, $upper, $x0, $(Fminbox(inner_optimizer)));
  [t1, t2, t3]
end

using CSV, DataFrames
let
  param_methods = ["Compile-time CSA", "DSA", "CSA"]
  param_timeings = DataFrame(methods=param_methods, LV=param_lv)
  bench_file_path = joinpath(@__DIR__, "..", "param_timings.csv")
  display(param_timeings)
  @info "Writing the benchmark results to $bench_file_path"
  CSV.write(bench_file_path, param_timeings)
end
