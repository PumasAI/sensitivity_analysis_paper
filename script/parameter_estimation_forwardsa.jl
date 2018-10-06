include("sensitivity.jl")
include("lotka-volterra.jl")
using Optim, BenchmarkTools, Random

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
t = collect(range(0, stop=10, length=200));
prob_original = ODEProblem(lvdf,u0,tspan,p)
data = solve(prob_original,Vern6(),abstol=1e-5,reltol=1e-7,saveat=t)

function l2loss(oursol_type, data)
    l2loss = zero(eltype(data))
    for j in 1:size(data)[2]
        for i in 1:size(data)[1]
            l2loss += (oursol_type[i,j] - data[i,j])^2
        end
    end
    l2loss
end

function costfunc(p,data,df)
    tmp_prob = ODEProblem(df,u0,tspan,p)
    sol = solve(tmp_prob,Vern6(),abstol=1e-5,reltol=1e-7,saveat=t)
    loss = l2loss(sol,data)
    loss
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
    costfunc(p,data,df)
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
res = optimize(p->costfunc(p,data,lvdf), (grad,p)->costfunc_gradient_comp(p,grad,lvcom_df,[u0;zeros(6)],tspan,data,t), lower, upper, x0, Fminbox(inner_optimizer));
@show Optim.minimizer(res), Optim.f_calls(res)
res = optimize(p->costfunc(p,data,lvdf), (grad,p)->costfunc_gradient_autosen(p,grad,lvdf,u0,tspan,data,t), lower, upper, x0, Fminbox(inner_optimizer));
@show Optim.minimizer(res), Optim.f_calls(res)
res = optimize(p->costfunc(p,data,lvdf), (grad,p)->costfunc_gradient_diffeq(p,grad,lvdf,u0,tspan,data,t), lower, upper, x0, Fminbox(inner_optimizer));
@show Optim.minimizer(res), Optim.f_calls(res)

Random.seed!(1)
@btime optimize(p->costfunc(p,$data,$lvdf), (grad,p)->costfunc_gradient_comp(p,grad,$lvcom_df,$([u0;zeros(6)]),$tspan,$data,$t), $lower, $upper, $x0, $(Fminbox(inner_optimizer)));
#  123.092 ms (303319 allocations: 41.61 MiB)
@btime optimize(p->costfunc(p,$data,$lvdf), (grad,p)->costfunc_gradient_autosen(p,grad,$lvdf,$u0,$tspan,$data,$t), $lower, $upper, $x0, $(Fminbox(inner_optimizer)));
#  135.546 ms (318034 allocations: 45.67 MiB)
@btime optimize(p->costfunc(p,$data,$lvdf), (grad,p)->costfunc_gradient_diffeq(p,grad,$lvdf,$u0,$tspan,$data,$t), $lower, $upper, $x0, $(Fminbox(inner_optimizer)));
#  300.071 ms (2277081 allocations: 141.48 MiB)
