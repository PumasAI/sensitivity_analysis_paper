include("sensitivity.jl")
include("forward_sensitivity_analysis_bench.jl")
using NLopt

t = collect(range(0, stop=10, length=200));
prob_original = ODEProblem(df,u0,tspan,p)
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

function diffeq_sen(f, init, tspan, p)
    prob = ODELocalSensitivityProblem(f,init,tspan,p)
    sol = solve(prob,Vern6(),abstol=1e-5,reltol=1e-7,saveat=t)
    extract_local_sensitivities(sol)
end

costfunc = function (p)
    tmp_prob = ODEProblem(df,u0,tspan,p)
    sol = solve(tmp_prob,Vern6(),abstol=1e-5,reltol=1e-7,saveat=t)
    loss = l2loss(sol,data)
    loss
end

function auto_sen(f, init, tspan, p)
    test_f(p) = begin
        prob = ODEProblem(f,eltype(p).(init),tspan,p)
        vec(solve(prob,Vern6(),saveat=t,abstol=1e-5,reltol=1e-7))
    end
    sens = ForwardDiff.jacobian(test_f, p)
    [reshape(sens[:,i]',length(init),length(t)) for i in 1:length(p)]
end

function l2lossgradient!(grad,oursol,data,sensitivities,num_p)
    fill!(grad,0.0)
    data_x_size = size(data)[1]
    my_grad = -1 .*2 .* (data .- oursol)
    for k in 1:size(my_grad)[2]
        for i in 1:num_p
            for j in 1:data_x_size
                grad[i] += my_grad[j,k]*sensitivities[i][j,k]
            end
        end
    end
end

function costfunc_gradient_diffeq(p,grad)
    sol,sensitivities = diffeq_sen_full(df,u0,tspan,p)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
    costfunc(p)
end

function costfunc_gradient_autosen(p,grad)
    prob = ODEProblem(f,u0,tspan,p)
    sol = solve(prob,Vern6(),saveat=t,abstol=1e-5,reltol=1e-7)
    sensitivities = auto_sen_full(df, u0, tspan, p)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
    costfunc(p)
end

function costfunc_gradient_comp(p,grad)
    com_u0 = [u0...;zeros(6)]
    comprob = ODEProblem(com_df, com_u0, tspan, p)
    sol = solve(comprob, Vern9(),abstol=1e-5,reltol=1e-7,save_everystep=false)
    l2lossgradient!(grad,sol[1:length(u0)],data,sol[length(u0)+1:end],length(p))
    costfunc(p)
end

opt = Opt(:LD_MMA, 3)
lower_bounds!(opt,[0.1,0.1,0.1])
upper_bounds!(opt,[3.0,2.0,5.0])
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)

min_objective!(opt, costfunc_gradient_autosen)
(minf,minx,ret) = NLopt.optimize(opt,[0.5,0.5,0.5])

min_objective!(opt, costfunc_gradient_diffeq)
(minf,minx,ret) = NLopt.optimize(opt,[0.5,0.5,0.5])

min_objective!(opt, costfunc_gradient_comp)
(minf,minx,ret) = NLopt.optimize(opt,[0.5,0.5,0.5])

