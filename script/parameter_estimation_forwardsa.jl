include("sensitivity.jl")
include("lotka-volterra.jl")
using NLopt

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

function costfunc_gradient_diffeq(p,grad,df,u0,tspan,data)
    sol,sensitivities = diffeq_sen_full(df,u0,tspan,p)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
    costfunc(p,data,df)
end

function costfunc_gradient_autosen(p,grad,df,u0,tspan,data)
    sol, sensitivities = auto_sen_full(df, u0, tspan, p)
    l2lossgradient!(grad,sol,data,sensitivities,length(p))
    costfunc(p,data,df)
end

function costfunc_gradient_comp(p,grad,comdf,u0,tspan,data,df)
    comprob = ODEProblem(comdf, u0, tspan, p)
    sol = solve(comprob, Vern6(),abstol=1e-5,reltol=1e-7,saveat=t)
    nvar = length(data[1])
    l2lossgradient!(grad,sol[1:nvar,:],data,[sol[i*nvar+1:i*nvar+nvar,:] for i in 1:length(p)],length(p))
    costfunc(p,data,df)
end

opt = Opt(:LD_MMA, 3)
lower_bounds!(opt,[0.1,0.1,0.1])
upper_bounds!(opt,[3.0,2.0,5.0])
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)

opt1 = deepcopy(opt)
min_objective!(opt, (p,grad)->costfunc_gradient_autosen(p,grad,lvdf,u0,tspan,data))
@time (minf,minx,ret) = NLopt.optimize(opt,[0.5,0.5,0.5])
println((minf,minx,ret))

opt2 = deepcopy(opt)
min_objective!(opt2, (p,grad)->costfunc_gradient_diffeq(p,grad,lvdf,u0,tspan,data))
@time (minf,minx,ret) = NLopt.optimize(opt2,[0.5,0.5,0.5])
println((minf,minx,ret))

opt3 = deepcopy(opt)
min_objective!(opt3, (p,grad)->costfunc_gradient_comp(p,grad,lvcom_df,[u0...;zeros(6)],tspan,data,lvdf))
@time (minf,minx,ret) = NLopt.optimize(opt3,[0.5,0.5,0.5])
println((minf,minx,ret))

#=
julia> include("parameter_estimation_forwardsa.jl")
  6.456667 seconds (16.57 M allocations: 874.876 MiB, 8.58% gc time)
(2.2409685972772607e-19, [1.5, 1.0, 3.0], :XTOL_REACHED)
  3.643371 seconds (8.69 M allocations: 462.276 MiB, 5.56% gc time)
(1.5920711791974714e-9, [1.5, 1.0, 3.0], :XTOL_REACHED)
  1.631290 seconds (3.20 M allocations: 175.343 MiB, 5.08% gc time)
(1.5920711791974714e-9, [1.5, 1.0, 3.0], :XTOL_REACHED)

julia> include("parameter_estimation_forwardsa.jl")
  5.095455 seconds (13.21 M allocations: 703.148 MiB, 10.32% gc time)
(2.2409685972772607e-19, [1.5, 1.0, 3.0], :XTOL_REACHED)
  2.262319 seconds (4.71 M allocations: 244.994 MiB, 3.95% gc time)
(1.5920711791974714e-9, [1.5, 1.0, 3.0], :XTOL_REACHED)
  1.647066 seconds (3.20 M allocations: 175.542 MiB, 4.88% gc time)
(1.5920711791974714e-9, [1.5, 1.0, 3.0], :XTOL_REACHED)
=#
