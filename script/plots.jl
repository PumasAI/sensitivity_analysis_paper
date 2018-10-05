using Plots

include("lotka-volterra.jl")

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
t = collect(range(0, stop=10, length=200));
sol, sensitivities = auto_sen_full(lvdf, u0, tspan, p)
p1 = plot(sol',xlabel = "Time",title="ODE Solution")
p2 = plot(sensitivities[2]',xlabel="Time",title="Sensitivity w.r.t. parameter \"b\" ")
fig = plot(p1,p2)
savefig(fig,"LVsol&sensb.png")

include("brusselator.jl")

bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(5)
sol_bruss, sensitivities_bruss = auto_sen_full(bfun, b_u0, (0.,10.), [3.4, 1., 10.])
p1_b = plot(sol_bruss',xlabel = "Time",title="ODE Solution",legend=false)
p2_b = plot(sensitivities_bruss[1]',xlabel="Time",title="Sensitivity w.r.t. parameter \"a\" ",legend=false)
fig_b = plot(p1_b,p2_b)
savefig(fig_b,"Brusssol&sensb.png")