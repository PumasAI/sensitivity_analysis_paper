using Plots

include("sensitivity.jl")
include("lotka-volterra.jl")
include("brusselator.jl")

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
t = collect(range(0, stop=10, length=200));
sol, sensitivities_ad = auto_sen_full(lvdf, u0, tspan, p)
s,sensitivities_diff = diffeq_sen_full(lvdf,u0,tspan,p)
p1 = plot(sol',xlabel = "Time",title="ODE Solution",legend=false)
p2 = plot(sensitivities_ad[1]',xlabel="Time",title="AD",legend=false)
p3 = plot(sensitivities_diff[1]',xlabel="Time",title="Diffeq",legend=false)
bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(5)
sol_bruss, sensitivities_bruss_ad = auto_sen_full(bfun, b_u0, (0.,10.), [3.4, 1., 10.])
s_b, sensitivities_bruss_diff = diffeq_sen_full(bfun, b_u0, (0.,10.), [3.4, 1., 10.])
p1_b = plot(sol_bruss',xlabel = "Time",title="ODE Solution",legend=false)
p2_b = plot(sensitivities_bruss_ad[1]',xlabel="Time",title="AD",legend=false)
p3_b = plot(sensitivities_bruss_diff[1]',xlabel="Time",title="Diffeq",legend=false)
fig1 = plot(p1,p2,p3,plot_title="LotkaVolterra",layout=(1,3))
fig2 = plot(p1_b,p2_b,p3_b,plot_title="Brusselator",layout=(1,3))
fig = plot(fig1,fig2,layout=(2,1),size=(600,600))
savefig(fig,"plot.png")