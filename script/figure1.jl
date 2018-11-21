using Plots
pgfplots()

include("sensitivity.jl")
include("lotka-volterra.jl")
include("brusselator.jl")

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
t = collect(range(0, stop=10, length=200));
sol, sensitivities_ad = auto_sen_full(lvdf, u0, tspan, p, t)
s,sensitivities_diff = diffeq_sen_full(lvdf,u0,tspan,p,t)
maximum(abs, vcat(map((x,y)->x.-y, sensitivities_ad, sensitivities_diff)...))
# 2.7513488653596596e-5
sol_labels = [raw"$x$" raw"$y$"]
sens_labels = [raw"$\frac{\partial x}{\partial p_1}$" raw"$\frac{\partial y}{\partial p_1}$"]
p1 = plot(t,sol',label=sol_labels,xlabel="time",ylabel="Lotka-Volterra",legend=:topleft,title="ODE Solution")
p2 = plot(t,sensitivities_ad[1]',label=sens_labels,xlabel="time",legend=:topleft,title="DSAAD")
p3 = plot(t,sensitivities_diff[1]',label=sens_labels,xlabel="time",legend=:topleft,title="Continuous SA")

bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(5)
sol_bruss, sensitivities_bruss_ad = auto_sen_full(bfun, b_u0, (0.,10.), [3.4, 1., 10.], t)
s_b, sensitivities_bruss_diff = diffeq_sen_full(bfun, b_u0, (0.,10.), [3.4, 1., 10.], t)
maximum(abs, vcat(map((x,y)->x.-y, sensitivities_bruss_ad, sensitivities_bruss_diff)...))
# 0.010689547914623933
sol_labels = [raw"$u_{11}$" raw"$v_{11}$"]
sens_labels = [raw"$\frac{\partial u_{11}}{\partial p_1}$" raw"$\frac{\partial v_{11}}{\partial p_1}$"]
p1_b = plot(t,sol_bruss'[:,[1,26]],label=sol_labels,xlabel = "time",ylabel="Brusselator",legend=:topleft)
p2_b = plot(t,sensitivities_bruss_ad[1]'[:,[1,26]],label=sens_labels,xlabel="time",legend=:topleft)
p3_b = plot(t,sensitivities_bruss_diff[1]'[:,[1,26]],label=sens_labels,xlabel="time",legend=:topleft)

fig1 = plot(p1,p2,p3,layout=(1,3))
fig2 = plot(p1_b,p2_b,p3_b,layout=(1,3))
fig = plot(fig1,fig2,layout=(2,1),size=(1200,800))
savefig(fig,"solution_sensitivty_plot.pdf")
