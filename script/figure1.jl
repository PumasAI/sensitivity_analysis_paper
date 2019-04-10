using Plots
pgfplots()

include("sensitivity.jl")
include("lotka-volterra.jl")
include("brusselator.jl")
DiffEqBase.has_tgrad(::ODELocalSensitivityFunction) = false
DiffEqBase.has_invW(::ODELocalSensitivityFunction) = false
DiffEqBase.has_jac(::ODELocalSensitivityFunction) = false

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
t = collect(range(0, stop=10, length=200));
sol, sensitivities_ad = auto_sen_full(lvdf, u0, tspan, p, t, alg=Tsit5(), reltol=1e-7, abstol=1e-7)
s,sensitivities_diff = diffeq_sen_full(lvdf, u0, tspan, p, t, alg=Tsit5(), reltol=1e-7, abstol=1e-7)
maximum(abs, vcat(map((x,y)->x.-y, sensitivities_ad, sensitivities_diff)...))
# 1.1427628923144084e-5
sol_labels = [raw"$x$" raw"$y$"]
sens_labels = [raw"$\frac{\partial x}{\partial p_1}$" raw"$\frac{\partial y}{\partial p_1}$"]
p1 = plot(t,sol',label=sol_labels,xlabel="time",ylabel="Lotka-Volterra",legend=:topleft,title="ODE Solution");
p2 = plot(t,sensitivities_ad[1]',label=sens_labels,xlabel="time",legend=:topleft,title="DSAAD");
p3 = plot(t,sensitivities_diff[1]',label=sens_labels,xlabel="time",legend=:topleft,title="Continuous SA");

n = 3
bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
sol_bruss, sensitivities_bruss_ad = auto_sen_full(bfun, b_u0, (0.,10.), b_p, t, alg=Rodas5(), reltol=1e-7, abstol=1e-7)
s_b, sensitivities_bruss_diff = diffeq_sen_full(bfun, b_u0, (0.,10.), b_p, t, alg=Rodas5(autodiff=false), reltol=1e-7, abstol=1e-7)
maximum(abs, vcat(map((x,y)->x.-y, sensitivities_bruss_ad, sensitivities_bruss_diff)...))
# 0.00031005546364148984
sol_labels = [raw"$u_{11}$" raw"$v_{11}$"]
sens_labels = [raw"$\frac{\partial u_{11}}{\partial p_1^{11}}$" raw"$\frac{\partial v_{11}}{\partial p_1^{11}}$"]
p1_b = plot(t,sol_bruss'[:,[1,n^2+1]],label=sol_labels,xlabel = "time",ylabel="Brusselator",legend=:topleft);
p2_b = plot(t,sensitivities_bruss_ad[1]'[:,[1,n^2+1]],label=sens_labels,xlabel="time",legend=:topleft);
p3_b = plot(t,sensitivities_bruss_diff[1]'[:,[1,n^2+1]],label=sens_labels,xlabel="time",legend=:topleft);

fig1 = plot(p1,p2,p3,layout=(1,3));
fig2 = plot(p1_b,p2_b,p3_b,layout=(1,3));
fig = plot(fig1,fig2,layout=(2,1),size=(1200,800));
savefig(fig,"solution_sensitivty_plot.pdf")
