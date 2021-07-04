using Plots; gr()
eval.(Meta.parse.(split(String(read(joinpath(@__DIR__, "..", "bruss_scaling_data.txt"))), '\n')))

n_to_param(n) = 4n^2

plt = plot(title="Sensitivity Scaling on Brusselator");
plot!(plt, n_to_param.(forwarddiffn), forwarddiff, lab="Forward-Mode DSAAD", lw=3);
plot!(plt, n_to_param.(reversediffn), reversediff, lab="Reverse-Mode DSAAD", lw=3);
csadata = [[csa[j][i] for j in eachindex(csa)] for i in eachindex(csa[1])]
plot!(plt, n_to_param.(csan), csadata[1], lab="Interpolating CASA user-Jacobian", lw=3);
plot!(plt, n_to_param.(csan), csadata[2], lab="Interpolating CASA AD-Jacobian", lw=3);
plot!(plt, n_to_param.(csan), csadata[3], lab=raw"Interpolating CASA AD-$v^{T}J$ seeding", lw=3);
plot!(plt, n_to_param.(csan), csadata[1+3], lab="Quadrature CASA user-Jacobian", lw=3);
plot!(plt, n_to_param.(csan), csadata[2+3], lab="Quadrature CASA AD-Jacobian", lw=3);
plot!(plt, n_to_param.(csan), csadata[3+3], lab=raw"Quadrature CASA AD-$v^{T}J$ seeding", lw=3);
plot!(plt, n_to_param.(numdiffn), numdiff, lab="Numerical Differentiation", lw=3);
plot!(plt, legend=:bottomright);
xaxis!(plt, "Number of Parameters", :log10);
yaxis!(plt, "Runtime (s)", :log10);
savefig(plt, "../figure2.pdf")
