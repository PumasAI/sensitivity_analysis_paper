using Plots; pgfplots()
eval.(Meta.parse.(split(String(read(joinpath(@__DIR__, "..", "bruss_scaling_data.txt"))), '\n')))

plt = plot(title="Brusselator Scaling");
plot!(plt, forwarddiffn, forwarddiff, lab="Forward-Mode DSAAD");
plot!(plt, reversediffn, reversediff, lab="Reverse-Mode DSAAD");
plot!(plt, csan, csa, lab="CASA AD-Jacobian");
plot!(plt, csaseedn, csaseed, lab=raw"CASA AD-$v^{T}J$ seeding");
plot!(plt, numdiffn, numdiff, lab="Numerical Differentiation");
plot!(plt, legend=:bottomright);
xaxis!(plt, "Dimension");
yaxis!(plt, "Runtime (s)", :log10);
savefig(plt, "figure2.pdf")
