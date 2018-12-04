using DelayDiffEq
p = [0.2,0.3,1,5,0.2,0.3,1,1,1,1,1,1]
function bc_model(du,u,h,p,t)
  p0,q0,v0,d0,p1,q1,v1,d1,d2,beta0,beta1,tau = p
  du[1] = (v0/(1+beta0*(h(p, t-tau)[3]^2))) * (p0 - q0)*u[1] - d0*u[1]
  du[2] = (v0/(1+beta0*(h(p, t-tau)[3]^2))) * (1 - p0 + q0)*u[1] +
          (v1/(1+beta1*(h(p, t-tau)[3]^2))) * (p1 - q1)*u[2] - d1*u[2]
  du[3] = (v1/(1+beta1*(h(p, t-tau)[3]^2))) * (1 - p1 + q1)*u[2] - d2*u[3]
end
lags = [1.0]
h(p, t) = ones(3)
tspan = (0.0,10.0)
u0 = [1.0,1.0,1.0]
prob = DDEProblem(bc_model,u0,h,tspan,p; constant_lags=lags)
sol = solve(prob,MethodOfSteps(Tsit5()))

function G(p)
  prob = DDEProblem(bc_model,eltype(p).(u0),h,tspan,p; constant_lags=lags)
  solve(prob,MethodOfSteps(Tsit5()))[end]
end

using ForwardDiff
ForwardDiff.jacobian(G,p)
using Calculus
Calculus.finite_difference_jacobian(G,p)
