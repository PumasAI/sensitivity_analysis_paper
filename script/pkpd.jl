using ParameterizedFunctions, OrdinaryDiffEq, LinearAlgebra

pkpdf = @ode_def begin
  dEv      = -Ka1*Ev
  dCent    =  Ka1*Ev - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp) - Q2*(Cent/Vc)  + Q2*(Periph2/Vp2)
  dPeriph  =  Q*(Cent/Vc)  - Q*(Periph/Vp)
  dPeriph2 =  Q2*(Cent/Vc)  - Q2*(Periph2/Vp2)
  dResp   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
end Ka1 CL Vmax Km Vc Q Q2 Vp Vp2 Kin Kout IC50 IMAX γ

pkpdp = 0.2.*ones(14)

pkpdu0 = [0.1,0.1,0.1,0.1,0.1]
pkpdcondition = function (u,t,integrator)
  t in 1:2:49
end
pkpdaffect! = function (integrator)
  integrator.u[1] += 0.1
end
pkpdcb = DiscreteCallback(pkpdcondition, pkpdaffect!)
pkpdtspan = (0.,50.)
pkpdprob = ODEProblem(pkpdf.f, pkpdu0, pkpdtspan, pkpdp)

pkpdfcomp = let pkpdf=pkpdf
  function (du, u, p, t)
    p, J, JP, tmpdu, tmpu = p
    vec(tmpu)  .= @view(u[6:end])
    pkpdf(@view(du[1:5]), u, p, t)
    pkpdf.jac(J,u,p,t)
    pkpdf.paramjac(JP,u,p,t)
    mul!(tmpdu, J, tmpu)
    du[6:end] .= vec(tmpdu) .+ vec(JP)
    nothing
  end
end
pkpdcompprob = ODEProblem(pkpdfcomp, [pkpdprob.u0;zeros(5*14)], pkpdprob.tspan, (pkpdprob.p,zeros(5,5), zeros(5,14), zeros(5,14), zeros(5,14)))
#sol = solve(pkpdprob, Tsit5(), tstops=1:2:49, callback=pkpdcb)
#plot(sol)
