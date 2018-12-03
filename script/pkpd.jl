using ParameterizedFunctions, OrdinaryDiffEq, LinearAlgebra

pkpdf = @ode_def begin
  dEv      = -Ka1*Ev
  dCent    =  Ka1*Ev - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp) - Q2*(Cent/Vc)  + Q2*(Periph2/Vp2)
  dPeriph  =  Q*(Cent/Vc)  - Q*(Periph/Vp)
  dPeriph2 =  Q2*(Cent/Vc)  - Q2*(Periph2/Vp2)
  dResp   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
end Ka1 CL Vc Q Vp Kin Kout IC50 IMAX γ Vmax Km Q2 Vp2

pkpdp = [
        1, # Ka1  Absorption rate constant 1 (1/time)
        1, # CL   Clearance (volume/time)
        20, # Vc   Central volume (volume)
        2, # Q    Inter-compartmental clearance (volume/time)
        10, # Vp   Peripheral volume of distribution (volume)
        10, # Kin  Response in rate constant (1/time)
        2, # Kout Response out rate constant (1/time)
        2, # IC50 Concentration for 50% of max inhibition (mass/volume)
        1, # IMAX Maximum inhibition
        1, # γ    Emax model sigmoidicity
        0, # Vmax Maximum reaction velocity (mass/time)
        2,  # Km   Michaelis constant (mass/volume)
        0.5, # Q2    Inter-compartmental clearance2 (volume/time)
        100 # Vp2   Peripheral2 volume of distribution (volume)
        ]

pkpdu0 = [100, 0, 0, 0, 5.]
pkpdcondition = function (u,t,integrator)
  t in 0:24:240
end
pkpdaffect! = function (integrator)
  integrator.u[1] += 100
end
pkpdcb = DiscreteCallback(pkpdcondition, pkpdaffect!, save_positions=(false, true))
pkpdtspan = (0.,240.)
pkpdprob = ODEProblem(pkpdf.f, pkpdu0, pkpdtspan, pkpdp)

pkpdfcomp = let pkpdf=pkpdf, J=zeros(5,5), JP=zeros(5,14), tmpdu=zeros(5,14), tmpu=zeros(5,14)
  function (du, u, p, t)
    vec(tmpu)  .= @view(u[6:end])
    pkpdf(@view(du[1:5]), u, p, t)
    pkpdf.jac(J,u,p,t)
    pkpdf.paramjac(JP,u,p,t)
    mul!(tmpdu, J, tmpu)
    du[6:end] .= vec(tmpdu) .+ vec(JP)
    nothing
  end
end
pkpdcompprob = ODEProblem(pkpdfcomp, [pkpdprob.u0;zeros(5*14)], pkpdprob.tspan, pkpdprob.p)
#sol = solve(pkpdprob, Tsit5(), tstops=0:24:240, callback=pkpdcb)
#plot(sol)
