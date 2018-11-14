using ParameterizedFunctions, LinearAlgebra
pollution = @ode_def begin
  dy1  = -k1 *y1-k10*y11*y1-k14*y1*y6-k23*y1*y4-k24*y19*y1+
        k2 *y2*y4+k3 *y5*y2+k9 *y11*y2+k11*y13+k12*y10*y2+k22*y19+k25*y20
  dy2  = -k2 *y2*y4-k3 *y5*y2-k9 *y11*y2-k12*y10*y2+k1 *y1+k21*y19
  dy3  = -k15*y3+k1 *y1+k17*y4+k19*y16+k22*y19
  dy4  = -k2 *y2*y4-k16*y4-k17*y4-k23*y1*y4+k15*y3
  dy5  = -k3 *y5*y2+k4 *y7+k4 *y7+k6 *y7*y6+k7 *y9+k13*y14+k20*y17*y6
  dy6  = -k6 *y7*y6-k8 *y9*y6-k14*y1*y6-k20*y17*y6+k3 *y5*y2+k18*y16+k18*y16
  dy7  = -k4 *y7-k5 *y7-k6 *y7*y6+k13*y14
  dy8  = k4 *y7+k5 *y7+k6 *y7*y6+k7 *y9
  dy9  = -k7 *y9-k8 *y9*y6
  dy10 = -k12*y10*y2+k7 *y9+k9 *y11*y2
  dy11 = -k9 *y11*y2-k10*y11*y1+k8 *y9*y6+k11*y13
  dy12 = k9 *y11*y2
  dy13 = -k11*y13+k10*y11*y1
  dy14 = -k13*y14+k12*y10*y2
  dy15 = k14*y1*y6
  dy16 = -k18*y16-k19*y16+k16*y4
  dy17 = -k20*y17*y6
  dy18 = k20*y17*y6
  dy19 = -k21*y19-k22*y19-k24*y19*y1+k23*y1*y4+k25*y20
  dy20 = -k25*y20+k24*y19*y1
end k1  k2  k3  k4  k5  k6  k7  k8  k9  k10  k11  k12  k13  k14  k15  k16  k17  k18  k19  k20  k21  k22  k23  k24  k25
function make_pollution()
  comp = let pollution = pollution
    function comp(du, u, p, t)
      p, J, JP, tmpdu, tmpu = p
      tmpu  .= @view( u[:, 2:26])
      pollution(@view(du[:, 1]), u, p, t)
      pollution.jac(J,u,p,t)
      pollution.paramjac(JP,u,p,t)
      mul!(tmpdu, J, tmpu)
      du[:,2:26] .= tmpdu .+ JP
      nothing
    end
  end

  u0 = zeros(20)
  p = [.35e0, .266e2, .123e5, .86e-3, .82e-3, .15e5, .13e-3, .24e5, .165e5, .9e4, .22e-1, .12e5, .188e1, .163e5, .48e7, .35e-3, .175e-1, .1e9, .444e12, .124e4, .21e1, .578e1, .474e-1, .178e4, .312e1]
  u0[2]  = 0.2
  u0[4]  = 0.04
  u0[7]  = 0.1
  u0[8]  = 0.3
  u0[9]  = 0.01
  u0[17] = 0.007
  compu0 = zeros(20, 26)
  compu0[1:20] .= u0
  comp, u0, p, compu0
end
