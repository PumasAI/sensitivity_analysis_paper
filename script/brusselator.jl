using LinearAlgebra
function makebrusselator(N=8)
  xyd_brusselator = range(0,stop=1,length=N)
  @inline function limit(a, N)
    if a == N+1
      return 1
    elseif a == 0
      return N
    else
      return a
    end
  end
  brusselator_f(x, y, t) = ifelse((((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) &&
                                  (t >= 1.1), 5., 0.)
  kernel_u! = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
    @inline function (du, u, A, B, α, II, I, t)
      i, j = Tuple(I)
      x = xyd[I[1]]
      y = xyd[I[2]]
      ip1 = limit(i+1, N); im1 = limit(i-1, N)
      jp1 = limit(j+1, N); jm1 = limit(j-1, N)
      du[II[i,j,1]] = α[II[i,j,1]]*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2 +
      B[II[i,j,1]] + u[II[i,j,1]]^2*u[II[i,j,2]] - (A[II[i,j,1]] + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
    end
  end
  kernel_v! = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
    @inline function (du, u, A, B, α, II, I, t)
      i, j = Tuple(I)
      ip1 = limit(i+1, N)
      im1 = limit(i-1, N)
      jp1 = limit(j+1, N)
      jm1 = limit(j-1, N)
      du[II[i,j,2]] = α[II[i,j,2]]*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2 +
      A[II[i,j,1]]*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
    end
  end
  kernel_u_oop! = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
    @inline function (u, A, B, α, II, I, t)
      i, j = Tuple(I)
      x = xyd[I[1]]
      y = xyd[I[2]]
      ip1 = limit(i+1, N); im1 = limit(i-1, N)
      jp1 = limit(j+1, N); jm1 = limit(j-1, N)
      α[II[i,j,1]]*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2 +
      B[II[i,j,1]] + u[II[i,j,1]]^2*u[II[i,j,2]] - (A[II[i,j,1]] + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
    end
  end
  kernel_v_oop! = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
    @inline function (u, A, B, α, II, I, t)
      i, j = Tuple(I)
      ip1 = limit(i+1, N)
      im1 = limit(i-1, N)
      jp1 = limit(j+1, N)
      jm1 = limit(j-1, N)
      α[II[i,j,2]]*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2 +
      A[II[i,j,1]]*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
    end
  end
  brusselator_2d = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
    function (du, u, p, t)
      @inbounds begin
        ii1 = N^2
        ii2 = ii1+N^2
        ii3 = ii2+2(N^2)
        A = @view p[1:ii1]
        B = @view p[ii1+1:ii2]
        α = @view p[ii2+1:ii3]
        II = LinearIndices((N, N, 2))
        kernel_u!.(Ref(du), Ref(u), Ref(A), Ref(B), Ref(α), Ref(II), CartesianIndices((N, N)), t)
        kernel_v!.(Ref(du), Ref(u), Ref(A), Ref(B), Ref(α), Ref(II), CartesianIndices((N, N)), t)
        return nothing
      end
    end
  end
  brusselator_2d_oop = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
    function (u, p, t)
      @inbounds begin
        ii1 = N^2
        ii2 = ii1+N^2
        ii3 = ii2+2(N^2)
        A = @view p[1:ii1]
        B = @view p[ii1+1:ii2]
        α = @view p[ii2+1:ii3]
        II = LinearIndices((N, N, 2))
        Flux.Tracker.collect(vec(cat(kernel_u_oop!.(Ref(u), Ref(A), Ref(B), Ref(α), Ref(II), CartesianIndices((N, N)), t),
            kernel_v_oop!.(Ref(u), Ref(A), Ref(B), Ref(α), Ref(II), CartesianIndices((N, N)), t),
            dims=3)))
      end
    end
  end
  function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
      x = xyd[I[1]]
      y = xyd[I[2]]
      u[I,1] = 22*(y*(1-y))^(3/2)
      u[I,2] = 27*(x*(1-x))^(3/2)
    end
    vec(u)
  end
  dx = step(xyd_brusselator)
  e1 = ones(N-1)
  off = N-1
  e4 = ones(N-off)
  T = diagm(0=>-2ones(N), -1=>e1, 1=>e1, off=>e4, -off=>e4) ./ dx^2
  Ie = Matrix{Float64}(I, N, N)
  # A + df/du
  Op = kron(Ie, T) + kron(T, Ie)
  brusselator_jac = let N=N, Op=Op
    (J,a,p,t) -> begin
      ii1 = N^2
      ii2 = ii1+N^2
      ii3 = ii2+2(N^2)
      A = @view p[1:ii1]
      B = @view p[ii1+1:ii2]
      α = @view p[ii2+1:ii3]
      u = @view a[1:end÷2]
      v = @view a[end÷2+1:end]
      N2 = length(a)÷2
      α1 = @view α[1:end÷2]
      α2 = @view α[end÷2+1:end]
      fill!(J, 0)

      J[1:N2, 1:N2] .= α1.*Op
      J[N2+1:end, N2+1:end] .= α2.*Op

      J1 = @view J[1:N2,     1:N2]
      J2 = @view J[N2+1:end, 1:N2]
      J3 = @view J[1:N2,     N2+1:end]
      J4 = @view J[N2+1:end, N2+1:end]
      J1[diagind(J1)] .+= @. 2u*v-(A+1)
      J2[diagind(J2)] .= @. A-2u*v
      J3[diagind(J3)] .= @. u^2
      J4[diagind(J4)] .+= @. -u^2
      nothing
    end
  end
  brusselator_jac_oop = let N=N, Op=Op
    (a,p,t) -> begin
      ii1 = N^2
      ii2 = ii1+N^2
      ii3 = ii2+2(N^2)
      A = @view p[1:ii1]
      B = @view p[ii1+1:ii2]
      α = @view p[ii2+1:ii3]
      u = @view a[1:end÷2]
      v = @view a[end÷2+1:end]
      N2 = length(a)÷2
      α1 = @view α[1:end÷2]
      α2 = @view α[end÷2+1:end]

      J11 = α1.*Op
      J22 = α2.*Op
      J11 += diagm(0=>@.(2u*v-(A+1)))
      J21 = diagm(0=>@.(A-2u*v))
      J12 = diagm(0=>@.(u^2))
      J22 += diagm(0=>@.(-u^2))
      Flux.Tracker.collect([J11 J12
                            J21 J22])
    end
  end
  Jmat = zeros(2N*N, 2N*N)
  dp = zeros(2N*N, 4N*N)
  brusselator_comp = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator), Jmat=Jmat, dp=dp, brusselator_jac=brusselator_jac
    function brusselator_comp(dus, us, p, t)
      @inbounds begin
        ii1 = N^2
        ii2 = ii1+N^2
        ii3 = ii2+2(N^2)
        @views u, s = us[1:ii2], us[ii2+1:end]
        du = @view dus[1:ii2]
        ds = @view dus[ii2+1:end]
        fill!(dp, 0)
        A = @view p[1:ii1]
        B = @view p[ii1+1:ii2]
        α = @view p[ii2+1:ii3]
        dfdα = @view dp[:, ii2+1:ii3]
        diagind(dfdα)
        for i in 1:ii1
          dp[i, ii1+i] = 1
        end
        II = LinearIndices((N, N, 2))
        uu = @view u[1:end÷2]
        for i in eachindex(uu)
          dp[i, i] = -uu[i]
          dp[i+ii1, i] = uu[i]
        end
        for I in CartesianIndices((N, N))
          x = xyd[I[1]]
          y = xyd[I[2]]
          i = I[1]
          j = I[2]
          ip1 = limit(i+1, N); im1 = limit(i-1, N)
          jp1 = limit(j+1, N); jm1 = limit(j-1, N)
          au = dfdα[II[i,j,1],II[i,j,1]] = (u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2
          du[II[i,j,1]] = α[II[i,j,1]]*(au) + B[II[i,j,1]] + u[II[i,j,1]]^2*u[II[i,j,2]] - (A[II[i,j,1]] + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
        end
        for I in CartesianIndices((N, N))
          i = I[1]
          j = I[2]
          ip1 = limit(i+1, N)
          im1 = limit(i-1, N)
          jp1 = limit(j+1, N)
          jm1 = limit(j-1, N)
          av = dfdα[II[i,j,2],II[i,j,2]] = (u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2
          du[II[i,j,2]] = α[II[i,j,2]]*(av) + A[II[i,j,1]]*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
        end
        brusselator_jac(Jmat,u,p,t)
        BLAS.gemm!('N', 'N', 1., Jmat, reshape(s, 2N*N, 4N*N), 1., dp)
        copyto!(ds, vec(dp))
        return nothing
      end
    end
  end
  u0 = init_brusselator_2d(xyd_brusselator)
  p = [fill(3.4,N^2); fill(1.,N^2); fill(10.,2*N^2)]
  brusselator_2d, brusselator_2d_oop, u0, p, brusselator_jac, brusselator_jac_oop, ODEProblem(brusselator_comp, copy([u0;zeros((N^2*2)*(N^2*4))]), (0.,10.), p)
end
