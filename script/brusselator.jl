using LinearAlgebra
function makebrusselator(N=8)
    xyd_brusselator = range(0,stop=1,length=N)
    function limit(a, N)
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
    function brusselator_2d_loop(du, u, p, t)
        @inbounds begin
            A, B, α  = p
            xyd = xyd_brusselator
            dx = step(xyd)
            N = length(xyd)
            α = α/dx^2
            II = LinearIndices((N, N, 2))
            for I in CartesianIndices((N, N))
                x = xyd[I[1]]
                y = xyd[I[2]]
                i = I[1]
                j = I[2]
                ip1 = limit(i+1, N); im1 = limit(i-1, N)
                jp1 = limit(j+1, N); jm1 = limit(j-1, N)
                du[II[i,j,1]] = α*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]]) +
                    B + u[II[i,j,1]]^2*u[II[i,j,2]] - (A + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
            end
            for I in CartesianIndices((N, N))
              i = I[1]
              j = I[2]
              ip1 = limit(i+1, N)
              im1 = limit(i-1, N)
              jp1 = limit(j+1, N)
              jm1 = limit(j-1, N)
              du[II[i,j,2]] = α*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]]) +
                  A*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
            end
            return nothing
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
    brusselator_jac = (J,a,p,t)-> begin
        A, B, α  = p
        u = @view a[1:end÷2]
        v = @view a[end÷2+1:end]
        N2 = length(a)÷2
        fill!(J, 0)

        J[1:N2, 1:N2] .= α.*Op
        J[N2+1:end, N2+1:end] .= α.*Op

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
    Jmat = zeros(2N*N, 2N*N)
    dp = zeros(2N*N, 3)
    function brusselator_comp(dus, us, p, t)
        @inbounds begin
            @views u, s = us[1:N*N*2], us[N*N*2+1:end]
            du = @view dus[1:N*N*2]
            fill!(dp, 0)
            dp[1:end÷2, 1] .= -@view u[1:end÷2]
            dp[end÷2+1:end, 1] .= @view u[1:end÷2]
            A, B, α  = p
            dfdα = @view dp[:, 3]
            @. dp[1:end÷2, 2] .= 1
            xyd = xyd_brusselator
            dx = step(xyd)
            N = length(xyd)
            II = LinearIndices((N, N, 2))
            for I in CartesianIndices((N, N))
                x = xyd[I[1]]
                y = xyd[I[2]]
                i = I[1]
                j = I[2]
                ip1 = limit(i+1, N); im1 = limit(i-1, N)
                jp1 = limit(j+1, N); jm1 = limit(j-1, N)
                au = dfdα[II[i,j,1]] = (u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2
                du[II[i,j,1]] = α*(au) + B + u[II[i,j,1]]^2*u[II[i,j,2]] - (A + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
            end
            for I in CartesianIndices((N, N))
                i = I[1]
                j = I[2]
                ip1 = limit(i+1, N)
                im1 = limit(i-1, N)
                jp1 = limit(j+1, N)
                jm1 = limit(j-1, N)
                av = dfdα[II[i,j,2]] = (u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2
                du[II[i,j,2]] = α*(av) + A*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
            end
            brusselator_jac(Jmat,u,p,t)
            BLAS.gemm!('N', 'N', 1., Jmat, reshape(s, N*N*2, 3), 1., dp)
            dus[N*N*2+1:end] .= vec(dp)
            #@show t
            return nothing
        end
    end
    u0 = init_brusselator_2d(xyd_brusselator)
    brusselator_2d_loop, u0, brusselator_jac, ODEProblem(brusselator_comp, [u0;zeros(N*N*2*3)], (0,10.), [3.4, 1., 10.])
end
