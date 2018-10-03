# =============================================================== #
# Small regime (2x3 Jacobian matrix)

include("sensitivity.jl")
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, BenchmarkTools, StaticArrays#, Profile, ProfileView


df = function (du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

com_df = function (du, u, p, t)
    a,b,c = p
    x, y, s1, s2, s3, s4, s5, s6 = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    #####################
    #     [a-by -bx]
    # J = [        ]
    #     [y    x-c]
    #####################
    J  = @SMatrix [a-b*y -b*x
                   y    x-c]
    JS = J*@SMatrix[s1 s3 s5
                    s2 s4 s6]
    G  = @SMatrix [x -x*y 0
                   0  0  -y]
    du[3:end] .= vec(JS+G)
    nothing
end

df_with_jacobian = ODEFunction(df, jac=(J,u,p,t)->begin
                                   a,b,c = p
                                   x, y = u
                                   J[1] = a-b*y
                                   J[2] = y
                                   J[3] = -b*x
                                   J[4] = x-c
                                   nothing
                               end)

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]
com_u0 = [u0...;zeros(6)]
comprob = ODEProblem(com_df, com_u0, tspan, p)
#=
f = function (u, p, t)
    a,b,c = p
    x, y = u
    dx = a*x - b*x*y
    dy = -c*y + x*y
    @SVector [dx, dy]
end

# Using `SVector` doesn't give any significant run-time benefits, but increases
# compile time drastically

u0s = @SVector [1.,1.]; sp = @SVector [1.5,1.0,3.0];
@time auto_sen(f, u0s, tspan, sp, Vern9(), abstol=1e-5,reltol=1e-7)
# 139.719343 seconds (22.66 M allocations: 1.558 GiB, 0.58% gc time)
@btime auto_sen($f, $u0s, $tspan, $sp, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 135.706 μs (499 allocations: 84.55 KiB)
=#

@time auto_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 13.564837 seconds (43.31 M allocations: 2.326 GiB, 9.15% gc time)
@time diffeq_sen(df, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 5.712511 seconds (16.38 M allocations: 931.242 MiB, 9.13% gc time)
@time diffeq_sen(df_with_jacobian, u0, tspan, p, Vern9(), abstol=1e-5,reltol=1e-7)
# 2.679172 seconds (6.21 M allocations: 320.881 MiB, 5.36% gc time)
@time solve(comprob, Vern9(),abstol=1e-5,reltol=1e-7,save_everystep=false)
# 3.484515 seconds (8.10 M allocations: 417.261 MiB, 7.50% gc time)

@btime auto_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 99.404 μs (485 allocations: 57.11 KiB)
@btime diffeq_sen($df, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 308.289 μs (8137 allocations: 391.78 KiB)
@btime diffeq_sen($df_with_jacobian, $u0, $tspan, $p, $(Vern9()), abstol=1e-5,reltol=1e-7)
# 263.562 μs (8084 allocations: 389.08 KiB)
@btime solve($comprob, $(Vern9()),abstol=1e-5,reltol=1e-7,save_everystep=false)
# 36.517 μs (111 allocations: 14.67 KiB)

@btime auto_sen($df, $u0, $tspan, $p, $(Vern9()))
# 90.698 μs (467 allocations: 56.95 KiB)
@btime diffeq_sen($df, $u0, $tspan, $p, $(Vern9()))
# 255.729 μs (5849 allocations: 292.16 KiB)
@btime diffeq_sen($df_with_jacobian, $u0, $tspan, $p, $(Vern9()))
# 192.806 μs (5619 allocations: 273.50 KiB)
@btime solve($comprob, $(Vern9()),save_everystep=false)
# 27.937 μs (112 allocations: 14.66 KiB)

# =============================================================== #
# Large regime (128x3 Jacobian matrix)
using LinearAlgebra, Test
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

bfun, b_u0, brusselator_jac,brusselator_comp = makebrusselator(5)
# Run low tolerance to test correctness
sol1 = @time auto_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  8.943112 seconds (50.37 M allocations: 2.323 GiB, 9.23% gc time)
sol2 = @time diffeq_sen(bfun, b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  13.934268 seconds (195.79 M allocations: 10.914 GiB, 16.79% gc time)
sol3 = @time diffeq_sen(ODEFunction(bfun, jac=brusselator_jac), b_u0, (0.,10.), [3.4, 1., 10.], abstol=1e-5,reltol=1e-7)
#  9.747963 seconds (175.60 M allocations: 10.206 GiB, 20.70% gc time)
sol4 = @time solve(brusselator_comp, Tsit5(), abstol=1e-5,reltol=1e-7,save_everystep=false)
#  3.850392 seconds (36.08 M allocations: 941.787 MiB, 7.35% gc time)

difference1 = copy(sol1)
difference2 = copy(sol1)
difference3 = vec(sol1) .- vec(sol4[2][5*5*2+1:end])
for i in eachindex(sol2)
    difference1[:, i] .-= sol2[i]
    difference2[:, i] .-= sol3[i]
end
@test norm(difference1) < 0.01 && norm(difference2) < 0.01 && norm(difference3) < 0.01

# High tolerance to benchmark
bfun_n, b_u0_n, brusselator_jacn, b_comp = makebrusselator(8)
@time auto_sen(bfun_n, b_u0_n, (0.,10.), [3.4, 1., 10.])
#  13.632362 seconds (238.33 M allocations: 10.063 GiB, 15.94% gc time)
@time diffeq_sen(bfun_n, b_u0_n, (0.,10.), [3.4, 1., 10.])
# 302.428220 seconds (3.42 G allocations: 216.285 GiB, 12.05% gc time)
@time diffeq_sen(ODEFunction(bfun_n, jac=brusselator_jacn), b_u0_n, (0.,10.), [3.4, 1., 10.])
#  36.712953 seconds (442.08 M allocations: 10.215 GiB, 6.01% gc time)
@time solve(b_comp, Tsit5(), abstol=1e-5,reltol=1e-7,save_everystep=false)
#  12.807239 seconds (249.44 M allocations: 4.843 GiB, 10.15% gc time)
