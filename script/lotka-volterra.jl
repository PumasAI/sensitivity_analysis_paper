using OrdinaryDiffEq, StaticArrays
function lvdf(du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

function lvcom_df(du, u, p, t)
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

lvdf_with_jacobian = ODEFunction(lvdf, jac=(J,u,p,t)->begin
                                   a,b,c = p
                                   x, y = u
                                   J[1] = a-b*y
                                   J[2] = y
                                   J[3] = -b*x
                                   J[4] = x-c
                                   nothing
                               end)
