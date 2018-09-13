## Section 1: Sensitivity benchmark
In this section, we are going to make a table that compares between autodiff,
hand derived diffeq, standard sensitivity analysis, and finite differentiation
in both forward and adjoint sensitivity analyses for small, medium and large
systems with non-stiff and stiff solvers.

We plan to use
Lotka-Volterra: small
Variable coefficient PDE: medium to large

## Section 2: Event handling

Mention parameter estimation in PK models, which leads to section 3.

Examples:
    1. Multiple dosing on a PK model, standard sensitivity should work
    2. With parameter-dependent dosages, standard sensitivity won't work
    3. With parameter-dependent duration, standard sensitivity won't work
    4. With parameters as dependent variables, standard sensitivity won't work

## Section 3: Timing for parameter estimation
Compare autodiff, hand derived diffeq, standard sensitivity analysis, and
finite differentiation in parameter estimation problems.

Mention the difficulty of doing standard sensitivity analysis on DDE, DAE and
SDE.

## Section 4/Conclusion: Caveats of autodiff
Julia code can be differentiable by autodiff, but needs to be careful for
certain cases, e.g. [ODE
interpolation](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/blob/master/src/dense/generic_dense.jl#L172-L174).
However, for multi-language platforms like `scipy`, it is very hard to perform
autodiff.

Standard sensitivity analysis works when the solution is continuous, and has
great difficulties to generalize.

With autodiff, one gets event handling for free, but needs proper care.

Cons:
    - Difficulty of using cache arrays with autodiff.

    - Long compile time and a lot of codegen, but it might not be a problem in
      parameter estimation where one needs to run a model many times. However, for
      project like `PuMaS.jl` compile time can still be significant.

    - Mention `Capstan.jl` and `Cassette.jl`.
