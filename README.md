# sensitivity_analysis_paper

The benchmark is ran with the following packages

```
  [6e4b80f9] BenchmarkTools v0.4.1
  [336ed68f] CSV v0.4.3
  [49dc2e85] Calculus v0.4.1
  [a93c6f00] DataFrames v0.14.1
  [2b5f629d] DiffEqBase v4.31.1
  [459566f4] DiffEqCallbacks v2.4.0
  [01453d9d] DiffEqDiffTools v0.7.1
  [9fdde737] DiffEqOperators v3.4.0
  [a077e3f3] DiffEqProblemLibrary v4.1.0
  [41bf760c] DiffEqSensitivity v2.4.0
  [163ba53b] DiffResults v0.0.3
  [f6369f11] ForwardDiff v0.10.1
  [429524aa] Optim v0.17.2
  [1dea7af3] OrdinaryDiffEq v4.18.0
  [65888b18] ParameterizedFunctions v4.0.0
  [91a5bcdd] Plots v0.21.0
  [37e2e3b7] ReverseDiff v0.3.1
  [90137ffa] StaticArrays v0.10.0
```

To run the scripts you can `cd` into the `script` directory and run

```julia
include("parameter_estimation_bench.jl")
include("forward_sensitivity_analysis_bench.jl")
include("adjoint_sensitivity_analysis_bench.jl")
include("bruss_scaling.jl")
```
