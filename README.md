# sensitivity_analysis_paper

To run the scripts you can do:

```bash
julia --startup-file=no --project=sensitivity_analysis_paper -e 'import Pkg; Pkg.instantiate(); include("sensitivity_analysis_paper/run.jl")'
```

The benchmarks are run on
```
Julia Version 1.5.0
Commit 96786e22cc (2020-08-01 23:44 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake)
Environment:
  JULIA_NUM_THREADS = 8
```
