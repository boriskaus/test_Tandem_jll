# test_Tandem_jll

CI harness for [`Tandem_jll`](https://github.com/JuliaPackaging/Yggdrasil/tree/master/T/Tandem),
the Yggdrasil build of [TEAR-ERC/tandem](https://github.com/TEAR-ERC/tandem) — a discontinuous
Galerkin code for sequences of earthquakes and aseismic slip (SEAS).

It exists to check that the JLL's binaries actually run on Linux, macOS and Windows before the
recipe goes to Yggdrasil, the same way [`test_PETSc_jll`](https://github.com/boriskaus/test_PETSc_jll)
does for PETSc.

## What is tested

`Tandem_jll` ships 12 executables — `tandem` and `static`, each compiled for
`DOMAIN_DIMENSION` ∈ {2, 3} × `POLYNOMIAL_DEGREE` ∈ {1, 2, 3}, because both are compile-time
constants in tandem.

| Test set | What it covers |
|---|---|
| all 12 executables start | every product is present, links, and runs (no missing shared library, no segfault) |
| static, manufactured solution (2D) | a real Poisson solve against an analytic solution; asserts the L2 error and that it falls with polynomial degree. Exercises PETSc, Lua, Eigen and the yateto-generated kernels |
| static in parallel (2 ranks) | MPI + ParMETIS partitioning; asserts the 2-rank answer matches the serial one |
| tandem, QD SEAS time loop | the quasi-dynamic time integrator, the Lua scenario interface and HDF5 probe output |

The inputs in `test/data/` are copied from tandem's own `examples/` and all use
`[generate_mesh]`, so no gmsh and no external mesh files are needed.

## Running against a deployed JLL

By default the tests install the JLL from <https://github.com/boriskaus/Tandem_jll.jl>, which is
what CI does:

```julia
using Pkg
Pkg.test()
```

## Running against a locally built JLL

After `julia build_tarballs.jl --deploy=local <triplet>` in `T/Tandem`, point
`TANDEM_JLL_LOCAL_PATH` at the generated JLL. Develop it in the project environment first,
otherwise `Pkg.test()` tries to instantiate the `Tandem_jll` recorded in `Manifest.toml` before
`runtests.jl` runs:

```bash
TANDEM_JLL_LOCAL_PATH=~/.julia/dev/Tandem_jll \
  julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["TANDEM_JLL_LOCAL_PATH"]); Pkg.test()'
```

## Notes

- Julia 1.12 is the floor: `PETSc_jll` 3.25.4 binds the SuiteSparse shipped with Julia 1.12 by
  soname.
- The JLL is built with `ARCH=noarch`, so no `-march` flags: it runs on any CPU of the target
  architecture, at some cost in kernel performance versus a tuned local build.
- libxsmm's generator is not used, so the yateto kernels fall back to their Eigen-based code path.
