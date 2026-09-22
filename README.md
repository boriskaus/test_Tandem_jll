# test_Tandem_jll

CI harness for `Tandem_jll`, the Yggdrasil build of **tandem**.

## About tandem

[**tandem**](https://github.com/TEAR-ERC/tandem) is a scalable discontinuous Galerkin code on
unstructured curvilinear grids for linear elasticity problems and sequences of earthquakes and
aseismic slip (SEAS). It is developed by the [TEAR-ERC](https://github.com/TEAR-ERC) group.

- Repository: <https://github.com/TEAR-ERC/tandem/>
- Documentation: <https://tandem.readthedocs.io/en/latest/>
- Licence: BSD-3-Clause, © 2020 Ludwig-Maximilians-Universität München

**Authors:** Carsten Uphoff, Dave May, Alice-Agnes Gabriel, Jeena Yun, Thomas Ulrich,
Nico Schliwa, Casper Pranger.

### Please cite

If you use tandem for your research, cite:

> Carsten Uphoff, Dave A. May, Alice-Agnes Gabriel (2023). *A discontinuous Galerkin method for
> sequences of earthquakes and aseismic slip on multiple faults using unstructured curvilinear
> grids.* Geophysical Journal International, 233(1), 586–626.
> <https://doi.org/10.1093/gji/ggac467>

The kernel generator is [YATeTo](https://doi.org/10.1145/3406835).

This repository is **not** affiliated with the tandem developers — it only packages and tests
their code for Julia. Please direct questions about tandem itself to the upstream repository.

## What this repository is for

It checks that the JLL's binaries actually run on Linux, macOS and Windows before the Yggdrasil
recipe goes to a PR, the same way
[`test_PETSc_jll`](https://github.com/boriskaus/test_PETSc_jll) does for PETSc.

`Tandem_jll` ships 12 executables — `tandem` and `static`, each compiled for
`DOMAIN_DIMENSION` ∈ {2, 3} × `POLYNOMIAL_DEGREE` ∈ {1, 2, 3}, because both are compile-time
constants in tandem.

| Test set | What it covers |
|---|---|
| all 12 executables start | every product is present, links, and runs |
| static, manufactured solution (2D) | a Poisson solve against an analytic solution; asserts the L2 error and that it falls with polynomial degree. Exercises PETSc, Lua, Eigen and the YATeTo-generated kernels |
| static in parallel (2 ranks) | MPI + ParMETIS partitioning; asserts the 2-rank answer matches the serial one |
| tandem, QD SEAS time loop | the quasi-dynamic time integrator, the Lua scenario interface and HDF5 probe output |

The inputs in `test/data/` are copied from tandem's own `examples/` and all use
`[generate_mesh]`, so no gmsh and no external mesh files are needed.

> **Note:** this is a lightweight smoke/consistency suite, *not* tandem's own test suite.
> Upstream ships a much fuller one (12 C++ unit binaries plus pytest regression, convergence,
> parallel-consistency and probe-writer integration tests) — see
> [`test/README.md`](https://github.com/TEAR-ERC/tandem/blob/main/test/README.md).
>
> Their **pytest integration tests are run here** by `test/integration.jl` (see below); the
> 12 C++ unit binaries are not, since they are not shipped in the JLL. Those were run
> separately against the same source and dependency versions: 31/31 passing.

## Test suites

| File | Trigger | What it runs |
|---|---|---|
| `test/runtests.jl` | every push (`ci.yml`) | the smoke suite above — fast, no network |
| `test/integration.jl` | on demand (`integration.yml`) | **tandem's own pytest suite** against the JLL binaries: static and SEAS regression vs upstream reference data, convergence slope, 1/2/4/8-rank consistency, volume tagging, HDF5 probe-writer |
| `test/docs_examples.jl` | on demand (`integration.yml`) | the worked examples from tandem's documentation: the elasticity solver, a direct MUMPS/LU solve, and gmsh `.geo` → `.msh` → tandem |

The integration workflow is `workflow_dispatch` only: it clones tandem and
`tandem_test_data`, installs a Python stack and runs real simulations. It takes
dimension and degree as inputs, so you can run one configuration or a matrix.

Set `TANDEM_DIM` / `TANDEM_DEGREE` to choose the configuration when running locally:

```bash
TANDEM_DIM=2 TANDEM_DEGREE=2 julia --project=. test/integration.jl
julia --project=. test/docs_examples.jl
```

Both need `pytest`, `numpy`, `pandas`, `meshio` and `vtk` importable; gmsh comes from
`gmsh_jll`, resolved in a throwaway project because it pins `HDF5_jll < 2` and so cannot
share an environment with `Tandem_jll`.

## Running against a deployed JLL

```julia
using Pkg
Pkg.test()
```

## Running against a locally built JLL

After `julia build_tarballs.jl --deploy=local <triplet>` in `T/Tandem`, point
`TANDEM_JLL_LOCAL_PATH` at the generated JLL:

```bash
TANDEM_JLL_LOCAL_PATH=~/.julia/dev/Tandem_jll \
  julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["TANDEM_JLL_LOCAL_PATH"]); Pkg.test()'
```

## Notes

- Julia 1.12 is the floor: `PETSc_jll` 3.25.4 binds the SuiteSparse shipped with Julia 1.12.
- The JLL is built with `ARCH=noarch`, so no `-march` flags: it runs on any CPU of the target
  architecture, at some cost in kernel performance versus a tuned local build.
- libxsmm's generator is not used, so the YATeTo kernels fall back to their Eigen code path.
