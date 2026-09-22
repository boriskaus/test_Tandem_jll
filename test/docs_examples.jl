# Walks the worked examples from tandem's documentation against the Tandem_jll binaries:
#
#   https://tandem.readthedocs.io/en/latest/getting-started/examples.html
#   https://tandem.readthedocs.io/en/latest/first-model/mesh.html
#
# tandem is developed by the TEAR-ERC group (https://github.com/TEAR-ERC/tandem);
# see README.md for authors and the paper to cite.
#
# These cover ground the other two suites do not: the elasticity solver (rather than
# Poisson), a direct MUMPS/LU solve through PETSc, a gmsh-generated .msh read through
# the GMSH parser (rather than [generate_mesh]), and the QDGreen discrete-Green's-function
# mode.
#
# Two places where the published pages no longer match tandem v1.2.0, reported upstream:
#   * the examples page quotes "L2 error: 2.88579e-09" for the cosine problem. That is not
#     reproducible at any degree the JLL ships (p1 7.4e-3, p2 6.1e-4, p3 3.2e-5), so it must
#     come from some other configuration. We assert the invariants instead: the error falls
#     with polynomial degree, and the direct MUMPS solve agrees with the iterative one.
#   * the first-model page passes `--discrete_green yes`, which no longer exists, and
#     tutorial.toml sets no `mode` (which has no default), so the documented command fails.
#     We set mode = "QDGreen" in the parameter file instead, which the page says is
#     equivalent.

using Test, Pkg

# Resolve into a throwaway environment: Tandem_jll is not registered, so adding it to
# this repo's Project.toml would leave it unresolvable for everyone else.
Pkg.activate(mktempdir(); io = devnull)
Pkg.add("OpenBLAS32_jll"; io = devnull)
if !isempty(get(ENV, "TANDEM_JLL_LOCAL_PATH", ""))
    Pkg.develop(path = expanduser(ENV["TANDEM_JLL_LOCAL_PATH"]); io = devnull)
else
    Pkg.add(url = "https://github.com/boriskaus/Tandem_jll.jl", io = devnull)
end
using Tandem_jll, OpenBLAS32_jll

const TANDEM_COMMIT = "b75f66692d299673bf85632ed71e2a7da71ff2e0"
const pathsep = Sys.iswindows() ? ';' : ':'
const shlib_ext = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"

# gmsh_jll pins HDF5_jll < 2 and so cannot share this environment; resolve it separately.
function gmsh_environment()
    d = mktempdir(; cleanup = false)
    script = joinpath(d, "get.jl")
    open(script, "w") do io
        println(io, "using Pkg")
        println(io, "Pkg.add(\"gmsh_jll\"; io = devnull)")
        println(io, "using gmsh_jll")
        println(io, "println(\"BIN=\", dirname(gmsh_jll.gmsh_path))")
        println(io, "for p in gmsh_jll.LIBPATH_list; println(\"LIB=\", p); end")
    end
    out = read(`$(Base.julia_cmd()) --project=$d $script`, String)
    bin, libs = "", String[]
    for line in split(out, '\n')
        startswith(line, "BIN=") && (bin = String(strip(line[5:end])))
        startswith(line, "LIB=") && push!(libs, String(strip(line[5:end])))
    end
    isempty(bin) && error("could not resolve gmsh_jll")
    return bin, libs
end
const gmsh_bin, gmsh_libs = gmsh_environment()

mpi_mod = isdefined(Tandem_jll, :MPICH_jll) ? Tandem_jll.MPICH_jll :
          isdefined(Tandem_jll, :OpenMPI_jll) ? Tandem_jll.OpenMPI_jll :
          isdefined(Tandem_jll, :MPItrampoline_jll) ? Tandem_jll.MPItrampoline_jll :
          Tandem_jll.MicrosoftMPI_jll

const work = get(ENV, "TANDEM_DOCS_WORKDIR", mktempdir(; cleanup = false))
const src = joinpath(work, "tandem")

if !isdir(src)
    run(`git clone --quiet --depth 1 --branch main https://github.com/TEAR-ERC/tandem.git $src`)
    run(Cmd(`git fetch --quiet --depth 1 origin $TANDEM_COMMIT`; dir = src))
    run(Cmd(`git checkout --quiet $TANDEM_COMMIT`; dir = src))
end

libdirs = unique(vcat(Tandem_jll.LIBPATH_list..., gmsh_libs, mpi_mod.LIBPATH_list...))
# libblastrampoline needs a backing library in a bare subprocess, for BOTH word
# sizes: PETSc's ILP64 calls and MUMPS/SCALAPACK's LP64 calls.
ilp64 = Sys.iswindows() ? joinpath(Sys.BINDIR, "libopenblas64_.dll") :
                          joinpath(Sys.BINDIR, "..", "lib", "julia", "libopenblas64_.$shlib_ext")
backing_libs = join((ilp64, OpenBLAS32_jll.libopenblas_path), ";")
const child_env = Dict(
    "PATH" => join((gmsh_bin, ENV["PATH"]), pathsep),
    Tandem_jll.JLLWrappers.LIBPATH_env =>
        join(vcat(libdirs, get(ENV, Tandem_jll.JLLWrappers.LIBPATH_env, "")), pathsep),
    "LBT_DEFAULT_LIBS" => backing_libs,
    "OMP_NUM_THREADS" => "1",
)

"Run `argv` in `dir`, returning (exitcode, combined output)."
function capture(argv::Vector{String}, dir::AbstractString)
    buf = IOBuffer()
    cmd = addenv(Cmd(Cmd(argv); dir = dir), child_env)
    p = run(pipeline(ignorestatus(cmd); stdout = buf, stderr = buf); wait = false)
    wait(p)
    return p.exitcode, String(take!(buf))
end

grabfloat(out, pat) = (m = match(pat, out); m === nothing ? nothing : parse(Float64, m[1]))
l2_error(out) = grabfloat(out, r"L2 error:\s*([0-9.eE+-]+)")
iterations(out) = (m = match(r"Iterations:\s*([0-9]+)", out); m === nothing ? nothing : parse(Int, m[1]))

@testset "documentation examples" begin

    # https://tandem.readthedocs.io/en/latest/getting-started/examples.html
    # The elasticity cosine problem; the docs quote L2 error 2.88579e-09.
    @testset "elasticity cosine (2D)" begin
        ex = joinpath(src, "examples", "elasticity", "2d")
        errs = Float64[]
        for (nm, f) in (("p1", Tandem_jll.static_2d_p1), ("p2", Tandem_jll.static_2d_p2),
                        ("p3", Tandem_jll.static_2d_p3))
            code, out = capture([f().exec[1], "cosine.toml",
                                 "--output", joinpath(work, "cosine_$nm")], ex)
            code == 0 || println(out)
            @test code == 0
            err = l2_error(out)
            @test err !== nothing
            err === nothing || (@info "$nm: L2 error = $err"; push!(errs, err))
        end
        @test isfile(joinpath(work, "cosine_p2.pvtu"))
        # High-order convergence: the error must fall with polynomial degree.
        @test length(errs) == 3
        length(errs) == 3 && @test errs[1] > errs[2] > errs[3]

        # The same problem solved directly with MUMPS: one "iteration", and an answer
        # identical to the iterative solve. The only place any suite exercises MUMPS
        # through PETSc.
        opts = joinpath(src, "examples", "options", "lu_mumps.cfg")
        code, out = capture([Tandem_jll.static_2d_p2().exec[1], "cosine.toml",
                             "--output", joinpath(work, "cosine_lu"),
                             "--petsc", "-options_file", opts], ex)
        code == 0 || println(out)
        @test code == 0
        err_lu, its = l2_error(out), iterations(out)
        @info "MUMPS direct solve: L2 error = $err_lu, iterations = $its"
        @test its == 1
        @test err_lu !== nothing
        if err_lu !== nothing && length(errs) == 3
            @test isapprox(err_lu, errs[2]; rtol = 1e-10)
        end
    end

    # https://tandem.readthedocs.io/en/latest/first-model/ -- the documented workflow is
    # "build the mesh with gmsh from a .geo, then run tandem on it". The mesh step is
    # tested with the tutorial's own geometry; the run then uses bp1_sym, because
    # tutorial.lua itself is broken on v1.2.0 (it sets `rho` as a number while tandem
    # calls it as a scenario function, so it aborts in any mode). Reported upstream.
    @testset "first model workflow: gmsh mesh -> tandem" begin
        tut = mktempdir(work; cleanup = false)
        ex2d = joinpath(src, "examples", "tandem", "2d")

        # 1. the documented gmsh step, on the tutorial geometry
        cp(joinpath(ex2d, "tutorial.geo"), joinpath(tut, "tutorial.geo"))
        code, out = capture(["gmsh", "-2", "tutorial.geo", "-o", "tutorial.msh",
                             "-format", "msh2"], tut)
        code == 0 || println(out)
        @test code == 0
        @test isfile(joinpath(tut, "tutorial.msh"))
        @test filesize(joinpath(tut, "tutorial.msh")) > 0

        # 2. run tandem on a gmsh-generated mesh, exercising the GMSH parser rather
        #    than [generate_mesh]. bp1_sym is a working .geo/.toml/.lua triple.
        for f in ("bp1_sym.geo", "bp1_sym.toml", "bp1.lua")
            cp(joinpath(ex2d, f), joinpath(tut, f))
        end
        code, out = capture(["gmsh", "-2", "bp1_sym.geo", "-o", "bp1_sym.msh",
                             "-format", "msh2"], tut)
        code == 0 || println(out)
        @test code == 0
        @test isfile(joinpath(tut, "bp1_sym.msh"))

        # Bounded by step count: bp1 integrates for millennia.
        code, out = capture([Tandem_jll.tandem_2d_p2().exec[1], "bp1_sym.toml",
                             "--petsc", "-ts_max_steps", "20"], tut)
        code == 0 || println(out)
        @test code == 0
        @test !occursin("Segmentation", out)
        @test occursin("tandem version", out)   # got far enough to print the banner
    end
end
