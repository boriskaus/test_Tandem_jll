# Runs tandem's own pytest integration suite against the Tandem_jll binaries.
#
# tandem is developed by the TEAR-ERC group (https://github.com/TEAR-ERC/tandem);
# see README.md for authors and the paper to cite. The tests, reference configs and
# reference data used here are theirs.
#
# Unlike test/runtests.jl (a fast smoke suite), this checks physics: static and SEAS
# regression against saved reference results, the convergence-rate slope, and
# consistency across 1/2/4/8 MPI ranks. It needs network access (two git clones), a
# Python environment with pytest/numpy/pandas/meshio/vtk, gmsh, and an MPI launcher.
#
# Select the configuration with TANDEM_DIM and TANDEM_DEGREE (default 2 and 2).

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

# gmsh_jll cannot share an environment with Tandem_jll (gmsh pins HDF5_jll < 2 while
# Tandem_jll needs 2.2.2), so resolve it in a project of its own and use its binary.
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
    bin = ""
    libs = String[]
    for line in split(out, '\n')
        startswith(line, "BIN=") && (bin = String(strip(line[5:end])))
        startswith(line, "LIB=") && push!(libs, String(strip(line[5:end])))
    end
    isempty(bin) && error("could not resolve gmsh_jll")
    return bin, libs
end

const gmsh_bin, gmsh_libs = gmsh_environment()

# The commit Tandem_jll is built from. Keep in sync with T/Tandem/build_tarballs.jl:
# the test scripts and reference configs must match the binaries under test.
const TANDEM_COMMIT = "b75f66692d299673bf85632ed71e2a7da71ff2e0"

const DIM = parse(Int, get(ENV, "TANDEM_DIM", "2"))
const DEG = parse(Int, get(ENV, "TANDEM_DEGREE", "2"))
const pathsep = Sys.iswindows() ? ';' : ':'
const shlib_ext = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"

@info "tandem integration suite" DIM DEG

# --- MPI launcher ----------------------------------------------------------

mpi_mod = if isdefined(Tandem_jll, :MPICH_jll)
    Tandem_jll.MPICH_jll
elseif isdefined(Tandem_jll, :OpenMPI_jll)
    Tandem_jll.OpenMPI_jll
elseif isdefined(Tandem_jll, :MPItrampoline_jll)
    Tandem_jll.MPItrampoline_jll
elseif isdefined(Tandem_jll, :MicrosoftMPI_jll)
    Tandem_jll.MicrosoftMPI_jll
else
    error("no MPI launcher available in Tandem_jll")
end
const mpiexec_path = mpi_mod.mpiexec().exec[1]
const mpi_bin = dirname(mpiexec_path)

# --- workspace -------------------------------------------------------------

const work = get(ENV, "TANDEM_INTEGRATION_WORKDIR", mktempdir(; cleanup = false))
const src = joinpath(work, "tandem")
const testdir = joinpath(src, "test")
const exedir = joinpath(work, "build")

function run_checked(cmd::Cmd, what::AbstractString)
    @info "running: $what"
    p = run(ignorestatus(cmd))
    p.exitcode == 0 || error("$what failed with exit code $(p.exitcode)")
end

if !isdir(src)
    run_checked(`git clone --quiet --depth 1 --branch main
                 https://github.com/TEAR-ERC/tandem.git $src`, "clone tandem")
    run_checked(Cmd(`git fetch --quiet --depth 1 origin $TANDEM_COMMIT`; dir = src),
                "fetch pinned commit")
    run_checked(Cmd(`git checkout --quiet $TANDEM_COMMIT`; dir = src), "checkout pinned commit")
end

const datadir = joinpath(testdir, "test_data")
if !isdir(joinpath(datadir, ".git"))
    run_checked(`git clone --quiet --depth 1
                 https://github.com/TEAR-ERC/tandem_test_data $datadir`, "clone tandem_test_data")
end

# tandem's scripts expect <EXECUTABLE_DIR>/app/{static,tandem}; the JLL names its
# executables per configuration, so point those two paths at the right pair.
mkpath(joinpath(exedir, "app"))
for (generic, sym) in (("static", Symbol("static_$(DIM)d_p$(DEG)")),
                       ("tandem", Symbol("tandem_$(DIM)d_p$(DEG)")))
    isdefined(Tandem_jll, sym) || error("Tandem_jll has no $sym; is DIM/DEGREE in range?")
    target = getproperty(Tandem_jll, sym)().exec[1]
    # Copy rather than symlink: creating symlinks on Windows needs Developer Mode.
    # Windows also needs the .exe suffix for the scripts' bare `app/static` to resolve.
    for name in (Sys.iswindows() ? ("$generic.exe",) : (generic,))
        dest = joinpath(exedir, "app", name)
        rm(dest; force = true)
        cp(target, dest; force = true)
        Sys.iswindows() || chmod(dest, 0o755)
    end
    @info "  $generic -> $(basename(target))"
end

# The scripts hardcode `mpirun --oversubscribe`. --oversubscribe is an OpenMPI flag that
# MPICH's Hydra rejects, and `mpirun` does not exist at all on Windows, where MS-MPI ships
# only mpiexec. Point them at this platform's launcher by absolute path. Reported upstream.
launcher = replace(mpiexec_path, '\\' => '/')
for d in ("2D", "3D"), f in readdir(joinpath(testdir, "scripts", d); join = true)
    endswith(f, ".sh") || continue
    txt = read(f, String)
    occursin("mpirun", txt) || continue
    txt = replace(txt, "mpirun --oversubscribe" => "\"$launcher\"", "mpirun" => "\"$launcher\"")
    write(f, txt)
end

# test_regression_static.py compares point_data by array position. DG output stores one
# value per element-local node (many duplicate coordinates), and the element order follows
# the ParMETIS partition, which is platform-dependent -- so the test can only pass on the
# platform its reference .vtu was generated on. Rewrite the comparison to key on
# (cell centroid, point), as tandem's own conftest.py helpers do for the other tests.
# Same reference data, same tolerance, just order-independent. Proposed upstream.
let f = joinpath(testdir, "test_regression_static.py")
    txt = read(f, String)
    old = "    assert_allclose(out_data, ref_data, atol=tolerances[\"static\"])"
    new = join([
        "    import numpy as np",
        "",
        "    def _cell_point_map(mesh, name):",
        "        pts, vals = mesh.points, np.ravel(mesh.point_data[name])",
        "        table = {}",
        "        for blk in mesh.cells:",
        "            for ids in blk.data:",
        "                centroid = tuple(np.round(pts[ids].mean(axis=0), 10))",
        "                table[centroid] = {tuple(np.round(pts[i], 10)): vals[i] for i in ids}",
        "        return table",
        "",
        "    out_map, ref_map = _cell_point_map(out, field), _cell_point_map(ref, field)",
        "    assert set(out_map) == set(ref_map), (",
        "        f\"cell centroids differ: {len(out_map)} in output, {len(ref_map)} in reference\"",
        "    )",
        "    expected = sum(len(v) for v in ref_map.values())",
        "    diffs = [",
        "        abs(v - ref_map[c][p])",
        "        for c, cell_pts in out_map.items()",
        "        for p, v in cell_pts.items()",
        "        if p in ref_map[c]",
        "    ]",
        "    assert len(diffs) == expected, f\"matched {len(diffs)} of {expected} (cell, point) pairs\"",
        "    assert max(diffs) <= tolerances[\"static\"], (",
        "        f\"max |{field} diff| = {max(diffs):.6g} > {tolerances['static']}\"",
        "    )",
    ], "\n")
    if occursin(old, txt)
        write(f, replace(txt, old => new))
        @info "rewrote test_regression_static.py to match on (cell centroid, point)"
    else
        @warn "test_regression_static.py comparison not found; left as upstream has it"
    end
end

# --- environment for the child processes ------------------------------------

libdirs = unique(vcat(Tandem_jll.LIBPATH_list..., gmsh_libs,
                      mpi_mod.LIBPATH_list...))
# libblastrampoline needs a backing library in a bare subprocess, for BOTH word
# sizes: PETSc's ILP64 calls and MUMPS/SCALAPACK's LP64 calls.
ilp64 = Sys.iswindows() ? joinpath(Sys.BINDIR, "libopenblas64_.dll") :
                          joinpath(Sys.BINDIR, "..", "lib", "julia", "libopenblas64_.$shlib_ext")
backing_libs = join((ilp64, OpenBLAS32_jll.libopenblas_path), ";")

const child_env = Dict(
    "PATH" => join((gmsh_bin, mpi_bin, ENV["PATH"]), pathsep),
    Tandem_jll.JLLWrappers.LIBPATH_env =>
        join(vcat(libdirs, get(ENV, Tandem_jll.JLLWrappers.LIBPATH_env, "")), pathsep),
    "LBT_DEFAULT_LIBS" => backing_libs,
    "OMP_NUM_THREADS" => "1",
)

# --- generate reference outputs, then run the pytest suite ------------------

@testset "tandem integration ($(DIM)D, p$(DEG))" begin
    gen = addenv(Cmd(`bash scripts/generate_test_outputs.sh
                      $exedir $(joinpath(testdir, "temp_test_results")) $DIM $DEG ON`;
                     dir = testdir), child_env)
    ok = success(pipeline(ignorestatus(gen); stdout = stdout, stderr = stderr))
    if !ok
        # tandem's scripts redirect each run into temp_test_results/*.log; without this
        # a crash in one of them shows only as "Abort trap" with no diagnostics.
        tmpres = joinpath(testdir, "temp_test_results")
        for f in (isdir(tmpres) ? sort(readdir(tmpres)) : String[])
            endswith(f, ".log") || continue
            @info "----- $f -----"
            println(last(split(read(joinpath(tmpres, f), String), '\n'), 40) |> x -> join(x, "\n"))
        end
    end
    @test ok

    # Mirrors the selection in tandem's test/CMakeLists.txt
    tests = ["test_parallel_consistency_static.py", "test_convergence_static.py"]
    DIM == 2 && append!(tests, ["test_probe_writer_consistency.py",
                                "test_correctness_volume_tagging.py",
                                "test_parallel_consistency_volume_tagging.py"])
    if DEG in (2, 3)                       # degrees with reference data upstream
        push!(tests, "test_regression_static.py")
        DIM == 2 && push!(tests, "test_regression_SEAS.py")
    end

    for t in tests
        @testset "$t" begin
            cmd = addenv(Cmd(`python3 -m pytest -q $t
                              --domain_dimension=$DIM --polynomial_degree=$DEG`;
                             dir = testdir), child_env)
            @test success(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
        end
    end
end
