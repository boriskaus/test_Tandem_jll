using Test, Pkg
using CompilerSupportLibraries_jll, OpenBLAS32_jll

# By default this tests the Tandem_jll deployed to GitHub (what CI does).  To test a
# JLL built locally with BinaryBuilder (`julia build_tarballs.jl --deploy=local <triplet>`),
# point TANDEM_JLL_LOCAL_PATH at the generated JLL directory:
#   TANDEM_JLL_LOCAL_PATH=~/.julia/dev/Tandem_jll julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["TANDEM_JLL_LOCAL_PATH"]); Pkg.test()'
if !isempty(get(ENV, "TANDEM_JLL_LOCAL_PATH", ""))
    local_jll = expanduser(ENV["TANDEM_JLL_LOCAL_PATH"])
    println("Using locally built Tandem_jll from $local_jll")
    Pkg.develop(path=local_jll)
else
    Pkg.add(url="https://github.com/boriskaus/Tandem_jll.jl")
end
using Tandem_jll

@show Base.BinaryPlatforms.HostPlatform()
@show Tandem_jll.host_platform

# --- MPI launcher -----------------------------------------------------------

if isdefined(Tandem_jll, :MPICH_jll)
    const mpiexec_cmd = Tandem_jll.MPICH_jll.mpiexec()
    const MPI_LIBPATH = Tandem_jll.MPICH_jll.LIBPATH
elseif isdefined(Tandem_jll, :MicrosoftMPI_jll)
    const mpiexec_cmd = Tandem_jll.MicrosoftMPI_jll.mpiexec()
    const MPI_LIBPATH = Tandem_jll.MicrosoftMPI_jll.LIBPATH
elseif isdefined(Tandem_jll, :OpenMPI_jll)
    const mpiexec_cmd = Tandem_jll.OpenMPI_jll.mpiexec()
    const MPI_LIBPATH = Tandem_jll.OpenMPI_jll.LIBPATH
elseif isdefined(Tandem_jll, :MPItrampoline_jll)
    const mpiexec_cmd = Tandem_jll.MPItrampoline_jll.mpiexec()
    const MPI_LIBPATH = Tandem_jll.MPItrampoline_jll.LIBPATH
else
    println("No MPI library detected; parallel runs will be skipped")
    const mpiexec_cmd = nothing
    const MPI_LIBPATH = Ref{String}("")
end
@show mpiexec_cmd

const pathsep = Sys.iswindows() ? ';' : ':'
const shlib_ext = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"

# libblastrampoline has no backing library in a bare subprocess -- inside a Julia
# process the stdlib OpenBLAS registers itself, but these executables are launched
# directly, so unregistered ILP64 calls segfault. Point LBT at Julia's own ILP64
# OpenBLAS and at OpenBLAS32_jll's LP64 one; LBT detects each one's word size.
const ilp64_lib = Sys.iswindows() ?
    joinpath(Sys.BINDIR, "libopenblas64_.dll") :
    joinpath(Sys.BINDIR, "..", "lib", "julia", "libopenblas64_.$shlib_ext")
const backing_libs = join((ilp64_lib, OpenBLAS32_jll.libopenblas_path), ";")

function with_env(cmd::Cmd; extra_libpath::Vector{String}=String[])
    key = Tandem_jll.JLLWrappers.LIBPATH_env
    # Prepend to whatever the command already carries rather than replacing it.
    # On Windows this variable is PATH, and dropping its existing entries leaves
    # the child process unable to resolve the system DLLs.
    current = ""
    for e in something(cmd.env, String[])
        startswith(e, key * "=") && (current = e[length(key)+2:end])
    end
    isempty(current) && (current = get(ENV, key, ""))
    libdirs = unique(vcat(CompilerSupportLibraries_jll.LIBPATH_list...,
                          Tandem_jll.LIBPATH_list..., extra_libpath,
                          String.(filter(!isempty, split(current, pathsep)))))
    return addenv(cmd,
        "LBT_DEFAULT_LIBS" => backing_libs,
        key => join(libdirs, pathsep),
        "OMP_NUM_THREADS" => "1",
    )
end

# Interpolate cmd.exec, not cmd: Julia allows only the first interpolant to carry
# its own environment, and the JLL wrappers set one on both.
mpirun(n::Int, cmd::Cmd) =
    with_env(`$(mpiexec_cmd) -n $n $(cmd.exec)`;
             extra_libpath=vcat(MPI_LIBPATH[], Tandem_jll.LIBPATH[]))

serial(cmd::Cmd) = with_env(cmd)

const datadir = joinpath(@__DIR__, "data")

"Run `cmd` in `datadir`, returning (exitcode, combined output)."
function run_capture(cmd::Cmd)
    out = IOBuffer()
    p = run(pipeline(Cmd(cmd; dir=datadir); stdout=out, stderr=out); wait=false)
    wait(p)
    return p.exitcode, String(take!(out))
end

"Pull the `L2 error: <x>` value out of a `static` run."
function l2_error(output::AbstractString)
    m = match(r"L2 error:\s*([0-9.eE+-]+)", output)
    m === nothing && return nothing
    return parse(Float64, m.captures[1])
end

const tandem_exes = [
    (:tandem_2d_p1, Tandem_jll.tandem_2d_p1), (:tandem_2d_p2, Tandem_jll.tandem_2d_p2),
    (:tandem_2d_p3, Tandem_jll.tandem_2d_p3), (:tandem_3d_p1, Tandem_jll.tandem_3d_p1),
    (:tandem_3d_p2, Tandem_jll.tandem_3d_p2), (:tandem_3d_p3, Tandem_jll.tandem_3d_p3),
]
const static_exes = [
    (:static_2d_p1, Tandem_jll.static_2d_p1), (:static_2d_p2, Tandem_jll.static_2d_p2),
    (:static_2d_p3, Tandem_jll.static_2d_p3), (:static_3d_p1, Tandem_jll.static_3d_p1),
    (:static_3d_p2, Tandem_jll.static_3d_p2), (:static_3d_p3, Tandem_jll.static_3d_p3),
]

@testset "Tandem_jll" begin

    @testset "all 12 executables start" begin
        for (nm, f) in vcat(tandem_exes, static_exes)
            # No config file: argparse must reject it and exit non-zero without crashing.
            code, out = run_capture(ignorestatus(serial(`$(f())`)))
            @test code != 139           # not a segfault
            @test occursin("config", lowercase(out)) || occursin("usage", lowercase(out))
            @info "$nm started" exitcode=code
        end
    end

    # examples/poisson/2d/manufactured.toml: Poisson with an analytic solution on a
    # self-generated mesh, so no gmsh and no external mesh file are needed.
    @testset "static, manufactured solution (2D)" begin
        errs = Float64[]
        for (nm, f) in [(:static_2d_p1, Tandem_jll.static_2d_p1),
                        (:static_2d_p2, Tandem_jll.static_2d_p2),
                        (:static_2d_p3, Tandem_jll.static_2d_p3)]
            code, out = run_capture(serial(`$(f()) manufactured.toml`))
            @test code == 0
            err = l2_error(out)
            @test err !== nothing
            if err !== nothing
                @info "$nm: L2 error = $err"
                push!(errs, err)
                @test err < 1e-2
            end
        end
        # Higher polynomial degree must reduce the error on the same mesh.
        if length(errs) == 3
            @test errs[2] < errs[1]
            @test errs[3] < errs[2]
        end
    end

    @testset "static in parallel (2 ranks)" begin
        if mpiexec_cmd === nothing
            @info "no MPI launcher, skipping"
        else
            code, out = run_capture(mpirun(2, `$(Tandem_jll.static_2d_p2()) manufactured.toml`))
            @test code == 0
            err_par = l2_error(out)
            @test err_par !== nothing

            _, out_ser = run_capture(serial(`$(Tandem_jll.static_2d_p2()) manufactured.toml`))
            err_ser = l2_error(out_ser)
            if err_par !== nothing && err_ser !== nothing
                @info "serial $err_ser vs 2 ranks $err_par"
                # Same discretisation, so the partitioning must not change the answer.
                @test isapprox(err_par, err_ser; rtol=1e-6)
            end
        end
    end

    # mms1.toml is a quasi-dynamic SEAS run with a generated mesh; this exercises the
    # time integrator, the Lua scenario interface and the HDF5 probe writer.
    @testset "tandem, QD SEAS time loop" begin
        mktempdir() do tmp
            # Shorten the run: mms1.toml integrates for ~70 years, far too long for CI.
            cfg = replace(read(joinpath(datadir, "mms1.toml"), String),
                          r"final_time\s*=\s*\S+" => "final_time = 1.0e6")
            write(joinpath(tmp, "mms1.toml"), cfg)
            cp(joinpath(datadir, "mms1.lua"), joinpath(tmp, "mms1.lua"))

            # Bound by step count, not simulated time: the adaptive QD integrator can
            # take a very large number of steps to reach even a short final_time.
            cmd = Cmd(serial(`$(Tandem_jll.tandem_2d_p2()) mms1.toml --petsc -ts_max_steps 20`); dir=tmp)
            out = IOBuffer()
            p = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out); wait=false)
            wait(p)
            text = String(take!(out))
            @info "tandem QD run" exitcode=p.exitcode
            p.exitcode == 0 || println(text)
            @test p.exitcode == 0
            @test !occursin("Segmentation", text)
        end
    end
end
