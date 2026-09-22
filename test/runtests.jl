using Test, Pkg, Printf

# By default this tests the Tandem_jll deployed to GitHub (what CI does).  To test a
# JLL built locally with BinaryBuilder (`julia build_tarballs.jl --deploy=local <triplet>`),
# point TANDEM_JLL_LOCAL_PATH at the generated JLL directory:
#   TANDEM_JLL_LOCAL_PATH=~/.julia/dev/Tandem_jll julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["TANDEM_JLL_LOCAL_PATH"]); Pkg.test()'
if haskey(ENV, "TANDEM_JLL_LOCAL_PATH")
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

mpirun(n::Int, cmd::Cmd) = addenv(
    `$(mpiexec_cmd) -n $n $cmd`,
    Tandem_jll.JLLWrappers.LIBPATH_env => join((Tandem_jll.LIBPATH[], MPI_LIBPATH[]), pathsep),
    "OMP_NUM_THREADS" => "1",
)

serial(cmd::Cmd) = addenv(cmd, "OMP_NUM_THREADS" => "1")

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
                @info @sprintf("%s: L2 error = %.3e", nm, err)
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
                @info @sprintf("serial %.6e vs 2 ranks %.6e", err_ser, err_par)
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

            cmd = Cmd(serial(`$(Tandem_jll.tandem_2d_p2()) mms1.toml`); dir=tmp)
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
