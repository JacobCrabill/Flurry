//! Ties the pieces together: config -> mesh -> solver -> time loop.
//!
//! Flurry-cpp spread this between `main.cpp` and the `Solver` class. Keeping it
//! in the library instead of in `main.zig` means a whole run is reachable from a
//! test, and leaves `main.zig` as nothing but argument handling.

/// Largest number of conserved variables any supported equation set has
/// (3D Navier-Stokes).
const max_vars = 5;

pub const Error = error{
    /// `time.dt` is unset, and CFL-based time stepping is not ported yet
    NoTimeStep,
    /// The residual stopped being a finite number
    SolutionDiverged,
};

/// One complete run: the mesh, the solver over it, and the time loop.
///
/// The solver borrows `&Run.mesh`, so a `Run` must not be moved once it is set
/// up. `init` therefore fills in caller-provided storage rather than returning a
/// value, which makes that constraint impossible to get wrong:
///
///     var run: Run = undefined;
///     try run.init(gpa, io, &parsed.value);
///     defer run.deinit();
///     try run.run(&stdout.interface);
pub const Run = struct {
    gpa: std.mem.Allocator,
    io: Io,
    config: *const cfg.Config,
    mesh: Geo,
    solver: Solver,

    /// Whether the "nothing is written" notice has already been printed
    warned_no_output: bool = false,
    /// Whether the "no exact solution" notice has already been printed
    warned_no_error: bool = false,

    /// Build the mesh, set up the solver and apply the initial condition.
    ///
    /// `config` must outlive the run: the solver keeps a pointer to it, and the
    /// mesh's shallow copy still has slices pointing into the config's arena.
    pub fn init(r: *Run, gpa: std.mem.Allocator, io: Io, config: *const cfg.Config) !void {
        // Nothing to fall back on: a CFL-derived step size is not ported yet, and
        // a dt of zero would step forever without advancing.
        if (config.time.dt == null) return error.NoTimeStep;

        r.* = .{
            .gpa = gpa,
            .io = io,
            .config = config,
            .mesh = .{ .gpa = gpa, .io = io, .config = config.* },
            .solver = undefined,
        };
        errdefer r.mesh.deinit();

        // A `create_mesh` section is what selects mesh generation; without one
        // the mesh comes from `core.mesh_file`.
        if (config.create_mesh != null) {
            try r.mesh.createMesh();
        } else {
            try r.mesh.readGmsh(config.core.mesh_file);
        }
        try r.mesh.processConnectivity();
        try r.mesh.setupGlobalFpts(@as(usize, config.core.order) + 1);

        r.solver = try Solver.init(gpa, config, &r.mesh);
        try r.solver.initializeU();
    }

    pub fn deinit(r: *Run) void {
        r.solver.deinit();
        r.mesh.deinit();
    }

    /// Step until one of the stopping conditions is met, reporting to `w`.
    ///
    /// Stops at `time.n_steps`, at `time.tfinal`, or once the residual in
    /// `time.res_field` falls below `time.res_tol` -- whichever comes first.
    ///
    /// A run diverging into NaN is treated as a failure rather than a result.
    /// That is only noticed when the residual is looked at, which happens every
    /// step while `time.res_tol` is set and otherwise only when a report is due;
    /// with both `res_tol` and `output.report_freq` at zero the loop runs blind.
    pub fn run(r: *Run, w: *Io.Writer) !void {
        const t = r.config.time;
        const o = r.config.output;
        const s = &r.solver;

        try r.writeSummary(w);

        const started = Io.Timestamp.now(r.io, .awake);
        var res: [max_vars]f64 = @splat(0.0);

        // The residual of the initial condition, so the first reported line says
        // something real rather than the zeros `divf_spts` starts life with. The
        // stopping conditions deliberately do not apply to it: a run that exits
        // before taking a step looks broken even when it is right.
        try s.computeResidual(0);
        s.residualNorm(0, &res);

        try r.writeReportHeader(w);
        try r.writeReport(w, started, &res);

        var last_reported: u32 = s.current_iter;
        var reason: []const u8 = "reached n_steps";

        for (0..t.n_steps) |_| {
            // A CFL-derived dt would be recomputed here, once per step.
            try s.update();

            if (o.write_freq != 0 and s.current_iter % o.write_freq == 0) {
                try r.writeSolution(w);
            }

            if (o.error_freq != 0 and s.current_iter % o.error_freq == 0) {
                try r.writeError(w);
            }

            const due = o.report_freq != 0 and s.current_iter % o.report_freq == 0;
            if (due or t.res_tol > 0.0) {
                // `divf_spts` holds one residual per RK stage; stage 0 is the one
                // evaluated on the solution at the start of the step just taken,
                // which is the dU/dt convergence is measured against.
                s.residualNorm(0, &res);
                if (due) {
                    try r.writeReport(w, started, &res);
                    last_reported = s.current_iter;
                }
                if (residualStop(res[0..s.n_vars], t)) |why| {
                    reason = why;
                    break;
                }
            }

            if (s.flow_time >= t.tfinal) {
                reason = "reached tfinal";
                break;
            }
        }

        // The last step is always worth seeing, however the loop ended.
        if (last_reported != s.current_iter) {
            s.residualNorm(0, &res);
            try r.writeReport(w, started, &res);
            if (residualStop(res[0..s.n_vars], t)) |why| reason = why;
        }

        try w.print("\n {s} after {d} steps, t = {e:.6}\n", .{ reason, s.current_iter, s.flow_time });

        // Worth having whatever the cadence was: the error at the end is the
        // number a convergence study is after.
        if (o.error_freq != 0) try r.writeError(w);
        try w.flush();

        // Reported first, so the numbers that led here are on screen.
        for (res[0..s.n_vars]) |v| {
            if (!std.math.isFinite(v)) return error.SolutionDiverged;
        }
    }

    /// Why the loop should stop, if it should. A non-finite residual is left to
    /// the caller so the reason can be reported before the run fails.
    fn residualStop(res: []const f64, t: cfg.TimeConfig) ?[]const u8 {
        for (res) |v| {
            if (!std.math.isFinite(v)) return "residual is not finite";
        }
        if (t.res_tol > 0.0 and t.res_field < res.len and res[t.res_field] < t.res_tol) {
            return "residual below res_tol";
        }
        return null;
    }

    // ---- Reporting ----

    /// What was set up, before any of it starts moving.
    fn writeSummary(r: *Run, w: *Io.Writer) !void {
        const c = r.config;
        const s = &r.solver;
        const ele = &s.quad.ele;

        try w.print("\n flurry -- flux reconstruction\n\n", .{});

        if (c.create_mesh) |cm| {
            if (c.core.n_dims == 3) {
                try w.print(" mesh      generated, {d} x {d} x {d}\n", .{ cm.nx, cm.ny, cm.nz });
            } else {
                try w.print(" mesh      generated, {d} x {d}\n", .{ cm.nx, cm.ny });
            }
        } else {
            try w.print(" mesh      {s}\n", .{c.core.mesh_file});
        }

        try w.print(" cells     {d} ({d} verts, {d} faces: {d} interior, {d} boundary)\n", .{
            r.mesh.n_eles,      r.mesh.n_verts,     r.mesh.n_faces,
            r.mesh.n_int_faces, r.mesh.n_bnd_faces,
        });
        try w.print(" element   order {d}, {d} solution points, {d} flux points\n", .{
            c.core.order, ele.n_spts, ele.n_fpts,
        });
        try w.print(" equation  {t} ({s}), {d} variables in {d}D\n", .{
            c.equation.equation,
            if (c.equation.viscous) "viscous" else "inviscid",
            s.n_vars,
            s.n_dims,
        });
        try w.print(" stepping  {t}, dt = {e:.4}, up to {d} steps\n", .{
            c.time.dt_scheme, s.dt, c.time.n_steps,
        });

        // The flux point pairing is an assumption the mesh has to justify; say
        // so up front rather than letting a bad mesh look like bad physics.
        //
        // The periodic-aware measure forgives a whole domain length, so it is
        // only used where a periodic boundary makes that necessary -- otherwise
        // it would hide exactly the kind of gross mispairing this is here for.
        var periodic = false;
        for (r.mesh.bc_list.items) |bc| {
            if (bc == .periodic) periodic = true;
        }
        const pairing = if (periodic) s.fptPairingErrorPeriodic() else s.fptPairingError();
        try w.print(" pairing   flux point mismatch {e:.3}\n", .{pairing});
    }

    fn writeReportHeader(r: *Run, w: *Io.Writer) !void {
        const s = &r.solver;
        try w.print("\n{s:>8}{s:>14}", .{ "iter", "time" });
        for (0..s.n_vars) |n| {
            var buf: [16]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "res[{s}]", .{varName(r.config.equation.equation, s.n_dims, n)});
            try w.print("{s:>14}", .{name});
        }
        try w.print("{s:>10}\n", .{"wall (s)"});
    }

    /// One line of the report table. `res` is whatever the caller last measured
    /// -- see `run`, which decides how often that is worth doing.
    fn writeReport(r: *Run, w: *Io.Writer, started: Io.Timestamp, res: *const [max_vars]f64) !void {
        const s = &r.solver;

        const elapsed = started.durationTo(Io.Timestamp.now(r.io, .awake));
        const secs = @as(f64, @floatFromInt(elapsed.nanoseconds)) * 1e-9;

        try w.print("{d:>8}{e:>14.4}", .{ s.current_iter, s.flow_time });
        for (0..s.n_vars) |n| try w.print("{e:>14.4}", .{res[n]});
        try w.print("{d:>10.2}\n", .{secs});
        try w.flush();
    }

    /// L2 error against the exact solution, when the case has one.
    fn writeError(r: *Run, w: *Io.Writer) !void {
        const err = r.solver.l2Error(r.gpa) catch |e| switch (e) {
            // A case with no exact solution is a legitimate thing to run; say
            // so once rather than failing the run or repeating every interval.
            error.NoExactSolution, error.NoQuadraturePoints => {
                if (r.warned_no_error) return;
                r.warned_no_error = true;
                try w.print("\n note: error_freq is set, but this case has no exact solution ({t})\n\n", .{e});
                return;
            },
            else => return e,
        };
        try w.print("          L2 error[{s}] = {e:.6}\n", .{
            varName(r.config.equation.equation, r.solver.n_dims, r.config.test_case.err_field),
            err,
        });
    }

    /// Write the solution for ParaView, if the config asked for it.
    fn writeSolution(r: *Run, w: *Io.Writer) !void {
        if (!r.config.output.write_paraview) {
            if (r.warned_no_output) return;
            r.warned_no_output = true;
            try w.writeAll(
                "\n note: write_freq is set but write_paraview is off, so nothing is written\n\n",
            );
            return;
        }

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try vtu.writeSolution(
            &r.solver,
            r.gpa,
            r.io,
            Io.Dir.cwd(),
            r.config.output.output_prefix,
            &path_buf,
        );
        try w.print(" wrote {s}\n", .{path});
    }
};

/// Display name of conserved variable `n`.
fn varName(equation: cfg.Equation, n_dims: usize, n: usize) []const u8 {
    return switch (equation) {
        .adv_diff => "u",
        .euler_ns => switch (n) {
            0 => "rho",
            1 => "rhou",
            2 => "rhov",
            3 => if (n_dims == 3) "rhow" else "E",
            else => "E",
        },
    };
}

const std = @import("std");
const Io = std.Io;

const cfg = @import("config.zig");
const Geo = @import("geo.zig").Geo;
const Solver = @import("solver.zig").Solver;
const vtu = @import("vtu.zig");
