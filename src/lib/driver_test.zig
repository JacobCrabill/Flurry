const std = @import("std");
const testing = std.testing;
const Io = std.Io;

const cfg = @import("config.zig");
const driver = @import("driver.zig");
const Run = driver.Run;

/// A generated Cartesian box, ready to hand to `Run.init`.
///
/// `create_mesh` defaults every side to periodic, so a case that does not mean
/// to be periodic has to say so; these fixtures always set all four explicitly.
fn testConfig(order: u8, nx: u32, ny: u32, n_steps: u32, dt: ?f64) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = order };
    config.equation = .{
        .equation = .euler_ns,
        .viscous = false,
        .advdiff_A = .{ 1.0, 0.5, 0.0 },
        .advdiff_D = 0.1,
    };
    config.time = .{ .dt_scheme = .rk44, .n_steps = n_steps, .dt = dt };
    config.restart = null;
    config.output = .{ .output_prefix = "test", .write_freq = 0, .report_freq = 0 };
    config.test_case = .{ .test_case = .uniform, .err_field = 0, .n_qpts_1d = 0 };
    config.flux = .{};
    config.gas_properties = .{};
    config.freestream = .{ .mach_fs = 0.3, .norm_fs = .{ 1.0, 0.0, 0.0 }, .fix_vis = true };
    config.wall_conditions = .{};
    config.boundary_conditions = .{};
    config.signals = .{};
    config.create_mesh = .{
        .nx = nx,
        .ny = ny,
        .xmin = 0.0,
        .xmax = 2.0,
        .ymin = 0.0,
        .ymax = 2.0,
        .bc_bottom = .periodic,
        .bc_top = .periodic,
        .bc_left = .periodic,
        .bc_right = .periodic,
    };
    return config;
}

/// Run to completion, throwing the report away.
fn runQuiet(r: *Run) !void {
    var buf: [512]u8 = undefined;
    var sink: Io.Writer.Discarding = .init(&buf);
    return r.run(&sink.writer);
}

test "a run builds its mesh and solver from the config alone" {
    const config = testConfig(3, 4, 4, 0, 1e-3);

    var r: Run = undefined;
    try r.init(testing.allocator, testing.io, &config, .{});
    defer r.deinit();

    // The mesh came from `create_mesh`, and it is periodic, so it has no
    // boundary faces at all.
    try testing.expectEqual(@as(usize, 16), r.mesh.n_eles);
    try testing.expectEqual(@as(usize, 0), r.mesh.n_bnd_faces);

    try testing.expectEqual(@as(usize, 16), r.solver.element().n_spts);
    try testing.expectEqual(@as(usize, 4), r.solver.n_vars);
    try testing.expectEqual(@as(f64, 1e-3), r.solver.dt);
}

test "the time loop advances the solver by n_steps" {
    const config = testConfig(2, 4, 4, 7, 1e-3);

    var r: Run = undefined;
    try r.init(testing.allocator, testing.io, &config, .{});
    defer r.deinit();

    try runQuiet(&r);

    try testing.expectEqual(@as(u32, 7), r.solver.current_iter);
    try testing.expectApproxEqAbs(@as(f64, 7e-3), r.solver.flow_time, 1e-15);
}

test "free-stream survives a whole time integration" {
    const gpa = testing.allocator;

    // The end-to-end version of free-stream preservation: not one residual
    // evaluation but a hundred RK44 steps, so anything that accumulates -- a
    // slightly wrong periodic pairing, a metric term that is only nearly right
    // -- has room to show itself.
    const config = testConfig(3, 5, 5, 100, 1e-3);

    var r: Run = undefined;
    try r.init(gpa, testing.io, &config, .{});
    defer r.deinit();

    const s = &r.solver;
    var initial: [5]f64 = undefined;
    for (0..s.n_vars) |n| initial[n] = s.u_spts.get(0, n, 0);

    try runQuiet(&r);

    for (0..s.element().n_spts) |spt| {
        for (0..s.n_vars) |n| {
            for (0..s.n_eles) |e| {
                try testing.expectApproxEqAbs(initial[n], s.u_spts.get(spt, n, e), 1e-11);
            }
        }
    }
}

test "the run reports what it set up and how it stopped" {
    const gpa = testing.allocator;

    const config = testConfig(2, 4, 4, 4, 1e-3);

    var r: Run = undefined;
    try r.init(gpa, testing.io, &config, .{});
    defer r.deinit();

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try r.run(&out.writer);

    const text = out.written();
    for ([_][]const u8{
        "generated, 4 x 4", // the mesh
        "order 2", // the element
        "euler_ns (inviscid)", // the equation
        "rk44", // the scheme
        "res[rho]", // the report table
        "reached n_steps", // why it stopped
    }) |want| {
        if (std.mem.indexOf(u8, text, want) == null) {
            std.debug.print("missing '{s}' in:\n{s}\n", .{ want, text });
            return error.TestExpectedSubstring;
        }
    }
}

test "a run with no time step is rejected" {
    // CFL-based time stepping is not ported, so a config without `time.dt` has
    // nothing to run with. Failing here beats silently stepping by zero.
    const config = testConfig(2, 4, 4, 10, null);

    var r: Run = undefined;
    try testing.expectError(error.NoTimeStep, r.init(testing.allocator, testing.io, &config, .{}));
}

test "res_tol stops the run before n_steps" {
    const gpa = testing.allocator;

    // Free-stream leaves the residual at roundoff from the first step, so any
    // sane tolerance is met immediately -- the run must stop rather than burn
    // through all 10000 steps.
    var config = testConfig(2, 4, 4, 10_000, 1e-3);
    config.time.res_tol = 1e-6;
    config.time.res_field = 0;
    config.output.report_freq = 1;

    var r: Run = undefined;
    try r.init(gpa, testing.io, &config, .{});
    defer r.deinit();

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try r.run(&out.writer);

    try testing.expectEqual(@as(u32, 1), r.solver.current_iter);
    try testing.expect(std.mem.indexOf(u8, out.written(), "residual below res_tol") != null);
}

test "tfinal stops the run before n_steps" {
    const gpa = testing.allocator;

    var config = testConfig(2, 4, 4, 10_000, 1e-3);
    config.time.tfinal = 5e-3;

    var r: Run = undefined;
    try r.init(gpa, testing.io, &config, .{});
    defer r.deinit();

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try r.run(&out.writer);

    try testing.expectEqual(@as(u32, 5), r.solver.current_iter);
    try testing.expect(std.mem.indexOf(u8, out.written(), "reached tfinal") != null);
}

test "a diverging run fails instead of returning a NaN solution" {
    const gpa = testing.allocator;

    // A time step far past the stability limit. The point is not the exact
    // number of steps it survives but that the driver notices: a run that
    // quietly returns NaN has wasted however long it took to get there.
    var config = testConfig(3, 8, 8, 200, 1.0);
    config.output.report_freq = 1;

    var r: Run = undefined;
    try r.init(gpa, testing.io, &config, .{});
    defer r.deinit();

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try testing.expectError(error.SolutionDiverged, r.run(&out.writer));

    // ...and says so, with the last numbers still on screen
    try testing.expect(std.mem.indexOf(u8, out.written(), "residual is not finite") != null);
}
