//! Order-of-accuracy study: refine a periodic mesh under an exact solution and
//! check the error falls at the rate the scheme claims.
//!
//! This is the test free-stream preservation cannot do. A uniform state is
//! preserved by plenty of wrong schemes -- constants are in the span of every
//! basis, so the operators barely get exercised. An isentropic vortex is a
//! genuinely varying exact solution of the Euler equations, so the error it
//! leaves behind measures the whole discretization at once: `oppD`, `oppDiv`,
//! the DFR correction, the metric terms, the Rusanov flux and the periodic
//! pairing all have to be right together to see `h^(p+1)`.
//!
//! Run with `zig build convergence -- [orders...]` (default 1 2 3 4).

const std = @import("std");
const Io = std.Io;
const flurry = @import("flurry");
const cfg = flurry.config;
const Run = flurry.driver.Run;

/// Cell counts per side, refining by two each time.
const coarse = [_]u32{ 8, 16, 32 };

/// One level finer, for a solution with a feature to resolve before the rate
/// means anything.
const fine = [_]u32{ 16, 32, 64 };

/// The Shu vortex convects at (1, 1), so it crosses this box in 10 time units.
/// The error is measured well short of that -- a full period is a nicer story
/// but says nothing more about the rate.
const domain = 5.0;

/// Time to measure the error at. `--tfinal 0` takes no steps at all, which
/// measures the initial collocation on its own -- useful for telling a bad
/// operator apart from a bad initial condition.
var tfinal: f64 = 1.0;

/// Conserved variable the error is measured in. Density by default: on a finite
/// domain it is the field least polluted by the vortex's own non-periodicity.
var err_field: u32 = 0;

/// Step size, as a fraction of the explicit stability limit `h / (2p + 1)`.
///
/// The same `dt` is used across a refinement sweep, chosen from the *finest*
/// mesh, so the temporal error is identical at every point on the curve and
/// cannot be mistaken for a spatial rate.
const cfl = 0.1;

/// One case in the study: what it is, and the domain it is exactly periodic on.
const Study = struct {
    name: []const u8,
    test_case: cfg.TestCase,
    equation: cfg.Equation,
    /// Half-width of the square domain
    half: f64,
    /// Fastest wave speed, for sizing the step
    speed: f64,
    field: []const u8,
    /// Cell counts to sweep. A rate only means something once the solution's
    /// smallest feature is resolved: the vortex core has radius 1, so on the
    /// 8x8 mesh of a [-5,5] domain it spans 1.6 cells and the measured rate is
    /// pre-asymptotic noise.
    meshes: []const u32,
};

const studies = [_]Study{
    // Exactly periodic, so the only error is the discretization's own. When a
    // measured rate is in doubt, this is the one to believe.
    .{
        .name = "advected sine wave",
        .test_case = .sine_wave,
        .equation = .adv_diff,
        .half = 1.0,
        .speed = 1.5,
        .field = "u",
        .meshes = &coarse,
    },
    // The real target: a nonlinear system, where the flux, the wave speeds and
    // the Riemann solver all participate.
    .{
        .name = "Shu isentropic vortex",
        .test_case = .shu_vortex,
        .equation = .euler_ns,
        .half = domain,
        .speed = 3.0,
        .field = "density",
        .meshes = &fine,
    },
    // A second, independent Euler solution: weaker, subsonic in the mean and
    // with a wider core. If the two vortices disagree the fault is the case,
    // not the Euler path they share.
    .{
        .name = "Vincent isentropic vortex",
        .test_case = .vincent_vortex,
        .equation = .euler_ns,
        .half = 10.0,
        .speed = 3.5,
        .field = "density",
        .meshes = &fine,
    },
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var buf: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writer(io, &buf);
    const w = &stdout.interface;
    defer w.flush() catch {};

    var orders: std.ArrayList(u8) = .empty;
    defer orders.deinit(gpa);

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.skip();
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--field")) {
            err_field = try std.fmt.parseInt(u32, args.next() orelse return error.MissingValue, 10);
            continue;
        }
        if (std.mem.eql(u8, a, "--tfinal")) {
            tfinal = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingValue);
            continue;
        }
        try orders.append(gpa, try std.fmt.parseInt(u8, a, 10));
    }
    if (orders.items.len == 0) try orders.appendSlice(gpa, &.{ 1, 2, 3, 4 });

    for (studies) |study| {
        try w.print(
            \\
            \\ ===================================================
            \\ {s}
            \\
            \\ domain    [-{d}, {d}]^2, periodic
            \\ measured  L2 error in {s} at t = {d}
            \\
        , .{ study.name, study.half, study.half, study.field, tfinal });

        var worst_shortfall: f64 = 0.0;

        for (orders.items) |order| {
            const expected: f64 = @floatFromInt(order + 1);
            try w.print("\n order {d}   (expect {d:.1})\n", .{ order, expected });
            try w.print("{s:>6}{s:>12}{s:>16}{s:>10}\n", .{ "N", "h", "L2 error", "order" });

            // Sized for the finest mesh, so every run on this curve shares it
            const h_min = 2.0 * study.half / @as(f64, @floatFromInt(study.meshes[study.meshes.len - 1]));
            const dt_target = cfl * h_min /
                (study.speed * @as(f64, @floatFromInt(2 * order + 1)));
            const n_steps: u32 = if (tfinal <= 0.0) 0 else @intFromFloat(@ceil(tfinal / dt_target));
            const dt = if (n_steps == 0) dt_target else tfinal / @as(f64, @floatFromInt(n_steps));

            var prev_err: ?f64 = null;
            for (study.meshes) |n| {
                const h = 2.0 * study.half / @as(f64, @floatFromInt(n));
                const err = try runCase(gpa, io, study, order, n, dt, n_steps);

                try w.print("{d:>6}{e:>12.3}{e:>16.4}", .{ n, h, err });
                if (prev_err) |p| {
                    const rate = std.math.log2(p / err);
                    try w.print("{d:>10.2}\n", .{rate});
                    worst_shortfall = @max(worst_shortfall, expected - rate);
                } else {
                    try w.print("{s:>10}\n", .{"-"});
                }
                prev_err = err;
                try w.flush();
            }
        }

        try w.print("\n worst shortfall against the expected rate: {d:.2}\n", .{worst_shortfall});
    }
}

/// One run to `tfinal`, returning the L2 error in the study's field.
fn runCase(
    gpa: std.mem.Allocator,
    io: Io,
    study: Study,
    order: u8,
    n: u32,
    dt: f64,
    n_steps: u32,
) !f64 {
    var config = studyConfig(study, order, n, dt, n_steps);

    var run: Run = undefined;
    try run.init(gpa, io, &config, .{});
    defer run.deinit();

    // Driven directly rather than through `Run.run`, so the loop stops on
    // exactly `n_steps` with nothing printed in between.
    for (0..n_steps) |_| try run.solver.update();

    return run.solver.l2Error(gpa);
}

fn studyConfig(study: Study, order: u8, n: u32, dt: f64, n_steps: u32) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = order };
    config.equation = .{
        .equation = study.equation,
        .viscous = false,
        // Not aligned with the mesh, so the tangential direction is exercised
        .advdiff_A = .{ 1.0, 0.5, 0.0 },
        .advdiff_D = 0.0,
    };
    config.time = .{ .dt_scheme = .rk44, .n_steps = n_steps, .dt = dt };
    config.restart = null;
    config.output = .{ .output_prefix = "study", .write_freq = 0, .report_freq = 0 };
    // Two extra points beyond the solution rule, so the quadrature resolves the
    // error rather than the error resolving the quadrature.
    config.test_case = .{ .test_case = study.test_case, .err_field = err_field, .n_qpts_1d = order + 3 };
    config.flux = .{};
    config.gas_properties = .{};
    config.freestream = .{};
    config.wall_conditions = .{};
    config.boundary_conditions = .{};
    config.signals = .{};
    config.create_mesh = .{
        .nx = n,
        .ny = n,
        .xmin = -study.half,
        .xmax = study.half,
        .ymin = -study.half,
        .ymax = study.half,
        .bc_bottom = .periodic,
        .bc_top = .periodic,
        .bc_left = .periodic,
        .bc_right = .periodic,
    };
    return config;
}
