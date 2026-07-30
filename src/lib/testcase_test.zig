//! Tests for the analytic test cases and the error measure built on them.
//!
//! The convergence study proper lives in `src/convergence.zig` and runs under
//! `zig build convergence`: it needs an optimized build and minutes of CPU, so
//! it has no business here. What these do is pin down the pieces it rests on --
//! that the exact solutions really are exact, that `l2Error` measures what it
//! claims -- plus one small refinement sweep, so a broken operator fails the
//! suite rather than waiting for someone to run the study.

const std = @import("std");
const testing = std.testing;
const Io = std.Io;

const cfg = @import("config.zig");
const driver = @import("driver.zig");
const flux = @import("flux.zig");
const testcase = @import("testcase.zig");

fn testConfig(order: u8, n: u32, tc: u32, equation: cfg.Equation, half: f64) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = order };
    config.equation = .{
        .equation = equation,
        .viscous = false,
        .advdiff_A = .{ 1.0, 0.5, 0.0 },
        .advdiff_D = 0.0,
    };
    config.time = .{ .dt_scheme = .rk44, .n_steps = 0, .dt = 1e-3 };
    config.restart = null;
    config.output = .{ .output_prefix = "test", .write_freq = 0, .report_freq = 0 };
    config.test_case = .{ .test_case = tc, .err_field = 0, .n_qpts_1d = order + 3 };
    config.flux = .{};
    config.gas_properties = .{};
    config.freestream = .{};
    config.wall_conditions = .{};
    config.boundary_conditions = .{};
    config.signals = .{};
    config.create_mesh = .{
        .nx = n,
        .ny = n,
        .xmin = -half,
        .xmax = half,
        .ymin = -half,
        .ymax = half,
        .bc_bottom = .periodic,
        .bc_top = .periodic,
        .bc_left = .periodic,
        .bc_right = .periodic,
    };
    return config;
}

// ---------------------------------------------------------------------------
// The exact solutions
// ---------------------------------------------------------------------------

test "the isentropic vortex is isentropic" {
    // What makes it an exact solution of the Euler equations rather than an
    // approximate one: entropy p / rho^gamma is uniform, so there is nothing
    // for the energy equation to transport.
    const p: flux.FlowParams = .{ .gamma = 1.4 };
    const bounds: [2][2]f64 = .{ .{ -5, 5 }, .{ -5, 5 } };

    for ([_]testcase.TestCase{ .shu_vortex, .vincent_vortex }) |tc| {
        var reference: ?f64 = null;
        for ([_][2]f64{ .{ 0, 0 }, .{ 0.5, -0.3 }, .{ 1.5, 1.0 }, .{ -2.0, 3.0 } }) |xy| {
            const u = testcase.exactState(tc, p, xy[0], xy[1], 0.0, bounds);

            const rho = u[0];
            const ke = 0.5 * (u[1] * u[1] + u[2] * u[2]) / rho;
            const pres = (p.gamma - 1.0) * (u[3] - ke);
            const entropy = pres / std.math.pow(f64, rho, p.gamma);

            if (reference) |s0| {
                try testing.expectApproxEqRel(s0, entropy, 1e-14);
            } else {
                reference = entropy;
                try testing.expect(entropy > 0.0);
            }
        }
    }
}

test "the exact solution convects without changing shape" {
    // The vortex is steady in the frame moving with the free stream, which is
    // the whole basis for measuring error at t > 0. Anything sampled at
    // `x + v t` and time `t` must match the same point at time 0.
    const p: flux.FlowParams = .{ .gamma = 1.4 };
    const bounds: [2][2]f64 = .{ .{ -5, 5 }, .{ -5, 5 } };
    const vel: [2]f64 = .{ 1.0, 1.0 }; // Shu vortex

    for ([_]f64{ 0.7, 3.0, 10.0, 23.0 }) |t| {
        for ([_][2]f64{ .{ 0, 0 }, .{ 0.8, -1.2 }, .{ 2.0, 2.0 } }) |xy| {
            const now = testcase.exactState(
                .shu_vortex,
                p,
                xy[0] + vel[0] * t,
                xy[1] + vel[1] * t,
                t,
                bounds,
            );
            const start = testcase.exactState(.shu_vortex, p, xy[0], xy[1], 0.0, bounds);
            for (0..4) |n| try testing.expectApproxEqAbs(start[n], now[n], 1e-13);
        }
    }
}

test "a case belongs to one equation set" {
    // A vortex under advection-diffusion, or a sine wave under Euler, is a
    // configuration mistake -- better to say so than to quietly run the wrong
    // initial condition.
    var euler = testConfig(2, 4, 1, .euler_ns, 5.0);
    try testing.expectEqual(testcase.TestCase.shu_vortex, try testcase.TestCase.fromConfig(&euler));

    euler.equation.equation = .adv_diff;
    try testing.expectError(error.UnsupportedTestCase, testcase.TestCase.fromConfig(&euler));

    var advdiff = testConfig(2, 4, 2, .adv_diff, 1.0);
    try testing.expectEqual(testcase.TestCase.sine_wave, try testcase.TestCase.fromConfig(&advdiff));

    advdiff.equation.equation = .euler_ns;
    try testing.expectError(error.UnsupportedTestCase, testcase.TestCase.fromConfig(&advdiff));

    // A uniform state suits either
    var uniform = testConfig(2, 4, 0, .euler_ns, 1.0);
    try testing.expectEqual(testcase.TestCase.uniform, try testcase.TestCase.fromConfig(&uniform));
    uniform.equation.equation = .adv_diff;
    try testing.expectEqual(testcase.TestCase.uniform, try testcase.TestCase.fromConfig(&uniform));

    uniform.test_case.test_case = 7;
    try testing.expectError(error.UnsupportedTestCase, testcase.TestCase.fromConfig(&uniform));
}

// ---------------------------------------------------------------------------
// The error measure
// ---------------------------------------------------------------------------

test "l2Error is zero when the solution is the exact one" {
    const gpa = testing.allocator;

    // Not a tautology: the error is measured at quadrature points, which are
    // not the solution points, so this only comes out at roundoff if
    // `oppE_qpts` and the coordinate mapping agree with each other.
    // The residue is the interpolation error of the initial collocation, which
    // a high order over a small domain makes very small indeed.
    const config = testConfig(5, 8, 2, .adv_diff, 1.0);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    try testing.expect(try run.solver.l2Error(gpa) < 1e-7);
}

test "l2Error measures the size of a deliberate offset" {
    const gpa = testing.allocator;

    // Shifting the whole solution by a constant must show up as exactly that
    // constant in an RMS norm, which pins the volume normalization.
    const config = testConfig(3, 4, 2, .adv_diff, 1.0);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    const s = &run.solver;
    const offset = 0.25;
    for (0..s.quad.ele.n_spts) |spt| {
        for (0..s.n_eles) |e| s.u_spts.at(spt, 0, e).* += offset;
    }

    // Not exactly `offset`: the collocation error of the initial condition adds
    // in quadrature, contributing e^2 / (2 offset) with e ~ 8e-4 here.
    try testing.expectApproxEqAbs(offset, try s.l2Error(gpa), 1e-5);
}

test "a case with no exact solution says so" {
    const gpa = testing.allocator;

    const config = testConfig(2, 4, 0, .euler_ns, 1.0);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    try testing.expectError(error.NoExactSolution, run.solver.l2Error(gpa));
}

test "measuring error without a quadrature rule says so" {
    const gpa = testing.allocator;

    // `Loader.initialize` zeroes n_qpts_1d when error_freq is 0, so this is the
    // state a case that never asks for error ends up in.
    var config = testConfig(2, 4, 2, .adv_diff, 1.0);
    config.test_case.n_qpts_1d = 0;

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    try testing.expectError(error.NoQuadraturePoints, run.solver.l2Error(gpa));
}

// ---------------------------------------------------------------------------
// Order of accuracy
// ---------------------------------------------------------------------------

test "the scheme converges at its design order" {
    const gpa = testing.allocator;

    // The advected sine wave, which is exactly periodic on [-1,1]^2 and so
    // carries no modelling error of its own -- the measured rate is the
    // discretization's and nothing else. Free-stream preservation cannot see
    // any of this: a constant is in the span of every basis, so it survives
    // operators that are wrong.
    //
    // Kept small enough to belong in the suite. `zig build convergence` runs
    // the same thing on finer meshes, and adds the vortex.
    const cases = [_]struct { order: u8, expect: f64 }{
        .{ .order = 1, .expect = 2.0 },
        .{ .order = 2, .expect = 3.0 },
        .{ .order = 3, .expect = 4.0 },
    };

    for (cases) |c| {
        var errors: [2]f64 = undefined;
        for ([_]u32{ 4, 8 }, 0..) |n, i| {
            var config = testConfig(c.order, n, 2, .adv_diff, 1.0);

            // One dt for both meshes, sized for the finer, so the temporal
            // error is identical and cannot be mistaken for a spatial rate.
            const n_steps: u32 = 40;
            config.time.dt = 0.1 / @as(f64, @floatFromInt(n_steps));
            config.time.n_steps = n_steps;

            var run: driver.Run = undefined;
            try run.init(gpa, testing.io, &config, .{});
            defer run.deinit();

            for (0..n_steps) |_| try run.solver.update();
            errors[i] = try run.solver.l2Error(gpa);
        }

        const rate = std.math.log2(errors[0] / errors[1]);
        if (rate < c.expect - 0.4) {
            std.debug.print(
                "order {d}: expected rate {d:.1}, measured {d:.2} ({e:.3} -> {e:.3})\n",
                .{ c.order, c.expect, rate, errors[0], errors[1] },
            );
            return error.ConvergenceRateTooLow;
        }
    }
}
