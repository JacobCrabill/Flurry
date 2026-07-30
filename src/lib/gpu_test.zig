//! Tests for the GPU back end.
//!
//! The CPU path is the oracle throughout: every check runs the same thing both
//! ways and compares. Results are compared with a tolerance rather than for
//! equality -- the kernel is free to contract multiplies and adds into FMAs
//! where the host did not, so the last bit or two will differ.
//!
//! All of these need a Vulkan device, and skip when there isn't one, so the
//! suite still passes on a machine with no loader or no `shaderFloat64`.

const std = @import("std");
const testing = std.testing;

const cfg = @import("config.zig");
const driver = @import("driver.zig");
const gpu = @import("gpu.zig");

// ---------------------------------------------------------------------------
// Shared device
// ---------------------------------------------------------------------------

/// Deliberately not `std.testing.allocator`: the device outlives every
/// individual test, so the testing allocator would report it as a leak. Nothing
/// here is freed before the process exits, by design.
const device_gpa = std.heap.smp_allocator;

var device_state: ?gpu.Device = null;
var device_failed = false;

/// The process-wide device, created on first use.
///
/// Bringing up a Vulkan instance costs far more than any dispatch here, and the
/// test runner is sequential within a process, so one device serves them all.
fn device() !*gpu.Device {
    if (device_failed) return error.SkipZigTest;
    if (device_state == null) {
        device_state = gpu.Device.init(device_gpa, .{ .validate = true }) catch |err| {
            device_failed = true;
            if (err == error.NoComputeDevice) return error.SkipZigTest;
            return err;
        };
    }
    return &device_state.?;
}

/// Relative comparison, falling back to absolute near zero.
fn expectClose(expected: []const f64, actual: []const f64, tol: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        const scale = @max(@abs(e), @abs(a));
        const err = @abs(e - a);
        if (err > tol * @max(scale, 1.0)) {
            std.debug.print("element {d}: expected {e:.17}, got {e:.17}\n", .{ i, e, a });
            return error.TestExpectedApproxEq;
        }
    }
}

/// The reference `C = A*B` / `C += A*B`, written out plainly so a bug in the
/// solver's own `gemm` cannot hide a matching bug on the device.
fn refGemm(m: usize, n: usize, k: usize, a: []const f64, b: []const f64, c: []f64, accumulate: bool) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0.0;
            for (0..k) |kk| sum += a[i * k + kk] * b[kk * n + j];
            if (accumulate) c[i * n + j] += sum else c[i * n + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// The module and the device
// ---------------------------------------------------------------------------

test "the dgemm module reaches the binary as SPIR-V" {
    // Needs no device: this is the build wiring, not the runtime. The magic
    // number is what a driver looks at first, so a mis-wired LazyPath (an
    // object file, an empty file) fails here rather than at dispatch.
    const spv = @embedFile("spock/dgemm.spv");
    try testing.expect(spv.len > 4);
    try testing.expectEqualSlices(u8, &.{ 0x03, 0x02, 0x23, 0x07 }, spv[0..4]);
    try testing.expectEqual(@as(usize, 0), spv.len % 4); // SPIR-V is a word stream
}

test "the device comes up and names itself" {
    const d = try device();
    try testing.expect(d.name().len > 0);
}

// ---------------------------------------------------------------------------
// gemm
// ---------------------------------------------------------------------------

test "gemm on the device matches the host" {
    const gpa = testing.allocator;
    const d = try device();

    // Shapes chosen to be non-square, mutually distinct, and -- for the last --
    // to leave the final workgroup partly idle, which is where an off-by-one in
    // the bounds guard would show.
    const shapes = [_][3]usize{
        .{ 1, 1, 1 },
        .{ 4, 7, 3 },
        .{ 16, 4096, 16 }, // an order-3 quad mesh's extrapolation
        .{ 13, 129, 11 },
    };

    var prng: std.Random.DefaultPrng = .init(0x5EED);
    const rand = prng.random();

    for (shapes) |shape| {
        const m, const n, const k = shape;

        const a = try gpa.alloc(f64, m * k);
        defer gpa.free(a);
        const b = try gpa.alloc(f64, k * n);
        defer gpa.free(b);
        const got = try gpa.alloc(f64, m * n);
        defer gpa.free(got);
        const want = try gpa.alloc(f64, m * n);
        defer gpa.free(want);

        for (a) |*v| v.* = rand.floatNorm(f64);
        for (b) |*v| v.* = rand.floatNorm(f64);
        for (got, want) |*g, *w| {
            const v = rand.floatNorm(f64);
            g.* = v;
            w.* = v;
        }

        // Overwrite: whatever was in C must be gone
        try d.gemmHost(m, n, k, a, b, got, .overwrite);
        refGemm(m, n, k, a, b, want, false);
        try expectClose(want, got, 1e-13);

        // Accumulate: on top of the result just written, so a kernel that
        // ignored beta would leave the plain product behind instead
        try d.gemmHost(m, n, k, a, b, got, .accumulate);
        refGemm(m, n, k, a, b, want, true);
        try expectClose(want, got, 1e-13);
    }
}

test "the device's staging survives a shrinking call" {
    const gpa = testing.allocator;
    const d = try device();

    // Buffers are grown on demand and reused. A later, smaller product must not
    // pick up the tail of the larger one it inherited.
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const big = try gpa.alloc(f64, 3 * 64);
    defer gpa.free(big);
    for (big) |*v| v.* = 1.0;

    const out_big = try gpa.alloc(f64, 3 * 64);
    defer gpa.free(out_big);
    try d.gemmHost(3, 64, 3, &a, big, out_big, .overwrite);

    // 2x2 = A(2,1) * B(1,2); every element is a single product
    const small_a = [_]f64{ 2, 3 };
    const small_b = [_]f64{ 5, 7 };
    var out_small: [4]f64 = @splat(-1.0);
    try d.gemmHost(2, 2, 1, &small_a, &small_b, &out_small, .overwrite);
    try expectClose(&.{ 10, 14, 15, 21 }, &out_small, 1e-15);
}

// ---------------------------------------------------------------------------
// The solver operator
// ---------------------------------------------------------------------------

fn testConfig(order: u8, n: u32) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = order };
    config.equation = .{
        .equation = .euler_ns,
        .viscous = false,
        .advdiff_A = .{ 1.0, 0.5, 0.0 },
        .advdiff_D = 0.0,
    };
    config.time = .{ .dt_scheme = .rk44, .n_steps = 0, .dt = 1e-3 };
    config.restart = null;
    config.output = .{ .output_prefix = "gpu", .write_freq = 0, .report_freq = 0 };
    config.test_case = .{ .test_case = 1, .err_field = 0, .n_qpts_1d = 0 };
    config.flux = .{};
    config.gas_properties = .{};
    config.freestream = .{};
    config.wall_conditions = .{};
    config.boundary_conditions = .{};
    config.signals = .{};
    config.create_mesh = .{
        .nx = n,
        .ny = n,
        .xmin = -5.0,
        .xmax = 5.0,
        .ymin = -5.0,
        .ymax = 5.0,
        .bc_bottom = .periodic,
        .bc_top = .periodic,
        .bc_left = .periodic,
        .bc_right = .periodic,
    };
    return config;
}

test "extrapolateU on the device matches the CPU" {
    const gpa = testing.allocator;
    const d = try device();

    // A vortex, so the solution genuinely varies and a dropped term cannot
    // hide in a constant.
    const config = testConfig(3, 6);

    var cpu_run: driver.Run = undefined;
    try cpu_run.init(gpa, testing.io, &config, .{});
    defer cpu_run.deinit();
    try cpu_run.solver.extrapolateU();

    var gpu_run: driver.Run = undefined;
    try gpu_run.init(gpa, testing.io, &config, .{ .device = d });
    defer gpu_run.deinit();
    try gpu_run.solver.extrapolateU();

    try expectClose(cpu_run.solver.u_fpts.data, gpu_run.solver.u_fpts.data, 1e-13);
}

test "a residual with the operator on the device matches the CPU" {
    const gpa = testing.allocator;
    const d = try device();

    // The point of the exercise: one operator dispatched to the GPU inside an
    // otherwise unchanged step, and the answer still the same. This is what the
    // copy-in/copy-out around each dispatch buys -- and what makes it safe to
    // move the remaining operators over one at a time.
    const config = testConfig(3, 6);

    var cpu_run: driver.Run = undefined;
    try cpu_run.init(gpa, testing.io, &config, .{});
    defer cpu_run.deinit();
    try cpu_run.solver.computeResidual(0);

    var gpu_run: driver.Run = undefined;
    try gpu_run.init(gpa, testing.io, &config, .{ .device = d });
    defer gpu_run.deinit();
    try gpu_run.solver.computeResidual(0);

    try expectClose(cpu_run.solver.divf_spts.data, gpu_run.solver.divf_spts.data, 1e-11);
}

test "several steps with the operator on the device stay together" {
    const gpa = testing.allocator;
    const d = try device();

    // A single residual can agree by luck; the concern is drift. Ten RK44 steps
    // is forty dispatches feeding back into the solution each time.
    const config = testConfig(2, 5);

    var cpu_run: driver.Run = undefined;
    try cpu_run.init(gpa, testing.io, &config, .{});
    defer cpu_run.deinit();

    var gpu_run: driver.Run = undefined;
    try gpu_run.init(gpa, testing.io, &config, .{ .device = d });
    defer gpu_run.deinit();

    for (0..10) |_| {
        try cpu_run.solver.update();
        try gpu_run.solver.update();
    }

    try expectClose(cpu_run.solver.u_spts.data, gpu_run.solver.u_spts.data, 1e-11);
}

test "only the arrays a dispatch binds are device-resident" {
    const gpa = testing.allocator;
    const d = try device();

    // This is the split the port advances one operator at a time. `u_spts` and
    // `u_fpts` are bound by `extrapolateU`, so they live in device memory;
    // nothing dispatches over the geometry yet, so it does not.
    const config = testConfig(3, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    try testing.expect(s.deviceBufferFor(s.u_spts.data) != null);
    try testing.expect(s.deviceBufferFor(s.u_fpts.data) != null);

    try testing.expect(s.deviceBufferFor(s.nodes.data) == null);
    try testing.expect(s.deviceBufferFor(s.coord_spts.data) == null);
    try testing.expect(s.deviceBufferFor(s.divf_spts.data) == null);

    // ...and with no device nothing is
    var cpu_run: driver.Run = undefined;
    try cpu_run.init(gpa, testing.io, &config, .{});
    defer cpu_run.deinit();
    try testing.expect(cpu_run.solver.deviceBufferFor(cpu_run.solver.u_spts.data) == null);
}

test "device memory is still an ordinary slice to the CPU" {
    const gpa = testing.allocator;
    const d = try device();

    // The whole reason the port can proceed piecemeal: `u_spts` is a Vulkan
    // mapping, and the un-ported operations either side of a dispatch read and
    // write it exactly as before.
    const config = testConfig(2, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    for (s.u_spts.data, 0..) |*v, i| v.* = @floatFromInt(i);
    try s.extrapolateU();

    // The host writes survived the dispatch reading them...
    for (s.u_spts.data, 0..) |v, i| try testing.expectEqual(@as(f64, @floatFromInt(i)), v);

    // ...and what the dispatch wrote is visible without any read-back
    var nonzero: usize = 0;
    for (s.u_fpts.data) |v| {
        if (v != 0.0) nonzero += 1;
    }
    try testing.expect(nonzero > 0);
}
