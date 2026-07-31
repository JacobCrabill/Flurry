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
const mathf64 = @import("kernels/mathf64.zig");

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

fn testConfigBc(order: u8, n: u32, bc: cfg.BoundaryCondition) cfg.Config {
    var config = testConfig(order, n);
    config.create_mesh.?.bc_bottom = bc;
    config.create_mesh.?.bc_top = bc;
    config.create_mesh.?.bc_left = bc;
    config.create_mesh.?.bc_right = bc;
    return config;
}

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
    config.test_case = .{ .test_case = .shu_vortex, .err_field = 0, .n_qpts_1d = 0 };
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
    try gpu_run.solver.syncToHost();

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
    try gpu_run.solver.syncToHost();

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
    try gpu_run.solver.syncToHost();

    try expectClose(cpu_run.solver.u_spts.data, gpu_run.solver.u_spts.data, 1e-11);
}

test "only the arrays a dispatch binds are device-resident" {
    const gpa = testing.allocator;
    const d = try device();

    // This is the split the port advances one operator at a time, and it moves
    // every time one crosses over -- which is the point of pinning it here.
    // Resident: everything the three ported gemms bind. Not resident: the
    // geometry and the face arrays, which no dispatch touches yet.
    const config = testConfig(3, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    try testing.expect(s.deviceBufferFor(s.u_spts.data) != null); // extrapolateU
    try testing.expect(s.deviceBufferFor(s.u_fpts.data) != null);
    try testing.expect(s.deviceBufferFor(s.f_spts.data) != null); // computeDivFSpts
    try testing.expect(s.deviceBufferFor(s.f_comm.data) != null); // computeDivFFpts
    try testing.expect(s.deviceBufferFor(s.divf_spts.data) != null);
    try testing.expect(s.deviceBufferFor(s.inv_jaco_spts.data) != null); // computeFluxSpts

    // ...and the face arrays, now that the whole face path dispatches
    try testing.expect(s.deviceBufferFor(s.faces.u.data) != null);
    try testing.expect(s.deviceBufferFor(s.faces.f_comm.data) != null);
    try testing.expect(s.deviceBufferFor(s.faces.norm.data) != null);
    try testing.expect(s.deviceBufferFor(s.faces.d_a.data) != null);
    try testing.expect(s.deviceBufferFor(s.faces.wave_sp) != null);
    try testing.expect(s.deviceBufferFor(s.u_ini.data) != null); // the RK update
    try testing.expect(s.deviceBufferFor(s.jaco_det_spts.data) != null);

    try testing.expect(s.deviceBufferFor(s.nodes.data) == null);
    try testing.expect(s.deviceBufferFor(s.coord_spts.data) == null);
    try testing.expect(s.deviceBufferFor(s.jaco_spts.data) == null);
    try testing.expect(s.deviceBufferFor(s.faces.coord.data) == null);

    // ...and with no device nothing is
    var cpu_run: driver.Run = undefined;
    try cpu_run.init(gpa, testing.io, &config, .{});
    defer cpu_run.deinit();
    try testing.expect(cpu_run.solver.deviceBufferFor(cpu_run.solver.u_spts.data) == null);
}

test "host and device copies are pushed together explicitly" {
    const gpa = testing.allocator;
    const d = try device();

    // The arrays live in device-local memory now, which the host cannot read.
    // What the solver hands out is a host-side block that only matches after a
    // sync -- so writing it takes an upload and reading it takes a download.
    const config = testConfig(2, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    for (s.u_spts.data, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    try s.syncToDevice();

    try s.extrapolateU();

    // The dispatch left the host copy of its output alone...
    for (s.u_fpts.data) |v| try testing.expectEqual(@as(f64, 0.0), v);

    // ...until it is asked for
    try s.syncToHost();
    var nonzero: usize = 0;
    for (s.u_fpts.data) |v| {
        if (v != 0.0) nonzero += 1;
    }
    try testing.expect(nonzero > 0);

    // ...and the input the host wrote came back unchanged
    for (s.u_spts.data, 0..) |v, i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), v);
    }
}

test "a case with a CPU fallback keeps host-visible arrays" {
    const gpa = testing.allocator;
    const d = try device();

    // Device-local memory only works when the whole step is on the device.
    // Advection-diffusion still runs its flux on the CPU, which would read a
    // stale block, so those runs stay host-visible and pay for it.
    var config = testConfig(2, 4);
    config.equation.equation = .adv_diff;
    config.test_case.test_case = .sine_wave;
    config.create_mesh.?.xmin = -1.0;
    config.create_mesh.?.xmax = 1.0;
    config.create_mesh.?.ymin = -1.0;
    config.create_mesh.?.ymax = 1.0;

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    // Ten steps with no explicit sync anywhere: the host arrays have to be live
    // throughout, or the CPU flux would be working from stale data.
    const s = &run.solver;
    for (0..10) |_| try s.update();

    for (s.u_spts.data) |v| try testing.expect(std.math.isFinite(v));
}

// ---------------------------------------------------------------------------
// The hand-written flux kernel
// ---------------------------------------------------------------------------

test "the Euler flux kernel matches the CPU one" {
    const gpa = testing.allocator;
    const d = try device();

    // The first kernel written here rather than taken from spock, so it gets a
    // check of its own: a failure inside a whole residual would only say that
    // *something* disagreed.
    const config = testConfig(3, 5);

    var cpu_run: driver.Run = undefined;
    try cpu_run.init(gpa, testing.io, &config, .{});
    defer cpu_run.deinit();

    var gpu_run: driver.Run = undefined;
    try gpu_run.init(gpa, testing.io, &config, .{ .device = d });
    defer gpu_run.deinit();

    // Both start from the vortex, so `u_spts` already varies across the mesh
    try cpu_run.solver.computeFluxSpts();
    try gpu_run.solver.computeFluxSpts();
    try gpu_run.solver.syncToHost();

    try expectClose(cpu_run.solver.f_spts.data, gpu_run.solver.f_spts.data, 1e-13);
}

// ---------------------------------------------------------------------------
// Batching
// ---------------------------------------------------------------------------

test "a batched dispatch pair gives the same answer as two separate ones" {
    const gpa = testing.allocator;
    const d = try device();

    // `computeResidual` batches `computeFluxSpts` with `computeDivFSpts`. The
    // barrier between them is what makes the second see the first's writes; if
    // it were missing this would read stale `f_spts`.
    const config = testConfig(3, 5);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;

    // Unbatched: each dispatch submitted and waited for on its own
    try s.computeFluxSpts();
    try s.computeDivFSpts(0);
    const separate = try gpa.dupe(f64, s.divf_spts.data);
    defer gpa.free(separate);

    @memset(s.f_spts.data, 0.0);
    @memset(s.divf_spts.data, 0.0);

    try d.beginBatch();
    try s.computeFluxSpts();
    try s.computeDivFSpts(0);
    try d.submitBatch();

    try expectClose(separate, s.divf_spts.data, 1e-15);
}

test "recording a single-instance kernel twice in a batch is rejected" {
    const gpa = testing.allocator;
    const d = try device();

    // A kernel owns a single descriptor set, so a second recording would
    // overwrite the first's arguments and *both* dispatches would run with the
    // second's -- no error, a plausible wrong answer. dgemm really is dispatched
    // several times per residual and has an instance per recording; everything
    // else has one, and says so.
    const config = testConfig(2, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    try d.beginBatch();
    try s.computeFluxSpts();
    try testing.expectError(error.KernelAlreadyRecorded, s.computeFluxSpts());
    try d.submitBatch();
}

test "a batch runs out of dgemm instances rather than reusing one" {
    const gpa = testing.allocator;
    const d = try device();

    // The pool is sized for a residual's three products. Asking for more is the
    // same silent-overwrite hazard, so it is the same error.
    const config = testConfig(2, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    try d.beginBatch();
    defer d.abortBatch();

    // A residual dispatches dgemm exactly three times, and the pool is sized for
    // that; the fourth has nothing left to hand out.
    try s.extrapolateU();
    try s.computeDivFSpts(0);
    try s.computeDivFFpts(0);
    try testing.expectError(error.KernelAlreadyRecorded, s.computeDivFSpts(0));
}

test "a failed batch does not poison the device" {
    const gpa = testing.allocator;
    const d = try device();

    // Recording can fail part-way. If that left the batch open, every later
    // `beginBatch` would trip the assert and every dispatch would draw from an
    // exhausted pool -- so an error has to leave the device usable.
    const config = testConfig(2, 4);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    const s = &run.solver;
    try d.beginBatch();
    try s.extrapolateU();
    try s.computeDivFSpts(0);
    try s.computeDivFFpts(0);
    try testing.expectError(error.KernelAlreadyRecorded, s.computeDivFSpts(0));
    d.abortBatch();

    try testing.expect(!d.isBatching());
    // ...and a whole step still runs
    try s.update();
}

test "the kernels' f64 pow matches std.math" {
    // Vulkan has no f64 transcendentals, so these are hand-rolled -- and a
    // characteristic far field is built out of them, which makes them worth
    // checking against the real thing rather than assuming.
    const xs = [_]f64{ 1e-6, 0.1, 0.5, 0.9999, 1.0, 1.4, 2.0, 7.0, 1000.0, 1.0e6 };
    const ys = [_]f64{ -2.5, -1.4, -0.4, 0.0, 0.4, 1.0, 1.4, 2.5, 3.0 };

    for (xs) |x| {
        try std.testing.expectApproxEqRel(std.math.log2(x), mathf64.log2(x), 1e-15);
        for (ys) |y| {
            try std.testing.expectApproxEqRel(std.math.pow(f64, x, y), mathf64.pow(x, y), 1e-14);
            try std.testing.expectApproxEqRel(std.math.exp2(y), mathf64.exp2(y), 1e-15);
        }
    }

    // The exponents the characteristic boundary actually raises things to
    const gam = 1.4;
    for ([_]f64{ 0.3, 1.0, 1.7, 4.2 }) |rho| {
        try std.testing.expectApproxEqRel(std.math.pow(f64, rho, gam), mathf64.pow(rho, gam), 1e-14);
        try std.testing.expectApproxEqRel(
            std.math.pow(f64, rho, 1.0 / (gam - 1.0)),
            mathf64.pow(rho, 1.0 / (gam - 1.0)),
            1e-14,
        );
    }
}

test "boundary conditions on the device match the CPU" {
    const gpa = testing.allocator;
    const d = try device();

    // Every other GPU test here runs a periodic mesh, where there are no
    // boundary flux points at all and `face_bcs` never dispatches. These are the
    // conditions that do have a kernel; the characteristic one is the
    // interesting case, being where the hand-rolled f64 `pow` gets used.
    for ([_]cfg.BoundaryCondition{ .characteristic, .sup_in, .sup_out, .slip_wall, .symmetry }) |bc| {
        const config = testConfigBc(3, 5, bc);

        var cpu_run: driver.Run = undefined;
        try cpu_run.init(gpa, testing.io, &config, .{});
        defer cpu_run.deinit();

        var gpu_run: driver.Run = undefined;
        try gpu_run.init(gpa, testing.io, &config, .{ .device = d });
        defer gpu_run.deinit();

        try testing.expect(gpu_run.solver.faces.n_gfpts_bnd > 0); // or this proves nothing

        try cpu_run.solver.computeResidual(0);
        try gpu_run.solver.computeResidual(0);
        try gpu_run.solver.syncToHost();

        expectClose(cpu_run.solver.divf_spts.data, gpu_run.solver.divf_spts.data, 1e-10) catch |err| {
            std.debug.print("boundary condition: {t}\n", .{bc});
            return err;
        };
    }
}

test "a viscous wall keeps the boundary conditions on the CPU" {
    const gpa = testing.allocator;
    const d = try device();

    // No kernel covers it, so the whole step falls back rather than running a
    // condition the kernel would silently treat as a slip wall.
    var config = testConfigBc(2, 4, .adiabatic_noslip);
    config.equation.viscous = true;

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{ .device = d });
    defer run.deinit();

    try testing.expect(run.solver.deviceBufferFor(run.solver.u_spts.data) != null);
    try testing.expect(!run.solver.canBatchResidualForTest());
}

test "several steps with boundaries stay together" {
    const gpa = testing.allocator;
    const d = try device();

    // The same drift check as the periodic one, on a mesh where the boundary
    // kernel actually runs every stage.
    const config = testConfigBc(2, 5, .characteristic);

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
    try gpu_run.solver.syncToHost();

    try expectClose(cpu_run.solver.u_spts.data, gpu_run.solver.u_spts.data, 1e-10);
}
