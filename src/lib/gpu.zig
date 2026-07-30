//! GPU compute back end: a Vulkan device, and the kernels the solver dispatches
//! onto it.
//!
//! The solver's operator applications are all one shape -- a dense row-major
//! `C = A*B` with `A` the reference-element operator and `B` the solution laid
//! out with the element index last -- which is exactly what spock's prebuilt
//! `dgemm` computes. So the first operators to move across need no kernel of
//! our own; `Mode.overwrite` and `.accumulate` are just `beta = 0` and `1`.
//!
//! Everything here is f64. Some GPUs support `shaderFloat64` poorly or not at
//! all, and this will refuse to run on those -- which is the right trade for a
//! solver whose accuracy claims are the point.
//!
//! State of the port: `Solver.extrapolateU` only, and it still copies its
//! operands in and out around every dispatch. That cost exists solely because
//! the solver's arrays are ordinary host allocations; moving them into
//! device-visible memory is the next step and deletes the copies outright.

const std = @import("std");
const spock = @import("spock");

const dgemm = spock.kernels.blas.dgemm;
const dgemm_spv = @embedFile("spock/dgemm.spv");

pub const Error = error{
    /// No Vulkan loader, or no device on it with compute support
    NoComputeDevice,
    /// The device rejected something, or a dispatch failed. The underlying
    /// Vulkan error is reported to stderr -- it is not in this set because
    /// hoisting the whole of `vk.Error` into every caller's signature buys
    /// nothing they could act on.
    DeviceFailure,
    OutOfMemory,
};

/// Whether a product overwrites its destination or accumulates into it,
/// matching the CPU `gemm`'s `Mode`.
pub const Mode = enum { overwrite, accumulate };

/// Which GPU to take when there is more than one.
///
/// Mirrors spock's own `Pick`, which it does not export -- `Context.Options` is
/// only reachable as an inferred anonymous literal.
pub const Pick = enum { first, largest };

pub const Options = struct {
    pick: Pick = .largest,
    /// Turn on the Vulkan validation layer, if it is installed
    validate: bool = false,
};

/// A compute device plus the kernels compiled against it.
///
/// The `Context` is on the heap rather than inline. Every spock `Kernel` and
/// `Buffer` keeps a `*Context`, so an inline one would make a `Device` unsafe to
/// move -- copy it out of the local it was built in and every kernel is left
/// pointing at a dead stack frame, which shows up as a null function pointer at
/// the first dispatch rather than anywhere near the mistake. One allocation
/// removes the hazard instead of documenting it.
pub const Device = struct {
    gpa: std.mem.Allocator,
    ctx: *spock.Context,
    dgemm_kernel: spock.Kernel,

    /// Staging for `gemmHost`, grown on demand and reused across calls.
    stage_a: Scratch = .{},
    stage_b: Scratch = .{},
    stage_c: Scratch = .{},

    pub fn init(gpa: std.mem.Allocator, opts: Options) Error!Device {
        const ctx = try gpa.create(spock.Context);
        errdefer gpa.destroy(ctx);

        ctx.* = spock.Context.init(gpa, .{
            .pick = switch (opts.pick) {
                .first => .first,
                .largest => .largest,
            },
            .validate = opts.validate,
            .app_name = "flurry",
        }) catch |err| return translate(err, "creating the compute context");
        errdefer ctx.deinit();

        // spock's dgemm exports its entry point under its own name, not "main"
        const kernel = spock.Kernel.create(ctx, .{
            .spirv = dgemm_spv,
            .entry = "dgemm",
            .buffers = 3,
            .push_constant_size = @sizeOf(dgemm.PushConstants),
        }) catch |err| return translate(err, "building the dgemm pipeline");

        return .{ .gpa = gpa, .ctx = ctx, .dgemm_kernel = kernel };
    }

    pub fn deinit(d: *Device) void {
        d.stage_a.deinit();
        d.stage_b.deinit();
        d.stage_c.deinit();
        d.dgemm_kernel.deinit();
        d.ctx.deinit();
        d.gpa.destroy(d.ctx);
    }

    pub fn name(d: *const Device) []const u8 {
        return d.ctx.deviceName();
    }

    /// `C = A*B`, or `C += A*B`, over host slices: the GPU counterpart of the
    /// solver's `gemm`, with the same argument order.
    ///
    /// `A` is (m, k), `B` is (k, n), `C` is (m, n), all row-major and densely
    /// packed. The operands are copied into device-visible staging around the
    /// dispatch, which is the whole per-call overhead of this first step.
    pub fn gemmHost(
        d: *Device,
        m: usize,
        n: usize,
        k: usize,
        a: []const f64,
        b: []const f64,
        c: []f64,
        mode: Mode,
    ) Error!void {
        std.debug.assert(a.len >= m * k);
        std.debug.assert(b.len >= k * n);
        std.debug.assert(c.len >= m * n);
        if (m == 0 or n == 0 or k == 0) return;

        const a_buf = try d.stage_a.ensure(d.ctx, m * k);
        const b_buf = try d.stage_b.ensure(d.ctx, k * n);
        const c_buf = try d.stage_c.ensure(d.ctx, m * n);

        a_buf.copyFromHost(a[0 .. m * k]);
        b_buf.copyFromHost(b[0 .. k * n]);
        // With beta = 0 the kernel writes every element, so there is nothing
        // worth uploading.
        if (mode == .accumulate) c_buf.copyFromHost(c[0 .. m * n]);

        const pc: dgemm.PushConstants = .{
            .M = @intCast(m),
            .K = @intCast(k),
            .N = @intCast(n),
            .alpha = 1.0,
            .beta = if (mode == .accumulate) 1.0 else 0.0,
        };

        // One thread per output element
        const threads: u32 = @intCast(m * n);
        const groups = std.math.divCeil(u32, threads, dgemm.WgSize.x) catch unreachable;

        d.dgemm_kernel.dispatch(.{
            .buffers = &.{ a_buf.raw(), b_buf.raw(), c_buf.raw() },
            .push_constant = std.mem.asBytes(&pc),
            .groups = .{ groups, 1, 1 },
        }) catch |err| return translate(err, "dispatching dgemm");

        // `dispatch` submits without blocking; the results are not there until
        // the fence clears.
        d.ctx.wait() catch |err| return translate(err, "waiting on dgemm");

        c_buf.copyToHost(c[0 .. m * n]);
    }
};

/// A device buffer that grows to fit whatever it is asked to hold.
const Scratch = struct {
    buf: ?spock.Buffer(f64) = null,

    fn ensure(s: *Scratch, ctx: *spock.Context, n: usize) Error!spock.Buffer(f64) {
        if (s.buf) |b| {
            if (b.len >= n) return b;
            b.deinit();
            s.buf = null;
        }
        s.buf = spock.Buffer(f64).create(ctx, n) catch |err| {
            return translate(err, "allocating a device buffer");
        };
        return s.buf.?;
    }

    fn deinit(s: *Scratch) void {
        if (s.buf) |b| b.deinit();
        s.buf = null;
    }
};

/// Collapse a Vulkan error into this module's set, reporting what was lost.
fn translate(err: anyerror, doing: []const u8) Error {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.NoComputeDevice, error.InitializationFailed => error.NoComputeDevice,
        else => {
            std.debug.print("gpu: {t} while {s}\n", .{ err, doing });
            return error.DeviceFailure;
        },
    };
}
