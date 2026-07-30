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
//! State of the port: the whole residual, for the inviscid Euler equations in
//! 2D -- eight dispatches recorded into one command buffer and submitted once.
//! Their operands are device-resident, so a dispatch binds them where they lie
//! rather than copying them in and out; `gemmHost` still exists for callers
//! holding ordinary host memory.
//!
//! What is left on the CPU: the Runge-Kutta update, the residual norms, and
//! anything the kernels do not cover -- advection-diffusion, the viscous terms,
//! and the viscous wall conditions -- each of which falls back rather than
//! failing.

const std = @import("std");
const spock = @import("spock");

const dgemm = spock.kernels.blas.dgemm;
const dgemm_spv = @embedFile("spock/dgemm.spv");

const flux_euler = @import("kernels/flux_euler.zig");
const flux_euler_spv = @embedFile("flux_euler.spv");

pub const face_scatter = @import("kernels/face_scatter.zig");
pub const face_gather = @import("kernels/face_gather.zig");
pub const face_common_f = @import("kernels/face_common_f.zig");
pub const face_bcs = @import("kernels/face_bcs.zig");

const face_scatter_spv = @embedFile("face_scatter.spv");
const face_gather_spv = @embedFile("face_gather.spv");
const face_common_f_spv = @embedFile("face_common_f.spv");
const face_bcs_spv = @embedFile("face_bcs.spv");

/// What the index arrays use for "no global flux point here". The host's
/// `geo.none` is `maxInt(usize)`; it narrows to this on upload.
pub const none_u32 = face_scatter.none;

pub const Error = error{
    /// No Vulkan loader, or no device on it with compute support
    NoComputeDevice,
    /// A batch tried to record the same kernel twice. Each kernel owns one
    /// descriptor set, so the second recording would overwrite the first's
    /// arguments and both dispatches would run with the second's -- silently.
    KernelAlreadyRecorded,
    /// The device rejected something, or a dispatch failed. The underlying
    /// Vulkan error is reported to stderr -- it is not in this set because
    /// hoisting the whole of `vk.Error` into every caller's signature buys
    /// nothing they could act on.
    DeviceFailure,
    OutOfMemory,
};

/// The kernels a `Device` owns. A batch may record each at most once.
const KernelId = enum { flux_euler, face_scatter, face_gather, face_common_f, face_bcs };

/// How many separate dgemm dispatches one batch may hold.
///
/// A kernel owns a single descriptor set, so each recording needs its own
/// instance. A residual has three: `extrapolateU` and the two divergence
/// products.
const max_gemms_per_batch = 4;

/// A device buffer handle, as a dispatch binds it.
pub const Buffer = spock.vk.Buffer;

/// One storage-buffer argument: a whole buffer, or a byte range of one.
/// `.whole(buf)` is the common case.
pub const Binding = spock.Kernel.Binding;

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
    /// One instance per dgemm recording in a batch; see `max_gemms_per_batch`.
    dgemm_kernels: [max_gemms_per_batch]spock.Kernel,
    /// How many of them the open batch has used.
    gemms_used: usize = 0,
    flux_euler_kernel: spock.Kernel,
    face_scatter_kernel: spock.Kernel,
    face_gather_kernel: spock.Kernel,
    face_common_f_kernel: spock.Kernel,
    face_bcs_kernel: spock.Kernel,

    /// Reused across steps; `beginBatch` resets it rather than allocating a new
    /// command buffer every residual.
    batch: spock.Pipeline,
    /// Null unless a batch is open. Holds which kernels it has already recorded.
    recorded: ?std.EnumSet(KernelId) = null,

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
        var gemms: [max_gemms_per_batch]spock.Kernel = undefined;
        var built: usize = 0;
        errdefer for (gemms[0..built]) |*k| k.deinit();
        while (built < gemms.len) : (built += 1) {
            gemms[built] = try makeKernel(ctx, dgemm_spv, "dgemm", 3, @sizeOf(dgemm.PushConstants));
        }

        var flux_kernel = spock.Kernel.create(ctx, .{
            .spirv = flux_euler_spv,
            .entry = "flux_euler",
            .buffers = 3,
            .push_constant_size = @sizeOf(flux_euler.PushConstants),
        }) catch |err| return translate(err, "building the flux_euler pipeline");

        errdefer flux_kernel.deinit();

        var scatter = try makeKernel(ctx, face_scatter_spv, "face_scatter", 4, @sizeOf(face_scatter.PushConstants));
        errdefer scatter.deinit();
        var gather = try makeKernel(ctx, face_gather_spv, "face_gather", 4, @sizeOf(face_gather.PushConstants));
        errdefer gather.deinit();
        var common_f = try makeKernel(ctx, face_common_f_spv, "face_common_f", 5, @sizeOf(face_common_f.PushConstants));
        errdefer common_f.deinit();
        var bcs = try makeKernel(ctx, face_bcs_spv, "face_bcs", 3, @sizeOf(face_bcs.PushConstants));
        errdefer bcs.deinit();

        const batch = spock.Pipeline.create(ctx) catch |err| {
            return translate(err, "creating the dispatch batch");
        };

        return .{
            .gpa = gpa,
            .ctx = ctx,
            .dgemm_kernels = gemms,
            .flux_euler_kernel = flux_kernel,
            .face_scatter_kernel = scatter,
            .face_gather_kernel = gather,
            .face_common_f_kernel = common_f,
            .face_bcs_kernel = bcs,
            .batch = batch,
        };
    }

    pub fn deinit(d: *Device) void {
        d.stage_a.deinit();
        d.stage_b.deinit();
        d.stage_c.deinit();
        d.batch.deinit();
        d.face_bcs_kernel.deinit();
        d.face_common_f_kernel.deinit();
        d.face_gather_kernel.deinit();
        d.face_scatter_kernel.deinit();
        d.flux_euler_kernel.deinit();
        for (&d.dgemm_kernels) |*k| k.deinit();
        d.ctx.deinit();
        d.gpa.destroy(d.ctx);
    }

    pub fn name(d: *const Device) []const u8 {
        return d.ctx.deviceName();
    }

    // ---- Batching ----

    /// Begin a run of dispatches to submit together.
    ///
    /// Each dispatch on its own costs a submit and a fence wait, and for
    /// matrices this small that round trip dwarfs the arithmetic. Recorded into
    /// one command buffer, a run of consecutive GPU work costs one round trip
    /// instead of one each, with a storage barrier between them so each sees the
    /// last one's writes.
    ///
    /// Only *consecutive* GPU work can go in one batch: anything the CPU has to
    /// touch in between ends it.
    pub fn beginBatch(d: *Device) Error!void {
        std.debug.assert(d.recorded == null); // a batch is already open
        d.batch.reset() catch |err| return translate(err, "resetting the batch");
        d.recorded = .empty;
        d.gemms_used = 0;
    }

    /// Submit everything recorded since `beginBatch` and wait for it.
    pub fn submitBatch(d: *Device) Error!void {
        std.debug.assert(d.recorded != null); // no batch is open
        d.recorded = null;
        d.batch.submit() catch |err| return translate(err, "submitting the batch");
        d.ctx.wait() catch |err| return translate(err, "waiting on the batch");
    }

    /// Record a dispatch into the open batch, or run it on its own if none is.
    ///
    /// A kernel has one descriptor set, so recording the same instance twice
    /// into one command buffer would leave both dispatches using the second set
    /// of arguments -- no error, a plausible wrong answer. This catches it;
    /// dgemm, which really is dispatched several times per residual, has an
    /// instance per recording instead.
    fn run(
        d: *Device,
        id: KernelId,
        kernel: *spock.Kernel,
        args: spock.Kernel.DispatchArgs,
        what: []const u8,
    ) Error!void {
        if (d.recorded) |*seen| {
            if (seen.contains(id)) return error.KernelAlreadyRecorded;
            seen.insert(id);
        }
        return d.runOn(kernel, args, what);
    }

    fn runOn(
        d: *Device,
        kernel: *spock.Kernel,
        args: spock.Kernel.DispatchArgs,
        what: []const u8,
    ) Error!void {
        if (d.recorded != null) {
            d.batch.addKernel(kernel, args) catch |err| return translate(err, what);
            return;
        }

        kernel.dispatch(args) catch |err| return translate(err, what);
        // `dispatch` submits without blocking; the results are not there until
        // the fence clears.
        d.ctx.wait() catch |err| return translate(err, what);
    }

    /// `C = A*B`, or `C += A*B`, over buffers that are already on the device:
    /// the GPU counterpart of the solver's `gemm`, with the same argument order.
    ///
    /// `A` is (m, k), `B` is (k, n), `C` is (m, n), all row-major and densely
    /// packed. Nothing is copied -- this is what allocating the solver's arrays
    /// through `Heap` buys.
    pub fn gemm(
        d: *Device,
        m: usize,
        n: usize,
        k: usize,
        a: Binding,
        b: Binding,
        c: Binding,
        mode: Mode,
    ) Error!void {
        if (m == 0 or n == 0 or k == 0) return;

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

        try d.runOn(try d.nextGemm(), .{
            .buffers = &.{ a, b, c },
            .push_constant = std.mem.asBytes(&pc),
            .groups = .{ groups, 1, 1 },
        }, "dispatching dgemm");
    }

    /// The next unused dgemm instance in the open batch, or the first one when
    /// there is no batch.
    fn nextGemm(d: *Device) Error!*spock.Kernel {
        if (d.recorded == null) return &d.dgemm_kernels[0];
        if (d.gemms_used >= d.dgemm_kernels.len) return error.KernelAlreadyRecorded;
        defer d.gemms_used += 1;
        return &d.dgemm_kernels[d.gemms_used];
    }

    /// Inviscid Euler flux at the solution points, in reference space.
    ///
    /// One thread per `(spt, ele)`; see `kernels/flux_euler.zig` for the layouts
    /// it assumes. 2D and inviscid only, matching the kernel.
    pub fn fluxEuler(
        d: *Device,
        n_spts: usize,
        n_eles: usize,
        n_vars: usize,
        gamma: f64,
        u_spts: Binding,
        inv_jaco: Binding,
        f_spts: Binding,
    ) Error!void {
        if (n_spts == 0 or n_eles == 0) return;

        const pc: flux_euler.PushConstants = .{
            .n_spts = @intCast(n_spts),
            .n_eles = @intCast(n_eles),
            .n_vars = @intCast(n_vars),
            .gamma = gamma,
        };
        const threads: u32 = @intCast(n_spts * n_eles);
        const groups = std.math.divCeil(u32, threads, flux_euler.WgSize.x) catch unreachable;

        try d.run(.flux_euler, &d.flux_euler_kernel, .{
            .buffers = &.{ u_spts, inv_jaco, f_spts },
            .push_constant = std.mem.asBytes(&pc),
            .groups = .{ groups, 1, 1 },
        }, "dispatching flux_euler");
    }

    // ---- The face path ----

    /// Each element's flux-point solution -> the shared face arrays.
    pub fn faceScatter(
        d: *Device,
        n_fpts: usize,
        n_eles: usize,
        n_vars: usize,
        n_gfpts: usize,
        u_fpts: Binding,
        fpt2gfpt: Binding,
        fpt2slot: Binding,
        faces_u: Binding,
    ) Error!void {
        if (n_fpts == 0 or n_eles == 0) return;
        const p: face_scatter.PushConstants = .{
            .n_fpts = @intCast(n_fpts),
            .n_eles = @intCast(n_eles),
            .n_vars = @intCast(n_vars),
            .n_gfpts = @intCast(n_gfpts),
        };
        try d.run(.face_scatter, &d.face_scatter_kernel, .{
            .buffers = &.{ u_fpts, fpt2gfpt, fpt2slot, faces_u },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(n_fpts * n_eles, face_scatter.WgSize.x), 1, 1 },
        }, "dispatching face_scatter");
    }

    /// The common normal flux -> each element's own flux-point array.
    pub fn faceGather(
        d: *Device,
        n_fpts: usize,
        n_eles: usize,
        n_vars: usize,
        n_gfpts: usize,
        faces_f: Binding,
        fpt2gfpt: Binding,
        fpt2slot: Binding,
        f_comm: Binding,
    ) Error!void {
        if (n_fpts == 0 or n_eles == 0) return;
        const p: face_gather.PushConstants = .{
            .n_fpts = @intCast(n_fpts),
            .n_eles = @intCast(n_eles),
            .n_vars = @intCast(n_vars),
            .n_gfpts = @intCast(n_gfpts),
        };
        try d.run(.face_gather, &d.face_gather_kernel, .{
            .buffers = &.{ faces_f, fpt2gfpt, fpt2slot, f_comm },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(n_fpts * n_eles, face_gather.WgSize.x), 1, 1 },
        }, "dispatching face_gather");
    }

    /// Rusanov common normal flux at every global flux point.
    pub fn faceCommonF(
        d: *Device,
        n_gfpts: usize,
        n_vars: usize,
        gamma: f64,
        rus_k: f64,
        u: Binding,
        norm: Binding,
        d_a: Binding,
        f_comm: Binding,
        wave_sp: Binding,
    ) Error!void {
        if (n_gfpts == 0) return;
        const p: face_common_f.PushConstants = .{
            .n_gfpts = @intCast(n_gfpts),
            .n_vars = @intCast(n_vars),
            .gamma = gamma,
            .rus_k = rus_k,
        };
        try d.run(.face_common_f, &d.face_common_f_kernel, .{
            .buffers = &.{ u, norm, d_a, f_comm, wave_sp },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(n_gfpts, face_common_f.WgSize.x), 1, 1 },
        }, "dispatching face_common_f");
    }

    /// Ghost states at the boundary flux points.
    pub fn faceBcs(
        d: *Device,
        p: face_bcs.PushConstants,
        u: Binding,
        norm: Binding,
        bc_code: Binding,
    ) Error!void {
        if (p.n_gfpts_bnd == 0) return;
        try d.run(.face_bcs, &d.face_bcs_kernel, .{
            .buffers = &.{ u, norm, bc_code },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(p.n_gfpts_bnd, face_bcs.WgSize.x), 1, 1 },
        }, "dispatching face_bcs");
    }

    /// The same product over ordinary host slices, staged in and out around the
    /// dispatch.
    ///
    /// Only for callers whose data is not already device-resident; the solver's
    /// own arrays are, and use `gemm`. The staging buffers are grown on demand
    /// and reused.
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

        try d.gemm(m, n, k, .whole(a_buf.raw()), .whole(b_buf.raw()), .whole(c_buf.raw()), mode);

        c_buf.copyToHost(c[0 .. m * n]);
    }
};

fn makeKernel(
    ctx: *spock.Context,
    spirv: []const u8,
    entry: [*:0]const u8,
    buffers: u32,
    pc_size: u32,
) Error!spock.Kernel {
    return spock.Kernel.create(ctx, .{
        .spirv = spirv,
        .entry = entry,
        .buffers = buffers,
        .push_constant_size = pc_size,
    }) catch |err| translate(err, "building a compute pipeline");
}

fn groupsFor(threads: usize, wg: u32) u32 {
    return std.math.divCeil(u32, @intCast(threads), wg) catch unreachable;
}

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

/// A `std.mem.Allocator` whose allocations live in device-visible memory.
///
/// The memory is host-coherent and persistently mapped, so what comes back is an
/// ordinary slice: operations still running on the CPU keep writing it directly
/// while ported ones read the same bytes on the device. That is the whole point
/// -- it is what lets the port proceed one operator at a time instead of all at
/// once.
///
/// Every allocation is its own Vulkan buffer, because spock binds descriptors
/// with `offset = 0, range = WHOLE_SIZE` and gives no way to point at part of
/// one. That makes this the wrong tool for many small allocations and the right
/// one for the handful of large solver arrays it holds.
///
/// Watch the memory type this lands in. spock originally took the first
/// `host_visible | host_coherent` type, which on this hardware is *uncached*
/// (write-combined): CPU reads from it measured 25x slower than from an ordinary
/// allocation, and making one operator's operands resident cost the whole step
/// 8x. spock now prefers a cached type, which brings host reads back to parity
/// (36ms against 34ms on the same measurement) -- but a device offering no
/// cached host-visible heap would still pay it.
pub const Heap = struct {
    dev: *Device,
    /// For the bookkeeping list only; the blocks themselves are device memory
    gpa: std.mem.Allocator,
    blocks: std.ArrayList(Block),

    const Block = struct {
        ptr: [*]u8,
        len: usize,
        buf: spock.Buffer(u8),
    };

    pub fn init(dev: *Device, gpa: std.mem.Allocator) Heap {
        return .{ .dev = dev, .gpa = gpa, .blocks = .empty };
    }

    pub fn deinit(h: *Heap) void {
        for (h.blocks.items) |b| b.buf.deinit();
        h.blocks.deinit(h.gpa);
    }

    pub fn allocator(h: *Heap) std.mem.Allocator {
        return .{
            .ptr = h,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    /// The buffer holding `ptr`, or null if it did not come from here.
    ///
    /// A linear scan over a handful of blocks, and only when a dispatch is being
    /// recorded, so there is nothing to gain from an index.
    pub fn bufferFor(h: *const Heap, ptr: *const anyopaque) ?spock.vk.Buffer {
        const addr = @intFromPtr(ptr);
        for (h.blocks.items) |b| {
            const base = @intFromPtr(b.ptr);
            if (addr >= base and addr < base + b.len) return b.buf.raw();
        }
        return null;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const h: *Heap = @ptrCast(@alignCast(ctx));

        h.blocks.ensureUnusedCapacity(h.gpa, 1) catch return null;
        const buf = spock.Buffer(u8).create(h.dev.ctx, len) catch return null;

        // A mapped Vulkan allocation starts at the base of its own memory
        // object, so it is aligned far past anything an f64 array asks for --
        // but say so rather than assume it.
        const ptr = @as([*]u8, @ptrCast(buf.ptr));
        if (!alignment.check(@intFromPtr(ptr))) {
            buf.deinit();
            return null;
        }

        h.blocks.appendAssumeCapacity(.{ .ptr = ptr, .len = len, .buf = buf });
        return ptr;
    }

    /// Never in place: a block is a whole Vulkan buffer, and growing one means
    /// a new allocation and a new descriptor. The solver never resizes these.
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn free(ctx: *anyopaque, memory: []u8, _: std.mem.Alignment, _: usize) void {
        const h: *Heap = @ptrCast(@alignCast(ctx));
        for (h.blocks.items, 0..) |b, i| {
            if (b.ptr != memory.ptr) continue;
            b.buf.deinit();
            _ = h.blocks.swapRemove(i);
            return;
        }
        unreachable; // freed something this heap never handed out
    }
};

/// A standalone device buffer holding a copy of host data.
///
/// For the reference-element operators: they are constant for the whole run and
/// tiny -- an order-3 quad's `oppE` is 16x16 -- so uploading a copy at setup is
/// cheaper than routing `Element`'s allocations through a `Heap` would be
/// invasive.
pub const Array = struct {
    buf: spock.Buffer(f64),

    pub fn upload(d: *Device, src: []const f64) Error!Array {
        const buf = spock.Buffer(f64).create(d.ctx, src.len) catch |err| {
            return translate(err, "allocating a device array");
        };
        buf.copyFromHost(src);
        return .{ .buf = buf };
    }

    pub fn raw(a: Array) spock.vk.Buffer {
        return a.buf.raw();
    }

    pub fn binding(a: Array) Binding {
        return .whole(a.buf.raw());
    }

    pub fn deinit(a: *Array) void {
        a.buf.deinit();
    }
};

/// The integer counterpart of `Array`, for the connectivity a face kernel
/// walks: `fpt2gfpt`, its slot table, and the per-point boundary codes.
pub const IndexArray = struct {
    buf: spock.Buffer(u32),

    pub fn upload(d: *Device, src: []const u32) Error!IndexArray {
        const buf = spock.Buffer(u32).create(d.ctx, @max(src.len, 1)) catch |err| {
            return translate(err, "allocating a device index array");
        };
        buf.copyFromHost(src);
        return .{ .buf = buf };
    }

    pub fn binding(a: IndexArray) Binding {
        return .whole(a.buf.raw());
    }

    pub fn deinit(a: *IndexArray) void {
        a.buf.deinit();
    }
};
